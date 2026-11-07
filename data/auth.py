"""本地用户认证（MVP）。

- 用户保存在 `prism.db` 的 `users` 表；
- 密码使用 PBKDF2-HMAC-SHA256 + 随机盐，不保存明文；
- 仅用于本地访问控制，不替代生产级认证（无找回密码/HTTPS；含失败锁定）。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import time
from dataclasses import dataclass

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "prism.db")
_ITERATIONS = 200_000
_USERNAME_MIN = 2
_USERNAME_MAX = 32
_PASSWORD_MIN = 8
_PASSWORD_MAX = 128
_MAX_ATTEMPTS = 5
_LOCK_SECONDS = 300


class AuthError(ValueError):
    """用户名/密码不合法或用户名已存在。"""


@dataclass(frozen=True)
class AuthResult:
    ok: bool
    username: str | None = None
    message: str = ""
    locked_until: float = 0.0


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               username        TEXT PRIMARY KEY COLLATE NOCASE,
               password_hash   TEXT NOT NULL,
               salt            TEXT NOT NULL,
               created_at      REAL NOT NULL,
               failed_attempts INTEGER NOT NULL DEFAULT 0,
               locked_until    REAL NOT NULL DEFAULT 0
           )"""
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    if "failed_attempts" not in cols:
        conn.execute(
            "ALTER TABLE users ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0"
        )
    if "locked_until" not in cols:
        conn.execute(
            "ALTER TABLE users ADD COLUMN locked_until REAL NOT NULL DEFAULT 0"
        )


def _conn(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or _DB_PATH, timeout=10)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    _ensure_schema(conn)
    return conn


def _hash(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        _ITERATIONS,
    ).hex()


def _clean_username(username: str) -> str:
    value = (username or "").strip()
    if not (_USERNAME_MIN <= len(value) <= _USERNAME_MAX):
        raise AuthError(f"用户名长度需为 {_USERNAME_MIN}-{_USERNAME_MAX} 个字符")
    if any(ch.isspace() for ch in value):
        raise AuthError("用户名不能包含空格")
    return value


def _check_password(password: str) -> None:
    if not password or len(password) < _PASSWORD_MIN:
        raise AuthError(f"密码至少需要 {_PASSWORD_MIN} 位")
    if len(password) > _PASSWORD_MAX:
        raise AuthError("密码过长")


def create_user(username: str, password: str, db_path: str | None = None) -> str:
    """创建用户；成功返回清理后的用户名，用户名重复时抛 AuthError。"""
    name = _clean_username(username)
    _check_password(password)
    salt = os.urandom(16).hex()
    pwd_hash = _hash(password, salt)
    try:
        with _conn(db_path) as conn:
            conn.execute(
                "INSERT INTO users(username, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (name, pwd_hash, salt, time.time()),
            )
    except sqlite3.IntegrityError as exc:
        raise AuthError("用户名已存在") from exc
    return name


def verify_user(username: str, password: str, db_path: str | None = None) -> bool:
    """校验用户名和密码；空值、用户不存在或密码错误均返回 False。"""
    name = (username or "").strip()
    if not name or not password:
        return False
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?",
            (name,),
        ).fetchone()
    if row is None:
        return False
    return hmac.compare_digest(_hash(password, row[1]), row[0])


def authenticate(
    username: str, password: str, db_path: str | None = None
) -> AuthResult:
    """带失败次数/锁定策略的登录校验。

    连续失败 _MAX_ATTEMPTS 次后锁定 _LOCK_SECONDS 秒；锁定期间即使密码正确
    也会拒绝，避免暴力尝试。成功登录会清零失败计数。
    """
    name = (username or "").strip()
    if not name or not password:
        return AuthResult(False, None, "请输入用户名和密码")

    now = time.time()
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT username, password_hash, salt, failed_attempts, locked_until "
            "FROM users WHERE username = ?",
            (name,),
        ).fetchone()
        if row is None:
            return AuthResult(False, None, "用户名或密码错误")

        stored_name, pwd_hash, salt, failed_attempts, locked_until = row
        failed_attempts = int(failed_attempts or 0)
        locked_until = float(locked_until or 0.0)

        if locked_until > now:
            remaining = max(1, int(locked_until - now))
            return AuthResult(
                False, stored_name,
                f"登录尝试过多，请 {remaining} 秒后重试", locked_until,
            )

        # 锁定期已过，重新开始计数
        if locked_until and locked_until <= now:
            failed_attempts = 0

        if hmac.compare_digest(_hash(password, salt), pwd_hash):
            conn.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = 0 "
                "WHERE username = ?",
                (stored_name,),
            )
            return AuthResult(True, stored_name, "登录成功")

        failed_attempts += 1
        if failed_attempts >= _MAX_ATTEMPTS:
            locked_until = now + _LOCK_SECONDS
            conn.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? "
                "WHERE username = ?",
                (failed_attempts, locked_until, stored_name),
            )
            return AuthResult(
                False, stored_name,
                f"登录尝试过多，请 {_LOCK_SECONDS // 60} 分钟后重试",
                locked_until,
            )

        conn.execute(
            "UPDATE users SET failed_attempts = ? WHERE username = ?",
            (failed_attempts, stored_name),
        )
        return AuthResult(
            False, stored_name,
            f"用户名或密码错误，还可尝试 { _MAX_ATTEMPTS - failed_attempts } 次",
        )


def has_users(db_path: str | None = None) -> bool:
    """是否已有注册用户（用于首次使用提示）。"""
    with _conn(db_path) as conn:
        return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None
