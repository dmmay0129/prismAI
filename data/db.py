"""统一数据库访问层：云端 Postgres / 本地 SQLite 自动切换。

后端选择（模块导入时确定）：
- 配置了 ``DATABASE_URL`` / ``PRISMAI_DATABASE_URL`` 环境变量，或 Streamlit Secrets
  里有 ``DATABASE_URL``（或 ``postgres.url``）→ 使用 Postgres（psycopg3），用于云端持久化；
- 否则回退到本地 SQLite（项目根目录 ``prism.db``），保持本地开发行为不变。

storage / auth 只依赖这里的 ``get_conn``、``ph``、``PH``、``BACKEND`` 与
``IntegrityError``，不再直接 import sqlite3 / psycopg。
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SQLITE_PATH = os.path.join(_PROJECT_ROOT, "prism.db")


def _database_url() -> str | None:
    for name in ("DATABASE_URL", "PRISMAI_DATABASE_URL"):
        value = os.getenv(name)
        if value:
            return value
    try:
        import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", None) or {}
        url = secrets.get("DATABASE_URL")
        if not url:
            postgres = secrets.get("postgres")
            if isinstance(postgres, dict):
                url = postgres.get("url")
        if url:
            return str(url)
    except Exception:
        pass
    return None


_DATABASE_URL = _database_url()
BACKEND = "postgres" if _DATABASE_URL else "sqlite"
# 参数占位符：Postgres 用 %s，SQLite 用 ?
PH = "%s" if BACKEND == "postgres" else "?"

if BACKEND == "postgres":
    import psycopg
    from psycopg.errors import UniqueViolation as IntegrityError
else:
    IntegrityError = sqlite3.IntegrityError  # type: ignore


def ph(count: int) -> str:
    """返回 count 个占位符拼接，如 ``"%s, %s, %s"`` 或 ``"?, ?, ?"``。"""
    return ", ".join([PH] * count)


def connect() -> Any:
    """返回一个原始连接（未建表）；调用方用 ``with connect() as conn`` 提交/回滚。"""
    if BACKEND == "postgres":
        return psycopg.connect(_DATABASE_URL)
    conn = sqlite3.connect(_SQLITE_PATH, timeout=10)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _ensure_all_schema(conn: Any) -> None:
    """幂等建表；用户名在 Postgres 下用小写函数唯一索引实现大小写不敏感。"""
    if BACKEND == "postgres":
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username        TEXT PRIMARY KEY,
                password_hash   TEXT NOT NULL,
                salt            TEXT NOT NULL,
                created_at      DOUBLE PRECISION NOT NULL,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until    DOUBLE PRECISION NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower_uidx "
            "ON users (LOWER(username))"
        )
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username        TEXT PRIMARY KEY COLLATE NOCASE,
                password_hash   TEXT NOT NULL,
                salt            TEXT NOT NULL,
                created_at      DOUBLE PRECISION NOT NULL,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until    DOUBLE PRECISION NOT NULL DEFAULT 0
            )
            """
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            session_id   TEXT PRIMARY KEY,
            category     TEXT,
            overall      DOUBLE PRECISION,
            dims         TEXT,
            radar        TEXT,
            created_at   DOUBLE PRECISION,
            user_id      TEXT,
            report_json  TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_user_created "
        "ON results(user_id, created_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_base (
            user_id     TEXT PRIMARY KEY,
            payload     TEXT NOT NULL,
            updated_at  DOUBLE PRECISION NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interview_turns (
            session_id   TEXT NOT NULL,
            turn_index   INTEGER NOT NULL,
            user_id      TEXT NOT NULL,
            question     TEXT NOT NULL,
            transcript   TEXT NOT NULL,
            overall      DOUBLE PRECISION NOT NULL,
            report_json  TEXT NOT NULL,
            created_at   DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (session_id, turn_index)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_interview_turns_user_created "
        "ON interview_turns(user_id, created_at)"
    )


_schema_ready = False


def get_conn() -> Any:
    """返回一个已建好表、可用的连接。

    首次调用会用一个短连接建表（幂等），之后每次返回新连接。
    调用方请用 ``with get_conn() as conn:``（Postgres 下会自动提交并关闭连接）。
    """
    global _schema_ready
    if not _schema_ready:
        with connect() as conn:
            _ensure_all_schema(conn)
        _schema_ready = True
    return connect()
