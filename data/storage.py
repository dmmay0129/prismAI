"""数据管理层：SQLite 持久化评估结果、完整报告与按用户隔离的历史。

原始音视频不入库；仅保存结构化分析结果和完整报告 JSON。
``user_id`` 对应登录用户名；首次登录时可把升级前的旧记录认领给该用户。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

from core.settings import get_settings

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "prism.db")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS results (
               session_id  TEXT PRIMARY KEY,
               category    TEXT,
               overall     REAL,
               dims        TEXT,
               radar       TEXT,
               created_at  REAL,
               user_id     TEXT,
               report_json TEXT
           )"""
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    if "user_id" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN user_id TEXT")
    if "report_json" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN report_json TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_user_created "
        "ON results(user_id, created_at)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS knowledge_base (
               user_id TEXT PRIMARY KEY COLLATE NOCASE,
               payload TEXT NOT NULL,
               updated_at REAL NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS interview_turns (
               session_id TEXT NOT NULL,
               turn_index INTEGER NOT NULL,
               user_id TEXT NOT NULL COLLATE NOCASE,
               question TEXT NOT NULL,
               transcript TEXT NOT NULL,
               overall REAL NOT NULL,
               report_json TEXT NOT NULL,
               created_at REAL NOT NULL,
               PRIMARY KEY(session_id, turn_index)
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_interview_turns_user_created "
        "ON interview_turns(user_id, created_at)"
    )


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    _ensure_schema(conn)
    return conn


def save_result(report, username: str | None = None) -> None:
    """保存评估结果和完整报告；username 为空时保留为历史遗留记录。"""
    report_data = report.to_dict()
    report_data.pop("transcript", None)
    payload = json.dumps(report_data, ensure_ascii=False)
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO results "
            "(session_id, category, overall, dims, radar, created_at, "
            " user_id, report_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                report.session_id,
                report.question.get("category", ""),
                report.overall_score,
                json.dumps(report.dimension_scores, ensure_ascii=False),
                json.dumps(report.radar, ensure_ascii=False),
                report.created_at,
                username,
                payload,
            ),
        )


def load_history(limit: int | None = None, username: str | None = None) -> list:
    """历史趋势数据：按时间正序返回近期练习记录；有用户时只返回该用户。"""
    if limit is None:
        limit = int(get_settings()["storage"]["history_limit"])
    if not os.path.exists(_DB_PATH):
        return []
    with _conn() as conn:
        if username:
            rows = conn.execute(
                "SELECT session_id, overall, dims, created_at FROM results "
                "WHERE user_id COLLATE NOCASE = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id, overall, dims, created_at FROM results "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    history = []
    for session_id, overall, dims, created_at in reversed(rows):
        try:
            parsed_dims = json.loads(dims)
        except (TypeError, json.JSONDecodeError):
            parsed_dims = {}
        history.append(
            {
                "session_id": session_id,
                "overall": overall,
                "dims": parsed_dims,
                "created_at": created_at,
            }
        )
    return history


def claim_legacy_results(username: str) -> int:
    """把升级前没有 user_id 的记录认领给当前用户（首次登录时调用）。"""
    if not username:
        return 0
    with _conn() as conn:
        cursor = conn.execute(
            "UPDATE results SET user_id = ? WHERE user_id IS NULL",
            (username,),
        )
        return cursor.rowcount


def load_report(session_id: str, username: str | None = None) -> dict | None:
    """读取完整报告 JSON；用于历史详情等扩展。"""
    query = "SELECT report_json FROM results WHERE session_id = ?"
    params: list = [session_id]
    if username:
        query += " AND user_id COLLATE NOCASE = ?"
        params.append(username)
    with _conn() as conn:
        row = conn.execute(query, params).fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None


def load_knowledge(username: str) -> dict:
    """读取按用户隔离的面试知识库。"""
    if not username or not os.path.exists(_DB_PATH):
        return {}
    with _conn() as conn:
        row = conn.execute(
            "SELECT payload FROM knowledge_base WHERE user_id COLLATE NOCASE = ?",
            (username,),
        ).fetchone()
    if not row:
        return {}
    try:
        payload = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_knowledge(username: str, payload: dict) -> None:
    """持久化候选人背景、项目证据、岗位要求和追问风格。"""
    if not username:
        raise ValueError("保存知识库需要已登录用户")
    fields = ("candidate_profile", "project_evidence", "job_description",
              "interviewer_style", "interview_review")
    normalized = {key: str(payload.get(key, "")).strip()[:6000] for key in fields}
    encoded = json.dumps(normalized, ensure_ascii=False)
    with _conn() as conn:
        conn.execute(
            "INSERT INTO knowledge_base(user_id, payload, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload, "
            "updated_at=excluded.updated_at",
            (username, encoded, time.time()),
        )


def save_interview_turn(
    session_id: str, turn_index: int, username: str, question: str,
    transcript: str, report,
) -> None:
    """保存问答文本和评分，不保存原始音视频。"""
    if not username or not session_id:
        raise ValueError("保存面试记录需要用户和会话标识")
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO interview_turns "
            "(session_id, turn_index, user_id, question, transcript, overall, "
            " report_json, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                session_id, int(turn_index), username, question[:1000],
                transcript[:12000], report.overall_score,
                json.dumps(report.to_dict(), ensure_ascii=False), time.time(),
            ),
        )


def list_interview_sessions(username: str, limit: int = 20) -> list[dict]:
    """返回当前用户最近的多轮模拟面试列表。"""
    if not username or not os.path.exists(_DB_PATH):
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT session_id, COUNT(*), MAX(created_at), "
            "ROUND(AVG(overall), 1) FROM interview_turns "
            "WHERE user_id COLLATE NOCASE = ? GROUP BY session_id "
            "ORDER BY MAX(created_at) DESC LIMIT ?",
            (username, max(1, int(limit))),
        ).fetchall()
    return [
        {"session_id": row[0], "turn_count": row[1],
         "created_at": row[2], "average_score": row[3]}
        for row in rows
    ]


def load_interview_turns(session_id: str, username: str) -> list[dict]:
    """按用户权限读取一场模拟面试的问答与报告。"""
    if not session_id or not username or not os.path.exists(_DB_PATH):
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT turn_index, question, transcript, overall, report_json, created_at "
            "FROM interview_turns WHERE session_id = ? "
            "AND user_id COLLATE NOCASE = ? ORDER BY turn_index",
            (session_id, username),
        ).fetchall()
    turns = []
    for row in rows:
        try:
            report = json.loads(row[4])
        except (TypeError, json.JSONDecodeError):
            report = {}
        turns.append({
            "turn_index": row[0], "question": row[1], "transcript": row[2],
            "overall": row[3], "report": report, "created_at": row[5],
        })
    return turns
