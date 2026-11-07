"""数据管理层：持久化评估结果、完整报告与按用户隔离的历史。

原始音视频不入库；仅保存结构化分析结果和完整报告 JSON。
``user_id`` 对应登录用户名（与登录态大小写一致）；首次登录时把升级前的旧记录认领给该用户。

数据库后端见 ``data.db``：云端用 Postgres，本地无 DATABASE_URL 时回退 SQLite。
"""
from __future__ import annotations

import json
import time

from core.settings import get_settings
from data.db import BACKEND, PH, get_conn, ph


def save_result(report, username: str | None = None) -> None:
    """保存评估结果和完整报告；username 为空时保留为历史遗留记录。"""
    report_data = report.to_dict()
    report_data.pop("transcript", None)
    payload = json.dumps(report_data, ensure_ascii=False)
    cols = ("session_id", "category", "overall", "dims", "radar",
            "created_at", "user_id", "report_json")
    if BACKEND == "postgres":
        sql = (
            f"INSERT INTO results ({', '.join(cols)}) VALUES ({ph(8)}) "
            "ON CONFLICT (session_id) DO UPDATE SET "
            "category = EXCLUDED.category, overall = EXCLUDED.overall, "
            "dims = EXCLUDED.dims, radar = EXCLUDED.radar, "
            "created_at = EXCLUDED.created_at, user_id = EXCLUDED.user_id, "
            "report_json = EXCLUDED.report_json"
        )
    else:
        sql = f"INSERT OR REPLACE INTO results ({', '.join(cols)}) VALUES ({ph(8)})"
    with get_conn() as conn:
        conn.execute(
            sql,
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
    query = "SELECT session_id, overall, dims, created_at FROM results "
    params: list = []
    if username:
        query += f"WHERE user_id = {PH} "
        params.append(username)
    query += f"ORDER BY created_at DESC LIMIT {PH}"
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()

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
    with get_conn() as conn:
        cursor = conn.execute(
            f"UPDATE results SET user_id = {PH} WHERE user_id IS NULL",
            (username,),
        )
        return int(cursor.rowcount or 0)


def load_report(session_id: str, username: str | None = None) -> dict | None:
    """读取完整报告 JSON；用于历史详情等扩展。"""
    query = f"SELECT report_json FROM results WHERE session_id = {PH}"
    params: list = [session_id]
    if username:
        query += f" AND user_id = {PH}"
        params.append(username)
    with get_conn() as conn:
        row = conn.execute(query, tuple(params)).fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None


def load_knowledge(username: str) -> dict:
    """读取按用户隔离的面试知识库。"""
    if not username:
        return {}
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT payload FROM knowledge_base WHERE user_id = {PH}",
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
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO knowledge_base(user_id, payload, updated_at) "
            f"VALUES({ph(3)}) ON CONFLICT(user_id) DO UPDATE SET "
            f"payload = EXCLUDED.payload, updated_at = EXCLUDED.updated_at",
            (username, encoded, time.time()),
        )


def save_interview_turn(
    session_id: str, turn_index: int, username: str, question: str,
    transcript: str, report,
) -> None:
    """保存问答文本和评分，不保存原始音视频。"""
    if not username or not session_id:
        raise ValueError("保存面试记录需要用户和会话标识")
    cols = ("session_id", "turn_index", "user_id", "question", "transcript",
            "overall", "report_json", "created_at")
    if BACKEND == "postgres":
        sql = (
            f"INSERT INTO interview_turns ({', '.join(cols)}) VALUES ({ph(8)}) "
            "ON CONFLICT (session_id, turn_index) DO UPDATE SET "
            "user_id = EXCLUDED.user_id, question = EXCLUDED.question, "
            "transcript = EXCLUDED.transcript, overall = EXCLUDED.overall, "
            "report_json = EXCLUDED.report_json, created_at = EXCLUDED.created_at"
        )
    else:
        sql = (
            f"INSERT OR REPLACE INTO interview_turns ({', '.join(cols)}) "
            f"VALUES ({ph(8)})"
        )
    with get_conn() as conn:
        conn.execute(
            sql,
            (
                session_id, int(turn_index), username, question[:1000],
                transcript[:12000], report.overall_score,
                json.dumps(report.to_dict(), ensure_ascii=False), time.time(),
            ),
        )


def list_interview_sessions(username: str, limit: int = 20) -> list[dict]:
    """返回当前用户最近的多轮模拟面试列表。"""
    if not username:
        return []
    query = (
        "SELECT session_id, COUNT(*), MAX(created_at), AVG(overall) "
        "FROM interview_turns "
        f"WHERE user_id = {PH} GROUP BY session_id "
        f"ORDER BY MAX(created_at) DESC LIMIT {PH}"
    )
    with get_conn() as conn:
        rows = conn.execute(query, (username, max(1, int(limit)))).fetchall()
    return [
        {"session_id": row[0], "turn_count": row[1],
         "created_at": row[2], "average_score": round(float(row[3]), 1)}
        for row in rows
    ]


def load_interview_turns(session_id: str, username: str) -> list[dict]:
    """按用户权限读取一场模拟面试的问答与报告。"""
    if not session_id or not username:
        return []
    query = (
        "SELECT turn_index, question, transcript, overall, report_json, created_at "
        f"FROM interview_turns WHERE session_id = {PH} AND user_id = {PH} "
        "ORDER BY turn_index"
    )
    with get_conn() as conn:
        rows = conn.execute(query, (session_id, username)).fetchall()
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
