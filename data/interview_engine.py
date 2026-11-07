"""基于回答和用户资料生成下一轮面试追问（提示词来自 data/prompts.py）。"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.settings import get_settings
from data.prompts import follow_up_prompt

logger = logging.getLogger(__name__)

_CATEGORY_FOLLOW_UPS = {
    "行为类": "你在这件事中的个人职责是什么？请用一个可衡量的结果说明。",
    "技术类": "你如何验证这个技术方案有效？它的主要边界或取舍是什么？",
    "情景类": "你会用什么指标判断方案有效？如果结果不达预期会怎么调整？",
    "项目深挖": "你个人负责的关键决策是什么？最终结果如何量化？",
    "岗位业务": "你会用哪些业务数据验证这个判断？",
    "压力追问": "什么证据会让你改变当前方案？",
    "管理协作": "你如何确认各方达成一致，并追踪最终结果？",
    "动机匹配": "你会用哪些具体经历或标准验证这种匹配？",
}

# 评分指标未命中 → 对应的专业追问（与 data/rubric.py 指标 ID 一一对应）。
# 追问直接瞄准上一轮回答缺失的证据点，比按类别轮换的泛化追问专业得多。
_MISS_FOLLOWUPS = {
    "L1": "刚才的回答里，哪一句是直接回应题目所问的？请先把它说清楚。",
    "L2": "如果用一句话总结你的核心观点，你会怎么说？",
    "L3": "你做的这个选择，背后的关键依据是什么？为什么相信它有效？",
    "L4": "请用「首先、其次、最后」把论证过程重新梳理一遍。",
    "L5": "一句话收尾：你的最终结论是什么？",
    "D1": "你提到的结果有没有具体数字可以佐证？比如提升比例、耗时或规模。",
    "D2": "其中哪些部分由你个人负责？请区分个人贡献与团队成果。",
    "D3": "先介绍一下当时的项目背景：团队规模、周期和你承担的角色。",
    "D4": "具体用了哪些技术、工具或方法？可以展开讲一个关键细节。",
    "D5": "当时还有哪些备选方案？为什么选了这一个、放弃了什么？",
    "S1": "开头先总起一句：你接下来会从哪几个方面展开？",
    "S3": "从发现问题到最终落地，中间最关键的两步是什么？",
    "S4": "这件事最后的结果怎么样？是否达到了预期？",
    "A1": "回到题目本身：它最想考察的是什么？你的回答扣住这一点了吗？",
    "A3": "你引用的数据是从哪里来的？如果无法核验，请说明来源或范围。",
    "A4": "这个结论有没有边界条件？在什么情况下可能不成立？",
    "A5": "这个判断里，哪些部分你比较确定、哪些还只是推测？",
}

# 追问优先级：深度/逻辑缺口最值得挖，其次准确度与结构；表达类不适合做追问
_DIM_ORDER = {"depth": 0, "logic": 1, "accuracy": 2, "structure": 3}


def _miss_followups(report) -> list[dict]:
    """从报告评分清单里提取未命中指标对应的追问（按维度优先级排序）。"""
    out: list[dict] = []
    if report is None:
        return out
    details = getattr(report, "scoring_details", None) or {}
    checklist = details.get("checklist") or {}
    if not isinstance(checklist, dict):
        return out
    for dim in sorted(checklist.keys(), key=lambda d: _DIM_ORDER.get(d, 9)):
        for item in checklist[dim].get("missed", []):
            q = _MISS_FOLLOWUPS.get(item.get("id"))
            if q:
                out.append({
                    "question": q,
                    "target": item.get("name", ""),
                    "id": item.get("id", ""),
                })
    return out


def _fallback_question(
    question: dict, transcript: str, report=None, used: set[str] | None = None
) -> str:
    """规则降级追问：优先瞄准评分缺口，已问过的问题不再重复。"""
    used = {str(u).strip() for u in (used or ()) if u and str(u).strip()}
    # 1) 评分缺口驱动：哪个指标没命中就问哪个（专业、有回答依据）
    for cand in _miss_followups(report):
        if cand["question"] not in used:
            return cand["question"]
    # 2) 回答内容启发式（旧逻辑保留为后备）
    heuristics = []
    if not any(char.isdigit() for char in transcript):
        heuristics.append("你提到的结果能用什么数据或具体事实来验证？")
    if not any(word in transcript for word in ("我负责", "我主导", "我设计", "我推动")):
        heuristics.append("其中哪些部分由你亲自负责？请区分个人贡献与团队成果。")
    heuristics.append(
        _CATEGORY_FOLLOW_UPS.get(
            str(question.get("category") or ""), "你如何验证这个方案的结果？"
        )
    )
    for q in heuristics:
        if q not in used:
            return q
    # 3) 全部用尽 → 换角度深挖
    return "如果重新做一次，你会在哪个环节做得不一样？"


def _parse_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def generate_follow_up(
    question: dict,
    transcript: str,
    candidate_context: dict[str, str] | None = None,
    prior_turns: list[dict[str, str]] | None = None,
    report=None,
) -> dict[str, str]:
    """生成一条有回答依据的追问；本地模型不可用时返回明确的模板降级。

    report: 本轮评分报告。规则降级会瞄准评分清单里未命中的指标提问；
            已问过的问题（前几轮题目 + 追问）一律不再重复。
    """
    used = {str(question.get("content", "")).strip()}
    used |= {str(t.get("question", "")).strip() for t in (prior_turns or [])}
    used.discard("")
    fallback = _fallback_question(question, transcript, report=report, used=used)
    if not transcript.strip():
        return {"question": fallback, "source": "规则降级"}

    context = candidate_context or {}
    history = [
        {"question": turn.get("question", "")[:300],
         "answer": (turn.get("answer") or turn.get("transcript", ""))[:600]}
        for turn in (prior_turns or [])[-3:]
    ]
    missed_names = [c["target"] for c in _miss_followups(report)[:6]]
    prompt = follow_up_prompt(
        str(question.get("content", "")), transcript, context, history, missed_names
    )
    try:
        import requests

        cfg = get_settings()["content"]
        response = requests.post(
            cfg["ollama_url"],
            json={
                "model": cfg["model"],
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=float(cfg["timeout_sec"]),
        )
        response.raise_for_status()
        payload = _parse_object(str(response.json().get("response", "")))
        text = str(payload.get("question", "")).strip() if payload else ""
        valid = bool(
            text
            and len(text) <= 240
            and ("?" in text or "？" in text)
            and text not in used
        )
        if valid:
            return {"question": text, "source": "本地 AI"}
        logger.info("本地模型未返回有效追问，使用规则追问")
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        logger.info("本地追问模型不可用，使用规则追问: %s", exc)
    return {"question": fallback, "source": "规则降级"}
