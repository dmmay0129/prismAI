"""内容分析过滤器（策略模式具体策略之二，rubric 2.0）。

评分量化：五维各 5 个可观察指标（data/rubric.py），每项 20 分。
- LLM 路径：按同一指标清单逐项判断并输出命中/部分命中/证据（data/prompts.py 提示词）；
- 规则路径：对转写文本逐项检测可观察信号，与 LLM 使用同一指标 ID，保证可解释；
- 准确度维度：无可核验资料时按 rubric 规则固定 50 分，不臆测事实（NFR-09）。
"""
from __future__ import annotations

import json
import logging
import math
import re

from core.base_analyzer import BaseAnalyzer
from core.schemas import AnalysisInput, AnalyzerResult
from core.settings import get_settings
from data.prompts import llm_scoring_message
from data.rubric import INDICATORS, dimension_score, weighted_score

logger = logging.getLogger(__name__)

LOGIC_WORDS = ["首先", "其次", "然后", "最后", "第一", "第二", "因此", "所以",
               "因为", "同时", "另外", "此外", "而且", "并且"]
CAUSE_WORDS = ["因为", "由于", "因此", "所以", "导致", "为了", "源于", "基于"]
OPINION_WORDS = ["我认为", "我会", "我觉得", "应该", "需要", "选择", "建议"]
ACTION_WORDS = ["负责", "主导", "推动", "设计", "实现", "搭建", "制定", "发起",
                "落地", "优化", "重构", "验证", "复盘", "执行", "完成", "采取"]
RESULT_WORDS = ["结果", "成果", "最终", "完成", "达到", "提升", "降低", "上线",
                "落地", "产出", "收益", "增长"]
CONTEXT_WORDS = ["背景", "当时", "团队", "阶段", "时间", "规模", "部门", "公司",
                 "项目组", "业务"]
TRADEOFF_WORDS = ["取舍", "权衡", "成本", "风险", "限制", "放弃", "优先", "瓶颈",
                  "约束", "考虑"]
TECH_WORDS = ["接口", "数据库", "缓存", "架构", "框架", "算法", "部署", "集群",
              "服务", "协议", "模型", "组件", "模块", "流量", "延迟", "并发",
              "数据", "指标", "报表", "需求", "用户", "系统", "平台", "工具",
              "方案", "性能", "可用", "一致性", "分布式", "微服务", "容器"]
UNCERTAIN_WORDS = ["据我了解", "印象中", "可能", "大概", "估计", "不确定"]
OVERCLAIM_WORDS = ["绝对", "百分百", "百分之百", "从来都", "永远都", "保证",
                   "必然", "毫无悬念"]
_STOP_CHARS = set("的了呢吗啊吧我们你们他们这个那个什么怎么如何进行工作面试回答"
                  "请问说明介绍一下认为觉得自己以及或者就是然后因为所以可以如果"
                  "但是需要没有还是时候情况问题还有这些那些通过非常比较很多")

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
SENTENCE_RE = re.compile(r"[。！？!?；;\n]+")
REPEAT_RE = re.compile(r"(.{2,4})\1\1")  # 三连重复的 2-4 字片段 → 口头禅式冗余
D2_RE = re.compile(r"我[^。！？!?；;\n]{0,12}(负责|主导|推动|设计|实现|独立|搭建|制定|发起|落地|优化|重构)")


def _bigrams(text: str) -> set:
    """中文双字重叠度：过滤停用字，用于题意相关性与资料一致性判断。"""
    cleaned = re.sub(r"[^一-龥A-Za-z0-9]", "", text)
    return {
        cleaned[i:i + 2]
        for i in range(len(cleaned) - 1)
        if not _STOP_CHARS & set(cleaned[i:i + 2])
    }


def _has_any(text: str, words) -> bool:
    return any(word in text for word in words)


def _count_hits(text: str, words) -> int:
    return sum(text.count(word) for word in words)


def _rule_checklist(question: str, transcript: str,
                    context: dict) -> tuple[dict, dict]:
    """规则路径：逐指标检测，返回 (五维得分, 命中清单+证据)。"""
    hits: dict = {dim: [] for dim in INDICATORS}
    evidence: dict = {dim: [] for dim in INDICATORS}
    has_context = bool(context) and any(str(v).strip() for v in context.values())
    context_text = " ".join(str(v) for v in context.values())
    q_grams = _bigrams(question)
    a_grams = _bigrams(transcript)
    ctx_grams = _bigrams(context_text)
    numbers = NUMBER_RE.findall(transcript)
    sentences = [p for p in SENTENCE_RE.split(transcript) if p.strip()]
    n = len(transcript)
    tail = transcript[int(len(transcript) * 0.66):]

    def mark(dim: str, item_id: str, proof: str) -> None:
        hits[dim].append(item_id)
        evidence[dim].append(proof)

    # ---- 逻辑性 ----
    if len(q_grams & a_grams) >= 2:
        mark("logic", "L1", "回答与题目关键词相关（双字重叠度达标）")
    if _has_any(transcript, OPINION_WORDS):
        mark("logic", "L2", "检测到观点/结论句")
    if _has_any(transcript, CAUSE_WORDS):
        mark("logic", "L3", f"因果依据：{'、'.join(w for w in CAUSE_WORDS if w in transcript)[:60]}")
    if _count_hits(transcript, ["首先", "其次", "然后", "接着", "最后", "第一", "第二", "第三", "一方面", "另一方面"]) >= 2:
        mark("logic", "L4", "顺序词 ≥2 处，分步论证")
    if _has_any(tail, ["总结", "综上", "因此", "所以", "总之", "最终", "结论"]):
        mark("logic", "L5", "结尾有总结收束")

    # ---- 深度 ----
    if numbers:
        mark("depth", "D1", f"检测到 {len(numbers)} 处数字/比例")
    if D2_RE.search(transcript):
        mark("depth", "D2", "明确个人职责与动作")
    if _has_any(transcript, CONTEXT_WORDS):
        mark("depth", "D3", f"情境背景：{'、'.join(w for w in CONTEXT_WORDS if w in transcript)[:60]}")
    if re.search(r"[A-Za-z]{2,}", transcript) or _has_any(transcript, TECH_WORDS):
        mark("depth", "D4", "检测到技术/业务名词")
    if _has_any(transcript, TRADEOFF_WORDS):
        mark("depth", "D5", f"约束与取舍：{'、'.join(w for w in TRADEOFF_WORDS if w in transcript)[:60]}")

    # ---- 结构 ----
    if _has_any(transcript[:30], ["我会", "首先", "我认为", "我的回答", "从", "分", "以下"]):
        mark("structure", "S1", "开头有总起")
    if _count_hits(transcript, ["首先", "其次", "然后", "接着", "最后", "第一", "第二", "第三"]) >= 2 \
            or re.search(r"\d+[.、]", transcript):
        mark("structure", "S2", "分点/顺序词展开")
    if _count_hits(transcript, ACTION_WORDS) >= 2:
        mark("structure", "S3", "过程完整（动作词 ≥2 处）")
    if _has_any(transcript, RESULT_WORDS):
        mark("structure", "S4", f"结果导向：{'、'.join(w for w in RESULT_WORDS if w in transcript)[:60]}")
    if _has_any(tail, ["总结", "综上", "因此", "总之", "最终"]):
        mark("structure", "S5", "结尾收束")

    # ---- 表达 ----
    if 2 <= len(sentences) <= 6:
        mark("expression", "E1", f"句段数 {len(sentences)}（适中区间 2-6）")
    if 80 <= n <= 800:
        mark("expression", "E2", f"总长度 {n} 字（适中区间 80-800）")
    if not REPEAT_RE.search(transcript):
        mark("expression", "E3", "无口头禅式连续重复")
    distinct_actions = {w for w in ACTION_WORDS if w in transcript}
    if len(distinct_actions) >= 2:
        mark("expression", "E4", f"具体动作词 {len(distinct_actions)} 个：{'、'.join(sorted(distinct_actions))[:60]}")
    if _count_hits(transcript, LOGIC_WORDS) >= 2:
        mark("expression", "E5", "连接词 ≥2 处，衔接自然")

    # ---- 准确度（特殊规则：无资料时固定 50 分，不臆测）----
    if len(q_grams & a_grams) >= 2:
        mark("accuracy", "A1", "内容紧扣题意")
    if has_context:
        shared = a_grams & ctx_grams
        if len(shared) >= 3:
            mark("accuracy", "A2", f"与资料一致（双字重叠 {len(shared)} 处：{'、'.join(sorted(shared))[:60]}）")
        ctx_numbers = set(NUMBER_RE.findall(context_text))
        if not numbers or ctx_numbers & set(numbers):
            mark("accuracy", "A3", "数字/事实有出处或与资料一致" if numbers else "无未核验数据")
        if not _has_any(transcript, OVERCLAIM_WORDS):
            mark("accuracy", "A4", "无绝对化断言")
        if _has_any(transcript, UNCERTAIN_WORDS):
            mark("accuracy", "A5", "信息不足处标注了不确定性")

    scores = {}
    for dim in INDICATORS:
        if dim == "accuracy" and not has_context:
            scores[dim] = 50.0  # rubric：无资料 → 中性分，不臆测
        else:
            scores[dim] = dimension_score(len(hits[dim]))
    return scores, evidence, hits


def _checklist_payload(hits: dict, partial: dict | None,
                       scores: dict) -> dict:
    """组装量化指标命中清单，供报告层展示。"""
    payload = {}
    for dim, meta in INDICATORS.items():
        names = {item["id"]: item["name"] for item in meta["checks"]}
        hit_ids = list(hits.get(dim, []))
        partial_ids = list((partial or {}).get(dim, []))
        all_ids = [item["id"] for item in meta["checks"]]
        missed = [i for i in all_ids if i not in hit_ids and i not in partial_ids]
        payload[dim] = {
            "score": round(float(scores.get(dim, 0.0)), 1),
            "hits": [{"id": i, "name": names.get(i, i)} for i in hit_ids],
            "partial": [{"id": i, "name": names.get(i, i)} for i in partial_ids],
            "missed": [{"id": i, "name": names.get(i, i)} for i in missed],
        }
    return payload


def _parse_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _valid_ids() -> dict:
    return {dim: {item["id"] for item in meta["checks"]}
            for dim, meta in INDICATORS.items()}


def _parse_llm_payload(data: dict) -> dict | None:
    """解析 LLM 输出的评分 JSON（新契约 dims；兼容旧契约 scores）。"""
    dims = data.get("dims")
    if isinstance(dims, dict):
        scores, hits, partial, evidence, gaps = {}, {}, {}, {}, {}
        valid = _valid_ids()
        for dim in INDICATORS:
            item = dims.get(dim)
            if not isinstance(item, dict):
                continue
            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            scores[dim] = round(max(0.0, min(100.0, score)), 1)
            hits[dim] = [str(i) for i in item.get("hits", []) if str(i) in valid[dim]][:5]
            partial[dim] = [str(i) for i in item.get("partial", [])
                            if str(i) in valid[dim] and str(i) not in hits[dim]][:5]
            raw_ev = item.get("evidence", [])
            evidence[dim] = [str(e)[:240] for e in raw_ev[:3]] if isinstance(raw_ev, list) else []
            gaps[dim] = str(item.get("gap", "") or "").strip()[:240]
        if not scores:
            return None
        suggestions = data.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = []
        return {
            "scores": scores,
            "evidence": evidence,
            "gaps": gaps or {k: "" for k in INDICATORS},
            "checklist": _checklist_payload(hits, partial, scores),
            "suggestions": [str(s).strip()[:240] for s in suggestions[:4] if str(s).strip()],
        }

    # 旧契约：{"scores": {...}, "evidence": {...}, "gaps": {...}, "suggestions": [...]}
    if isinstance(data.get("scores"), dict):
        scores = {}
        for key in INDICATORS:
            try:
                value = float(data["scores"][key])
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            scores[key] = round(max(0.0, min(100.0, value)), 1)
        evidence = data.get("evidence", {})
        gaps = data.get("gaps", {})
        if not isinstance(evidence, dict) or not isinstance(gaps, dict):
            return None
        suggestions = data.get("suggestions", [])
        if not isinstance(suggestions, list):
            return None
        return {
            "scores": scores,
            "evidence": {
                key: [str(item)[:240] for item in evidence.get(key, [])[:3]]
                if isinstance(evidence.get(key, []), list) else []
                for key in INDICATORS
            },
            "gaps": {key: str(gaps.get(key, ""))[:240] for key in INDICATORS},
            "checklist": {},
            "suggestions": [str(item).strip()[:240] for item in suggestions[:4]
                            if str(item).strip()],
        }
    return None


def _llm_evaluate(question: str, transcript: str, context: dict) -> dict | None:
    """调用本地 Ollama；输出不满足评分契约时返回 None 进入规则降级。"""
    try:
        import requests

        cfg = get_settings()["content"]
        resp = requests.post(
            cfg["ollama_url"],
            json={
                "model": cfg["model"], "prompt": llm_scoring_message(
                    question, transcript, context
                ), "stream": False, "format": "json",
            },
            timeout=float(cfg["timeout_sec"]),
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        data = _parse_object(str(text))
        if not data:
            return None
        return _parse_llm_payload(data)
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        logger.info("Ollama 不可用，使用规则评估: %s", exc)
        return None


_MISS_TIPS = {
    "L1": "先直接回应题目所问，避免答非所问",
    "L2": "开头亮出明确观点或结论，再展开论证",
    "L3": "补充决策依据：因为…所以…，说明原因与结果的关系",
    "L4": "用 首先/其次/最后 等分点展开，逻辑更清晰",
    "L5": "结尾用一句总结呼应开头观点",
    "D1": "补充可核验的数字结果（提升比例、耗时、规模等）",
    "D2": "明确哪些部分由你个人负责，区分个人贡献与团队成果",
    "D3": "交代项目背景、团队规模与时间阶段",
    "D4": "加入具体技术、工具或业务名词，避免泛泛而谈",
    "D5": "说明约束、成本、风险与方案取舍",
    "S1": "开头先总起，预告回答的结构",
    "S2": "用分点或顺序词分层展开（≥2 处）",
    "S3": "补全过程链条：情境—行动—推进，至少两个动作",
    "S4": "给出结果、产出或结论，不要停在过程描述",
    "S5": "结尾总结收束，让回答完整",
    "E1": "控制句段数量在 2-6 个，过长请拆分",
    "E2": "篇幅控制在 80-800 字，过长请精简",
    "E3": "减少口头禅式的重复表达",
    "E4": "用具体动词（负责/推动/设计/验证）替代空泛描述",
    "E5": "用连接词串联，让表达更连贯",
    "A1": "紧扣题意作答，删除离题内容",
    "A2": "陈述应与候选人资料保持一致，先核对再作答",
    "A3": "数据要有出处，无法核验时说明来源",
    "A4": "避免绝对化断言，为结论留出边界条件",
    "A5": "信息不足时主动说明不确定性与假设",
}


def _suggestions_from_misses(checklist: dict) -> list:
    tips = []
    for dim, payload in checklist.items():
        for item in payload.get("missed", []):
            tip = _MISS_TIPS.get(item["id"])
            if tip and tip not in tips:
                tips.append(tip)
    return tips[:4]


def _rule_evaluate(question: str, transcript: str, context: dict) -> dict:
    """规则评分：按量化指标清单逐项检测可观察信号。"""
    scores, evidence, hits = _rule_checklist(question, transcript, context)
    checklist = _checklist_payload(hits, {}, scores)
    suggestions = _suggestions_from_misses(checklist)
    if not suggestions:
        suggestions = ["表达已有具体证据，可进一步说明方案取舍和验证方式"]
    gaps = {dim: "" for dim in INDICATORS}
    gaps["accuracy"] = (
        "无候选人资料，按 rubric 固定 50 分；请对照知识库和原始证据人工复核"
    )
    return {
        "scores": scores,
        "evidence": evidence,
        "gaps": gaps,
        "checklist": checklist,
        "suggestions": suggestions,
    }


class ContentAnalyzer(BaseAnalyzer):
    modality = "content"

    def _run(self, inp: AnalysisInput) -> AnalyzerResult:
        # MVP 简化：转写文本由语音分析产生，此处从共享输入获取或由调用方注入
        transcript = inp.question.get("_transcript", "")

        if not transcript:
            return AnalyzerResult(
                modality=self.modality, status="degraded", score=55.0,
                metrics={"five_dims": {}},
                suggestions=["无可用转写文本，内容维度降级为基线评估"],
            )

        context = inp.question.get("_candidate_context", {})
        if not isinstance(context, dict):
            context = {}
        evaluated = _llm_evaluate(
            inp.question.get("content", ""), transcript, context
        )
        status = "ok"
        if evaluated is None:
            evaluated = _rule_evaluate(
                inp.question.get("content", ""), transcript, context
            )
            status = "degraded"

        dims = evaluated["scores"]
        score = weighted_score(dims)
        return AnalyzerResult(
            modality=self.modality, status=status, score=score,
            metrics={
                "five_dims": dims,
                "evidence": evaluated["evidence"],
                "gaps": evaluated["gaps"],
                "rubric": {dim: {"name": meta["name"], "weight": meta["weight"],
                                "criterion": meta["criterion"]}
                           for dim, meta in INDICATORS.items()},
                "checklist": evaluated["checklist"],
                "provider": "ollama" if status == "ok" else "rule",
            },
            suggestions=evaluated["suggestions"],
        )
