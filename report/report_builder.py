"""报告生成层：将融合结果组织为五维雷达图指标、分项评分与改进建议。"""
from __future__ import annotations

from typing import List

from core.schemas import AnalysisInput, AnalyzerResult, FusionResult, Report

RADAR_DIMS = ["逻辑性", "深度", "结构", "表达", "准确度"]


def _radar_from(results: List[AnalyzerResult]) -> dict:
    """五维雷达图：内容维度五维优先取 LLM/规则输出，缺失时用其他模态近似。"""
    content = next((r for r in results if r.modality == "content"), None)
    five = (content.metrics.get("five_dims") if content else None) or {}
    speech = next((r for r in results if r.modality == "speech"), None)
    fallback = {
        "逻辑性": 60.0, "深度": 60.0, "结构": 60.0,
        "表达": speech.score if speech else 60.0, "准确度": 60.0,
    }
    mapping = {"逻辑性": "logic", "深度": "depth", "结构": "structure",
               "表达": "expression", "准确度": "accuracy"}
    return {zh: round(float(five.get(en, fallback[zh])), 1)
            for zh, en in mapping.items()}


def build_report(inp: AnalysisInput, results: List[AnalyzerResult],
                 fusion: FusionResult) -> Report:
    suggestions: List[str] = []
    for r in results:
        suggestions.extend(r.suggestions)
    suggestions.extend(inp.input_meta.get("warnings", []))
    if fusion.degraded_modalities:
        suggestions.append(
            f"以下模态本次降级/不可用：{'、'.join(fusion.degraded_modalities)}，"
            "完整体验请安装对应依赖（见 requirements.txt）")
    content = next((r for r in results if r.modality == "content"), None)
    speech = next((r for r in results if r.modality == "speech"), None)

    return Report(
        session_id=inp.session_id,
        question={k: v for k, v in inp.question.items() if not k.startswith("_")},
        overall_score=fusion.overall_score,
        dimension_scores=fusion.dimension_scores,
        radar=_radar_from(results),
        suggestions=suggestions,
        modality_status={r.modality: r.status for r in results},
        elapsed={r.modality: r.elapsed for r in results},
        input_meta=dict(inp.input_meta),
        errors={
            r.modality: r.error_message
            for r in results
            if r.status == "error" and r.error_message
        },
        confidence=fusion.confidence,
        scoring_details={
            "evidence": (content.metrics.get("evidence", {}) if content else {}),
            "gaps": (content.metrics.get("gaps", {}) if content else {}),
            "rubric": (content.metrics.get("rubric", {}) if content else {}),
            "checklist": (content.metrics.get("checklist", {}) if content else {}),
            "method": (content.metrics.get("provider", "unavailable") if content else "unavailable"),
        },
        transcript=(speech.metrics.get("transcript", "") if speech else ""),
    )
