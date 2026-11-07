"""多模态加权融合（第3/4周：多模态融合层 —— 权重配置加载器 + 分数聚合器）。

权重按岗位类别外置在 config/weights.json，运行时可修改、即改即生效（NFR-06）。
部分模态降级/失败时自动对剩余可用模态归一化权重（第4周异常分支：降级）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from core.schemas import AnalyzerResult, FusionResult

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "weights.json")

DEFAULT_WEIGHTS = {"content": 0.55, "speech": 0.45, "vision": 0.0}


def load_weights(job_category: str = "default") -> Dict[str, float]:
    """权重配置加载器：按岗位类别读取权重，配置文件缺失时回退默认值。"""
    try:
        with open(_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            table = json.load(f)
        return table.get(job_category, table.get("default", DEFAULT_WEIGHTS))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_WEIGHTS)


def _confidence(result: AnalyzerResult) -> float:
    if result.confidence is not None:
        try:
            return max(0.0, min(1.0, float(result.confidence)))
        except (TypeError, ValueError):
            pass
    return {"ok": 1.0, "degraded": 0.4, "error": 0.0}.get(result.status, 0.0)


def fuse(results: List[AnalyzerResult], job_category: str = "default") -> FusionResult:
    """置信度加权融合：降级模态降低权重，避免基线分拉偏综合分。"""
    weights = load_weights(job_category)
    usable = [
        r for r in results
        if r.status in ("ok", "degraded") and _confidence(r) > 0
    ]
    degraded = [r.modality for r in results if r.status != "ok"]

    if not usable:
        return FusionResult(0.0, {}, weights, degraded, 0.0)

    dim_scores = {
        r.modality: round(max(0.0, min(100.0, r.score)), 1)
        for r in usable
    }
    # 视觉只反馈画面可见性，不作为面试能力分；权重为 0 的模态也不能
    # 在其他模态缺失时通过等权回退进入综合分。
    scored = [r for r in usable if weights.get(r.modality, 0.0) > 0]
    if not scored:
        return FusionResult(0.0, dim_scores, {}, degraded, 0.0)

    effective = {
        r.modality: weights.get(r.modality, 0.0) * _confidence(r)
        for r in scored
    }
    total_w = sum(effective.values())
    if total_w <= 0:
        return FusionResult(0.0, dim_scores, {}, degraded, 0.0)

    base_total = sum(weights.get(r.modality, 0.0) for r in scored)
    if base_total > 0:
        overall_confidence = sum(
            weights.get(r.modality, 0.0) * _confidence(r) for r in scored
        ) / base_total
    else:
        overall_confidence = 0.0

    overall = 0.0
    for r in scored:
        w = effective.get(r.modality, 0.0) / total_w
        overall += r.score * w

    return FusionResult(
        round(max(0.0, min(100.0, overall)), 1),
        dim_scores,
        effective,
        degraded,
        round(max(0.0, min(1.0, overall_confidence)), 2),
    )
