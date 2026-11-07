"""标准化 JSON 数据契约（对应第4周第七节：统一接口与模块解耦设计）。

所有跨模块交互仅通过本模块定义的数据结构传递，
保证"契约稳定、内部可变更"，支撑模型可插拔替换。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


def new_session_id() -> str:
    """生成匿名会话 ID（不含任何 PII，对应安全性 NFR-09）。"""
    return uuid.uuid4().hex[:12]


@dataclass
class AnalysisInput:
    """进入分析流水线的统一输入。"""
    session_id: str
    question: dict               # {id, category, content}
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    job_category: str = "default"
    input_meta: dict = field(default_factory=dict)


@dataclass
class AnalyzerResult:
    """单个分析过滤器的标准化输出（对应第4周接口表）。

    status: ok / degraded / error —— 支持部分模态降级。
    """
    modality: str                          # speech / content / vision
    status: str                            # ok | degraded | error
    score: float = 0.0                     # 0-100 维度综合分
    metrics: dict = field(default_factory=dict)
    suggestions: list = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    elapsed: float = 0.0
    confidence: Optional[float] = None     # 0-1；None 时由状态推断

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FusionResult:
    """多模态融合层输出。"""
    overall_score: float
    dimension_scores: dict                 # {content: x, speech: y, vision: z}
    weights_used: dict
    degraded_modalities: list = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    """报告生成层输出，直接交付表现层渲染。"""
    session_id: str
    question: dict
    overall_score: float
    dimension_scores: dict
    radar: dict                            # 五维雷达图指标
    suggestions: list
    modality_status: dict
    elapsed: dict
    created_at: float = field(default_factory=time.time)
    input_meta: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    confidence: float = 0.0
    scoring_details: dict = field(default_factory=dict)
    transcript: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# 统一错误契约（第4周：统一错误返回契约）
def error_result(modality: str, code: str, message: str) -> AnalyzerResult:
    return AnalyzerResult(
        modality=modality, status="error",
        error_code=code, error_message=message,
    )
