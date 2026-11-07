"""领域异常：表现层按类型映射为友好提示，避免直接暴露底层异常。"""
from __future__ import annotations


class PrismAIError(Exception):
    """业务异常基类。"""


class AnalysisBusyError(PrismAIError):
    """并发分析槽位已满，等待超时。"""
