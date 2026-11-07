"""策略模式：分析过滤器统一抽象接口。

三路分析（语音/内容/视觉）均为 BaseAnalyzer 的具体策略实现，
编排层只依赖本抽象，不感知任何具体模型（faster-whisper / Qwen / OpenCV），
实现第3周决策2"输入输出稳定，内部模型可替换"。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from core.schemas import AnalysisInput, AnalyzerResult, error_result


class BaseAnalyzer(ABC):
    """分析策略抽象基类。所有模态过滤器必须实现 analyze()。"""

    modality: str = "base"

    @abstractmethod
    def _run(self, inp: AnalysisInput) -> AnalyzerResult:
        """子类实现具体分析逻辑。"""
        ...

    def analyze(self, inp: AnalysisInput) -> AnalyzerResult:
        """模板方法：统一计时与异常捕获，任何模型异常不外溢、转为降级结果。"""
        start = time.time()
        try:
            result = self._run(inp)
        except Exception as exc:  # 降级契约：单模态失败不阻塞整体流水线
            result = error_result(self.modality, "ANALYZER_EXCEPTION", str(exc))
        result.elapsed = round(time.time() - start, 2)
        if result.confidence is None:
            result.confidence = {"ok": 1.0, "degraded": 0.4, "error": 0.0}.get(
                result.status, 0.0
            )
        result.confidence = max(0.0, min(1.0, float(result.confidence)))
        return result
