"""工厂模式：分析器的配置化创建。

新增/替换分析模态只需向注册表登记，编排层与融合层代码零修改，
对应 NFR-05"新增模态可通过实现接口接入"与第4周模块适配器职责。
"""
from __future__ import annotations

from typing import Dict, List, Type

from core.base_analyzer import BaseAnalyzer


class AnalyzerFactory:
    _registry: Dict[str, Type[BaseAnalyzer]] = {}

    @classmethod
    def register(cls, name: str, analyzer_cls: Type[BaseAnalyzer]) -> None:
        if not issubclass(analyzer_cls, BaseAnalyzer):
            raise TypeError(f"{analyzer_cls} 必须继承 BaseAnalyzer")
        cls._registry[name] = analyzer_cls

    @classmethod
    def create(cls, name: str) -> BaseAnalyzer:
        if name not in cls._registry:
            raise KeyError(f"未注册的分析器: {name}")
        return cls._registry[name]()

    @classmethod
    def create_all(cls, enabled: List[str] | None = None) -> List[BaseAnalyzer]:
        # 注意：enabled=[] 表示"一个都不启用"，不能用 `enabled or ...`
        names = list(cls._registry) if enabled is None else list(enabled)
        return [cls.create(n) for n in names]

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._registry)


def register_defaults() -> None:
    """登记系统内置的三路分析策略。"""
    from analyzers.speech_analyzer import SpeechAnalyzer
    from analyzers.content_analyzer import ContentAnalyzer
    from analyzers.vision_analyzer import VisionAnalyzer

    AnalyzerFactory.register("speech", SpeechAnalyzer)
    AnalyzerFactory.register("content", ContentAnalyzer)
    AnalyzerFactory.register("vision", VisionAnalyzer)
