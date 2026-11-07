"""门面模式：InterviewCoachFacade 应用编排层统一入口。

表现层只与门面交互，无需感知会话管理、并行调度、模块适配、
异常降级、融合、持久化等子系统复杂性（第3周五.2 / 第4周五.2）。

第5周 MVP 架构发现：内容分析依赖语音转写文本，三路无法完全并行；
调整为两阶段调度——语音分析先行，内容+视觉并行（详见第5周文档第八节）。
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Optional

from core.analyzer_factory import AnalyzerFactory, register_defaults
from core.errors import AnalysisBusyError
from core.input_resolver import resolve_inputs
from core.schemas import AnalysisInput, AnalyzerResult, Report, new_session_id
from core.fusion import fuse
from core.settings import get_settings
from report.report_builder import build_report
from data.storage import save_result

logger = logging.getLogger(__name__)

register_defaults()

_SEMAPHORE = None
_SEMAPHORE_LOCK = threading.Lock()


def _get_semaphore() -> threading.BoundedSemaphore:
    global _SEMAPHORE
    with _SEMAPHORE_LOCK:
        if _SEMAPHORE is None:
            limit = max(
                1, int(get_settings()["concurrency"]["max_concurrent_analyses"])
            )
            _SEMAPHORE = threading.BoundedSemaphore(limit)
        return _SEMAPHORE


@contextmanager
def _analysis_slot():
    """限制进程内同时运行的分析数量，避免多会话争抢 CPU/模型。"""
    sem = _get_semaphore()
    timeout = max(0.0, float(get_settings()["concurrency"]["queue_timeout_sec"]))
    if not sem.acquire(timeout=timeout):
        raise AnalysisBusyError("当前分析任务较多，请稍后重试")
    try:
        yield
    finally:
        sem.release()


class InterviewCoachFacade:
    """模拟面试完整流程的统一门面。"""

    def __init__(self, enabled_analyzers: Optional[List[str]] = None):
        self._analyzers = {a.modality: a for a in AnalyzerFactory.create_all(enabled_analyzers)}

    def run_interview(
        self,
        question: dict,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        job_category: str = "default",
        username: Optional[str] = None,
        candidate_context: Optional[dict] = None,
    ) -> Report:
        resolved = resolve_inputs(audio_path, video_path)
        try:
            with _analysis_slot():
                inp = AnalysisInput(
                    session_id=new_session_id(),
                    question=dict(question),
                    audio_path=resolved.audio_path,
                    video_path=resolved.video_path,
                    job_category=job_category,
                    input_meta=resolved.meta(),
                )
                if candidate_context:
                    inp.question["_candidate_context"] = {
                        str(key): str(value)[:6000]
                        for key, value in candidate_context.items()
                        if str(value).strip()
                    }

                results: List[AnalyzerResult] = []

                # 阶段一：语音分析（产生转写文本，供内容分析使用）
                speech = self._analyzers.get("speech")
                if speech:
                    results.append(speech.analyze(inp))
                    transcript = results[-1].metrics.get("transcript", "")
                    inp.question["_transcript"] = transcript

                # 阶段二：内容 + 视觉并行（互不依赖）
                parallel = [
                    a for m in ("content", "vision")
                    if (a := self._analyzers.get(m))
                ]
                if parallel:
                    with ThreadPoolExecutor(max_workers=len(parallel)) as pool:
                        for f in [pool.submit(a.analyze, inp) for a in parallel]:
                            results.append(f.result())

                fusion = fuse(results, job_category)
                report = build_report(inp, results, fusion)
                save_result(report, username=username)
                logger.info(
                    "session %s user=%s finished: overall=%.1f confidence=%.2f status=%s",
                    report.session_id,
                    username or "-",
                    report.overall_score,
                    fusion.confidence,
                    report.modality_status,
                )
                return report
        finally:
            # 只清理本层创建的临时音频；用户上传文件由表现层负责清理
            resolved.cleanup()
