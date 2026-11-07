"""语音分析过滤器（策略模式具体策略之一）。

优先使用 faster-whisper 转写 + librosa 提取特征；
依赖缺失或音频不可读时自动降级为规则估计，保证流水线不中断（第4周异常分支）。
"""
from __future__ import annotations

import logging
import os
import threading
import wave
from typing import Any, Dict, Tuple

from core.base_analyzer import BaseAnalyzer
from core.schemas import AnalysisInput, AnalyzerResult
from core.settings import get_settings

logger = logging.getLogger(__name__)

FILLERS = ["嗯", "呃", "额", "那个", "就是", "然后然后", "对对对"]
IDEAL_RATE = (150, 180)  # 字/分，US-04 建议区间

# Whisper 模型加载成本高，进程内复用；Streamlit 重跑不会重复加载。
_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        size = os.path.getsize(path)
        return max(size / 32000.0, 1.0)  # 粗略估计：16kHz 16bit 单声道


def _get_model():
    """按配置加载并缓存 WhisperModel（线程安全）。"""
    cfg = get_settings()["speech"]
    key = (str(cfg["model_size"]), str(cfg["device"]), str(cfg["compute_type"]))
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(key[0], device=key[1], compute_type=key[2])
            _MODEL_CACHE[key] = model
            logger.info("Whisper 模型已加载: %s/%s/%s", *key)
        return model


def _transcribe(path: str) -> str | None:
    """faster-whisper 转写；模型或依赖不可用时返回 None 进入降级。"""
    try:
        cfg = get_settings()["speech"]
        model = _get_model()
        segments, _ = model.transcribe(
            path,
            language=cfg["language"],
            beam_size=int(cfg["beam_size"]),
            vad_filter=bool(cfg["vad_filter"]),
        )
        return "".join(s.text for s in segments).strip() or None
    except Exception as exc:
        logger.warning("语音转写失败，将降级: %s", exc)
        return None


class SpeechAnalyzer(BaseAnalyzer):
    modality = "speech"

    def _run(self, inp: AnalysisInput) -> AnalyzerResult:
        if not inp.audio_path or not os.path.exists(inp.audio_path):
            return AnalyzerResult(
                modality=self.modality, status="degraded", score=60.0,
                suggestions=["未检测到音频输入，语音维度按基线分处理"],
            )

        duration = _wav_duration(inp.audio_path)
        transcript = _transcribe(inp.audio_path)

        if transcript is None:  # 降级：仅基于时长估计
            return AnalyzerResult(
                modality=self.modality, status="degraded", score=65.0,
                metrics={"duration_sec": round(duration, 1)},
                suggestions=[
                    "语音转写不可用（模型加载失败或音频无法解码），本次为降级评估"
                ],
            )

        minutes = max(duration / 60.0, 1e-6)
        n_chars = len(transcript)
        rate = n_chars / minutes
        filler_hits = sum(transcript.count(f) for f in FILLERS)
        filler_per_min = filler_hits / minutes

        # 语速评分：落在理想区间满分，偏离线性扣分
        if IDEAL_RATE[0] <= rate <= IDEAL_RATE[1]:
            rate_score = 100.0
        else:
            dev = min(abs(rate - IDEAL_RATE[0]), abs(rate - IDEAL_RATE[1]))
            rate_score = max(100.0 - dev * 0.8, 30.0)
        filler_score = max(100.0 - filler_per_min * 12, 30.0)
        score = round(rate_score * 0.6 + filler_score * 0.4, 1)

        suggestions = []
        if rate > IDEAL_RATE[1]:
            suggestions.append(f"语速 {rate:.0f} 字/分，建议控制在 {IDEAL_RATE[0]}-{IDEAL_RATE[1]} 字/分")
        elif rate < IDEAL_RATE[0]:
            suggestions.append(f"语速 {rate:.0f} 字/分偏慢，建议提升至 {IDEAL_RATE[0]}-{IDEAL_RATE[1]} 字/分")
        if filler_per_min > 2:
            suggestions.append(f"填充词约 {filler_per_min:.1f} 次/分（{filler_hits} 次），建议减少口头禅")

        return AnalyzerResult(
            modality=self.modality, status="ok", score=score,
            metrics={
                "transcript": transcript,
                "duration_sec": round(duration, 1),
                "speech_rate": round(rate, 1),
                "filler_count": filler_hits,
                "model": get_settings()["speech"]["model_size"],
            },
            suggestions=suggestions or ["语音表达节奏良好"],
        )
