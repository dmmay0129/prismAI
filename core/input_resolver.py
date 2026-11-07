"""输入归一化层。

职责：
- 校验音频/视频文件是否存在、是否超过大小与时长限制；
- 当用户只上传视频时，用 PyAV（经 faster-whisper）提取 16kHz 单声道 WAV；
- 返回统一的 ``ResolvedInput``，并在生命周期结束时清理临时文件。

这一层把"输入格式差异"挡在分析器之外，分析器仍然只处理
``AnalysisInput.audio_path/video_path``，无需感知视频转音频细节。
"""
from __future__ import annotations

import logging
import os
import tempfile
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.settings import get_settings

logger = logging.getLogger(__name__)

# 延迟导入；测试可替换该模块变量，避免真正加载 PyAV。
decode_audio = None


class InputValidationError(ValueError):
    """输入缺失、不可读或超出限制。"""


@dataclass
class ResolvedInput:
    """解析后的输入及其派生资源。"""

    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    extracted_audio: bool = False
    duration_sec: float = 0.0
    warnings: List[str] = field(default_factory=list)
    _temp_paths: List[str] = field(default_factory=list, repr=False)

    def meta(self) -> dict:
        return {
            "audio_source": "video" if self.extracted_audio else (
                "upload" if self.audio_path else None
            ),
            "has_video": bool(self.video_path),
            "duration_sec": round(self.duration_sec, 2),
            "warnings": list(self.warnings),
        }

    def cleanup(self) -> None:
        """删除本层创建的临时文件（可重复调用）。"""
        for path in self._temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        self._temp_paths.clear()


def _validate_file(path: Optional[str], label: str) -> None:
    if not path:
        return
    if not os.path.isfile(path):
        raise InputValidationError(f"{label}文件不存在: {path}")
    size = os.path.getsize(path)
    if size <= 0:
        raise InputValidationError(f"{label}文件为空: {path}")
    max_mb = get_settings()["input"].get("max_upload_mb")
    if max_mb and size > int(max_mb) * 1024 * 1024:
        raise InputValidationError(
            f"{label}文件超过 {max_mb} MB 限制，请压缩或截短后重试"
        )


def _media_duration(path: Optional[str]) -> float:
    """尽力读取媒体时长；无法读取时返回 0，不阻塞分析。"""
    if not path:
        return 0.0
    try:
        with wave.open(path, "rb") as wav:
            return wav.getnframes() / float(wav.getframerate() or 1)
    except Exception:
        pass
    try:
        import av

        with av.open(path) as container:
            if container.duration:
                return float(container.duration) / av.time_base
            stream = next(
                (s for s in container.streams if s.type in ("audio", "video")), None
            )
            if stream is not None and stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
    except Exception:
        pass
    return 0.0


def _load_decode_audio():
    global decode_audio
    if decode_audio is None:
        try:
            from faster_whisper.audio import decode_audio as fn
        except Exception as exc:  # 依赖缺失或 ABI 不兼容
            raise InputValidationError(
                "未安装可用的 faster-whisper/PyAV，无法从视频提取音频；"
                "请直接上传 WAV 文件"
            ) from exc
        decode_audio = fn
    return decode_audio


def _extract_audio(video_path: str) -> Tuple[str, float]:
    """把视频音轨解码为 16kHz/16bit/单声道 WAV，返回 (路径, 时长)。"""
    try:
        import numpy as np
    except ImportError as exc:
        raise InputValidationError("从视频提取音频需要 numpy，请先安装依赖") from exc

    try:
        samples = _load_decode_audio()(video_path, sampling_rate=16000)
    except InputValidationError:
        raise
    except Exception as exc:
        raise InputValidationError(f"无法读取视频音轨: {exc}") from exc

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise InputValidationError("视频中没有可用音轨，请同时上传 WAV 音频")
    if samples.ndim > 1:
        samples = samples.reshape(-1)

    duration = samples.size / 16000.0
    max_sec = get_settings()["input"].get("max_audio_sec")
    if max_sec and duration > float(max_sec):
        raise InputValidationError(
            f"视频音轨时长 {duration:.0f}s 超过 {max_sec}s 限制，请截短后重试"
        )

    temp_dir = get_settings()["input"].get("temp_dir")
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)

    fd, path = tempfile.mkstemp(
        prefix="prismai_audio_", suffix=".wav", dir=temp_dir or None
    )
    try:
        with os.fdopen(fd, "wb") as f:
            with wave.open(f, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                pcm = (samples.clip(-1.0, 1.0) * 32767.0).astype("<i2")
                wav.writeframes(pcm.tobytes())
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path, duration


def resolve_inputs(
    audio_path: Optional[str] = None,
    video_path: Optional[str] = None,
) -> ResolvedInput:
    """校验输入并补齐音频；调用方负责在结束后执行 ``cleanup()``。"""
    _validate_file(audio_path, "音频")
    _validate_file(video_path, "视频")
    if not audio_path and not video_path:
        raise InputValidationError("请至少上传 WAV 音频或视频文件之一")

    settings = get_settings()["input"]
    max_audio = settings.get("max_audio_sec")
    max_video = settings.get("max_video_sec")

    video_duration = _media_duration(video_path)
    if video_path and max_video and video_duration > float(max_video):
        raise InputValidationError(
            f"视频时长 {video_duration:.0f}s 超过 {max_video}s 限制，请截短后重试"
        )

    resolved = ResolvedInput(video_path=video_path)
    if audio_path:
        audio_duration = _media_duration(audio_path)
        if max_audio and audio_duration > float(max_audio):
            raise InputValidationError(
                f"音频时长 {audio_duration:.0f}s 超过 {max_audio}s 限制，请截短后重试"
            )
        resolved.audio_path = audio_path
        resolved.duration_sec = audio_duration
        logger.info("使用上传音频: %s (%.1fs)", audio_path, audio_duration)
        return resolved

    # 只有视频：提取音轨作为语音/内容分析的输入
    extracted_path, duration = _extract_audio(video_path)  # type: ignore[arg-type]
    resolved.audio_path = extracted_path
    resolved.extracted_audio = True
    resolved.duration_sec = duration
    resolved._temp_paths.append(extracted_path)
    resolved.warnings.append(
        "未上传独立音频，已自动从视频提取音轨用于语音与内容分析。"
    )
    logger.info("已从视频提取音频: %s -> %s (%.1fs)", video_path, extracted_path, duration)
    return resolved

