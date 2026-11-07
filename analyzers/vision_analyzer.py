"""视觉分析过滤器（策略模式具体策略之三）。

使用 OpenCV Haar 级联做人脸检测，以"画面内稳定出镜率"作为自信度/紧张度代理指标；
无视频输入或依赖缺失时降级为中性基线分（接口预留置信度字段，对应第4周风险表）。
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading

from core.base_analyzer import BaseAnalyzer
from core.schemas import AnalysisInput, AnalyzerResult
from core.settings import get_settings

logger = logging.getLogger(__name__)

_CASCADE_NAMES = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_profileface.xml",
)
_CASCADE_LOCK = threading.Lock()
_CASCADE_CACHE: dict[str, str] = {}


def _cascade_path(cv2_module, name: str = _CASCADE_NAMES[0]) -> str | None:
    """返回 CascadeClassifier 可读取的级联路径。

    Windows 下 OpenCV 的 FileStorage 无法打开含中文的绝对路径，
    因此项目路径非 ASCII 时把 XML 复制到系统临时目录再加载。
    """
    with _CASCADE_LOCK:
        cached = _CASCADE_CACHE.get(name)
        if cached and os.path.isfile(cached):
            return cached

        root = getattr(getattr(cv2_module, "data", None), "haarcascades", None)
        if not root:
            return None
        source = os.path.join(root, name)
        if not os.path.isfile(source):
            return None
        if source.isascii():
            _CASCADE_CACHE[name] = source
            return source

        target = os.path.join(tempfile.gettempdir(), f"prismai_{name}")
        if not target.isascii():
            return None
        try:
            if (
                not os.path.isfile(target)
                or os.path.getsize(target) != os.path.getsize(source)
            ):
                shutil.copyfile(source, target)
        except OSError as exc:
            logger.warning("复制人脸级联文件失败: %s", exc)
            return None
        _CASCADE_CACHE[name] = target
        return target


def _face_ratio(video_path: str) -> tuple[float, int] | None:
    """等间隔采样视频帧统计人脸检出率；长视频使用 seek，避免逐帧解码。"""
    try:
        import cv2
    except ImportError:
        return None

    cfg = get_settings()["vision"]
    max_frames = max(1, int(cfg["max_frames"]))
    scale_factor = float(cfg["scale_factor"])
    min_neighbors = int(cfg["min_neighbors"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        names = [_CASCADE_NAMES[0]]
        if bool(cfg.get("profile_cascade", True)):
            names.append(_CASCADE_NAMES[1])
        cascades = []
        for name in names:
            path = _cascade_path(cv2, name)
            if not path:
                continue
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                cascades.append(cascade)
        if not cascades:
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        total = detected = 0

        if frame_count > max_frames * 2:
            # 长视频：跳到等间隔帧，只解码需要的帧
            if max_frames == 1:
                indices = [0]
            else:
                indices = [
                    round(i * (frame_count - 1) / (max_frames - 1))
                    for i in range(max_frames)
                ]
            for idx in indices:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)):
                    continue
                ok, frame = cap.read()
                if not ok:
                    continue
                total += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if any(
                    len(c.detectMultiScale(gray, scale_factor, min_neighbors)) > 0
                    for c in cascades
                ):
                    detected += 1
        else:
            step = max(frame_count // max_frames, 1) if frame_count else 1
            idx = 0
            while total < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    total += 1
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if any(
                        len(c.detectMultiScale(gray, scale_factor, min_neighbors)) > 0
                        for c in cascades
                    ):
                        detected += 1
                idx += 1

        if total == 0:
            return None
        return detected / total, total
    except Exception as exc:
        logger.info("视频人脸采样失败，将降级: %s", exc)
        return None
    finally:
        cap.release()


class VisionAnalyzer(BaseAnalyzer):
    modality = "vision"

    def _run(self, inp: AnalysisInput) -> AnalyzerResult:
        if not inp.video_path or not os.path.exists(inp.video_path):
            return AnalyzerResult(
                modality=self.modality, status="degraded", score=70.0,
                metrics={"confidence": 0.0},
                suggestions=["未提供视频，视觉维度按中性基线分处理"],
            )

        res = _face_ratio(inp.video_path)
        if res is None:
            return AnalyzerResult(
                modality=self.modality, status="degraded", score=70.0,
                metrics={"confidence": 0.0},
                suggestions=[
                    "视频解析不可用（需安装 opencv-python，或视频没有可读画面），"
                    "本次为降级评估"
                ],
            )

        ratio, sampled = res
        score = round(50 + ratio * 50, 1)  # 出镜稳定度 → 自信度代理
        suggestions = ([] if ratio >= 0.8 else
                       [f"采样 {sampled} 帧中人脸检出率 {ratio:.0%}，"
                        "建议保持正对镜头、稳定坐姿以展现自信状态"])
        return AnalyzerResult(
            modality=self.modality, status="ok", score=score,
            metrics={"face_ratio": round(ratio, 2), "sampled_frames": sampled,
                     "confidence": round(ratio, 2)},
            suggestions=suggestions or ["镜头状态稳定，精神面貌良好"],
        )
