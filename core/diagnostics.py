"""运行环境自检：给侧边栏「系统状态」提供可读的诊断项。"""
from __future__ import annotations

import os
import sqlite3
import sys
from typing import Any, Dict, List

from core.settings import get_settings

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "prism.db")


def _model_path(model_size: str) -> str | None:
    candidates = [
        os.getenv("HF_HOME"),
        os.path.join(os.path.dirname(__file__), "..", "models"),
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    ]
    repo = f"models--Systran--faster-whisper-{model_size}"
    for root in candidates:
        if not root:
            continue
        snapshots = os.path.join(root, "hub", repo, "snapshots")
        if not os.path.isdir(snapshots):
            continue
        for name in os.listdir(snapshots):
            path = os.path.join(snapshots, name, "model.bin")
            if os.path.isfile(path):
                return path
    return None


def check_environment() -> List[Dict[str, Any]]:
    settings = get_settings()
    items: List[Dict[str, Any]] = []

    items.append(
        {
            "name": "Python",
            "status": "ok" if sys.version_info >= (3, 10) else "warn",
            "detail": f"{sys.version.split()[0]}（建议 3.10+，当前环境为 3.12）",
        }
    )

    try:
        import av
        import faster_whisper

        av_major = int(str(av.__version__).split(".")[0])
        items.append(
            {
                "name": "语音依赖",
                "status": "ok" if av_major < 16 else "warn",
                "detail": (
                    f"faster-whisper {faster_whisper.__version__} / "
                    f"PyAV {av.__version__}"
                    + ("（PyAV 16+ 可能与 faster-whisper 不兼容）" if av_major >= 16 else "")
                ),
            }
        )
    except Exception as exc:
        items.append(
            {
                "name": "语音依赖",
                "status": "error",
                "detail": f"导入失败：{exc}",
            }
        )

    model_size = str(settings["speech"]["model_size"])
    model_path = _model_path(model_size)
    items.append(
        {
            "name": "Whisper 模型",
            "status": "ok" if model_path else "error",
            "detail": (
                f"{model_size} 模型已就绪"
                if model_path else f"未找到 {model_size} 模型，语音将降级"
            ),
        }
    )

    try:
        import cv2

        from analyzers.vision_analyzer import _cascade_path

        cascade = _cascade_path(cv2)
        items.append(
            {
                "name": "视觉依赖",
                "status": "ok" if cascade else "warn",
                "detail": (
                    f"OpenCV {cv2.__version__}，Haar 级联可用"
                    if cascade else f"OpenCV {cv2.__version__}，缺少 Haar 级联"
                ),
            }
        )
    except Exception as exc:
        items.append(
            {
                "name": "视觉依赖",
                "status": "error",
                "detail": f"OpenCV 不可用：{exc}",
            }
        )

    try:
        import requests

        url = str(settings["content"]["ollama_url"]).replace(
            "/api/generate", "/api/tags"
        )
        response = requests.get(url, timeout=0.8)
        items.append(
            {
                "name": "Ollama",
                "status": "ok" if response.ok else "warn",
                "detail": "已连接，内容维度可使用 LLM" if response.ok else "响应异常，将规则降级",
            }
        )
    except Exception:
        items.append(
            {
                "name": "Ollama",
                "status": "warn",
                "detail": "未运行，内容维度将使用规则评分",
            }
        )

    try:
        with sqlite3.connect(_DB_PATH, timeout=2) as conn:
            conn.execute("SELECT 1")
        items.append(
            {"name": "数据库", "status": "ok", "detail": "prism.db 可读写"}
        )
    except Exception as exc:
        items.append(
            {"name": "数据库", "status": "error", "detail": f"不可用：{exc}"}
        )

    return items
