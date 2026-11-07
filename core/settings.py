"""集中配置加载器。

配置优先级（从低到高）：
1. 本模块内置默认值；
2. ``config/settings.json``（可用 ``PRISMAI_SETTINGS_FILE`` 指向其他文件）；
3. 环境变量覆盖（便于部署时不改文件）。

调用方只依赖 ``get_settings()`` 返回的字典，避免模型名、超时、
采样帧数等参数散落在各个分析器中。
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Callable, Dict, Tuple

logger = logging.getLogger(__name__)

_SETTINGS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")
)

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "speech": {
        "model_size": "base",
        "device": "cpu",
        "compute_type": "int8",
        "language": "zh",
        "beam_size": 5,
        "vad_filter": True,
    },
    "content": {
        "ollama_url": "http://localhost:11434/api/generate",
        "model": "qwen2:1.5b",
        "timeout_sec": 30,
    },
    "vision": {
        "max_frames": 30,
        "scale_factor": 1.1,
        "min_neighbors": 5,
        "profile_cascade": True,
    },
    "input": {
        "max_audio_sec": 600,
        "max_video_sec": 600,
        "max_upload_mb": 200,
        "temp_dir": None,
    },
    "storage": {
        "history_limit": 20,
    },
    "auth": {
        "idle_timeout_sec": 3600,
    },
    "concurrency": {
        "max_concurrent_analyses": 1,
        "queue_timeout_sec": 120,
    },
}


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# (环境变量名, 配置路径, 类型转换)
_ENV_SPECS: Tuple[Tuple[str, Tuple[str, str], Callable[[str], Any]], ...] = (
    ("PRISMAI_WHISPER_MODEL", ("speech", "model_size"), str),
    ("PRISMAI_WHISPER_DEVICE", ("speech", "device"), str),
    ("PRISMAI_WHISPER_COMPUTE_TYPE", ("speech", "compute_type"), str),
    ("PRISMAI_WHISPER_LANGUAGE", ("speech", "language"), str),
    ("PRISMAI_WHISPER_BEAM_SIZE", ("speech", "beam_size"), int),
    ("PRISMAI_WHISPER_VAD", ("speech", "vad_filter"), _as_bool),
    ("PRISMAI_OLLAMA_URL", ("content", "ollama_url"), str),
    ("PRISMAI_OLLAMA_MODEL", ("content", "model"), str),
    ("PRISMAI_OLLAMA_TIMEOUT", ("content", "timeout_sec"), int),
    ("PRISMAI_VISION_MAX_FRAMES", ("vision", "max_frames"), int),
    ("PRISMAI_VISION_PROFILE", ("vision", "profile_cascade"), _as_bool),
    ("PRISMAI_MAX_AUDIO_SEC", ("input", "max_audio_sec"), int),
    ("PRISMAI_MAX_VIDEO_SEC", ("input", "max_video_sec"), int),
    ("PRISMAI_MAX_UPLOAD_MB", ("input", "max_upload_mb"), int),
    ("PRISMAI_TEMP_DIR", ("input", "temp_dir"), str),
    ("PRISMAI_AUTH_IDLE_TIMEOUT", ("auth", "idle_timeout_sec"), int),
    ("PRISMAI_MAX_CONCURRENT_ANALYSES", ("concurrency", "max_concurrent_analyses"), int),
    ("PRISMAI_QUEUE_TIMEOUT", ("concurrency", "queue_timeout_sec"), int),
)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_file() -> Dict[str, Any]:
    path = os.getenv("PRISMAI_SETTINGS_FILE", _SETTINGS_PATH)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取配置文件失败，使用默认配置: %s", exc)
        return {}


def _apply_env(settings: Dict[str, Any]) -> None:
    for env_name, path, cast in _ENV_SPECS:
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        node = settings
        for part in path[:-1]:
            node = node.setdefault(part, {})
        try:
            node[path[-1]] = cast(raw)
        except (TypeError, ValueError):
            logger.warning("忽略无效环境变量 %s=%r", env_name, raw)


@lru_cache(maxsize=1)
def get_settings() -> Dict[str, Any]:
    """返回合并后的配置；进程内缓存，测试时可调用 reload_settings()。"""
    settings = _deep_merge(_DEFAULT_SETTINGS, _load_file())
    _apply_env(settings)
    return settings


def reload_settings() -> None:
    """清空缓存，下一次 get_settings() 重新读取文件与环境变量。"""
    get_settings.cache_clear()
