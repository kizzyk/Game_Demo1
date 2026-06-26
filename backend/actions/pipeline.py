"""
帧 → mock NitroGen 预测 → 关键动作过滤。

实机接入时：将 mock_predict_from_time 替换为 ZMQ NitroGen 推理即可。
"""

from __future__ import annotations
import io
import logging
import math
from typing import Optional

from PIL import Image

from backend.actions.timeline import ActionTimeline, KeyAction
from backend.nitrogen.controls import signal_from_controls

logger = logging.getLogger(__name__)

# 与 mock_client 类似的演示操控序列
_DEMO_PATTERN = (
    (-0.75, 1, 0, "left_throttle"),
    (0.0, 0, 1, "brake"),
    (0.65, 1, 0, "right_throttle"),
    (0.0, 1, 0, "straight"),
    (-0.3, 1, 0, "slight_left"),
    (0.0, 0, 0, "coast"),
)


def mock_predict_from_time(t_sec: float) -> KeyAction:
    """按视频时间生成确定性 mock 预测（模拟 NitroGen + 后处理前）。"""
    idx = int(t_sec / 2.0) % len(_DEMO_PATTERN)
    steer, throttle, brake, label = _DEMO_PATTERN[idx]
    wobble = 0.12 * math.sin(t_sec * 1.3)
    steer = max(-1.0, min(1.0, steer + wobble))
    sig = signal_from_controls(steer, throttle, brake)
    return KeyAction(
        t_sec=round(t_sec, 3),
        steer=sig.steer,
        throttle=sig.throttle,
        brake=sig.brake,
        intent=sig.primary_intent,
        confidence=sig.confidence,
        label=label,
    )


def mock_predict_from_jpeg(jpeg_bytes: bytes, t_sec: float) -> KeyAction:
    """若有 JPEG 则校验可解码；预测仍基于时间（mock）。"""
    try:
        Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    except Exception as e:
        logger.warning("actions pipeline: bad jpeg at t=%.2f: %s", t_sec, e)
    return mock_predict_from_time(t_sec)


def _is_key_action(candidate: KeyAction, prev: Optional[KeyAction]) -> bool:
    """过滤：只保留操控变化明显的帧。"""
    if prev is None:
        return True
    if candidate.brake == 1 and prev.brake == 0:
        return True
    if candidate.throttle != prev.throttle:
        return True
    if abs(candidate.steer - prev.steer) >= 0.35:
        return True
    if candidate.intent != prev.intent and candidate.confidence >= 0.7:
        return True
    return False


def build_timeline_from_samples(
    samples: list[tuple[float, Optional[bytes]]],
    duration_sec: float,
    sample_interval_sec: float = 2.0,
    min_gap_sec: float = 2.0,
) -> ActionTimeline:
    """
  从 (t_sec, jpeg_bytes?) 列表构建过滤后的时间线。
    """
    timeline = ActionTimeline(
        source="mock_nitrogen",
        duration_sec=duration_sec,
        sample_interval_sec=sample_interval_sec,
    )
    last_kept: Optional[KeyAction] = None
    last_kept_t = -999.0

    for t_sec, jpeg in sorted(samples, key=lambda x: x[0]):
        raw = (
            mock_predict_from_jpeg(jpeg, t_sec)
            if jpeg
            else mock_predict_from_time(t_sec)
        )
        if not _is_key_action(raw, last_kept):
            continue
        if t_sec - last_kept_t < min_gap_sec and last_kept is not None:
            continue
        timeline.key_actions.append(raw)
        last_kept = raw
        last_kept_t = t_sec

    logger.info(
        "Action timeline built: %d key actions from %d samples (mock)",
        len(timeline.key_actions),
        len(samples),
    )
    return timeline


def build_mock_timeline(duration_sec: float, interval: float = 2.0) -> ActionTimeline:
    """无帧输入时按时间网格生成 mock 时间线。"""
    samples = [(t, None) for t in _time_grid(duration_sec, interval)]
    return build_timeline_from_samples(samples, duration_sec, interval)


def _time_grid(duration_sec: float, interval: float) -> list[float]:
    if duration_sec <= 0:
        return [0.0]
    n = max(1, int(duration_sec / interval) + 1)
    return [round(i * interval, 3) for i in range(n) if i * interval <= duration_sec]
