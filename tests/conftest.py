"""
公共 fixtures 和辅助函数。
所有测试均可在不启动 NitroGen / Whisper / Claude / edge-tts 的情况下运行。
"""

import sys
import os

# 确保从 demo/ 根目录能找到 backend 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from backend.nitrogen.parser import PerceptionSignal
from backend.fast.event import EventType, GameEvent


# ── NitroGen chunk 构造辅助 ───────────────────────────────────────────

def make_chunk(
    attack: float = 0.0,
    dodge: float  = 0.0,
    jx: float     = 0.0,
    jy: float     = 0.0,
) -> dict:
    """
    构造一个符合 NitroGen 输出格式的 chunk。
    attack / dodge: 对应语义组的按钮激活值（0~1）
    jx / jy: 左摇杆分量（-1~1）
    """
    buttons = np.zeros((16, 21), dtype=np.float32)
    j_left  = np.zeros((16, 2),  dtype=np.float32)

    # parser.py 中定义的按钮分组
    # ATTACK_BUTTONS = EAST(5) SOUTH(18) RIGHT_TRIGGER(16)
    if attack > 0:
        buttons[:, 5]  = attack
        buttons[:, 18] = attack
        buttons[:, 16] = attack

    # DODGE_BUTTONS = LEFT_TRIGGER(9) LEFT_SHOULDER(7) RIGHT_SHOULDER(14)
    if dodge > 0:
        buttons[:, 9]  = dodge
        buttons[:, 7]  = dodge
        buttons[:, 14] = dodge

    j_left[:, 0] = jx
    j_left[:, 1] = jy

    return {
        "j_left":  j_left,
        "j_right": np.zeros((16, 2), dtype=np.float32),
        "buttons": buttons,
    }


def make_signal(
    intent:    str   = "WAIT",
    confidence: float = 0.8,
    direction: str | None = None,
    magnitude: float = 0.0,
) -> PerceptionSignal:
    """快速构造 PerceptionSignal（用于 ActionFilter 测试）"""
    return PerceptionSignal(
        primary_intent=intent,
        confidence=confidence,
        move_direction=direction,
        move_magnitude=magnitude,
        horizon_sequence=[f"{intent}×16"],
    )


def make_event(
    etype:     EventType = EventType.SUDDEN_DODGE,
    timestamp: float     = 0.0,
    signal:    PerceptionSignal | None = None,
    fast:      bool = True,
    slow:      bool = False,
) -> GameEvent:
    """快速构造 GameEvent"""
    if signal is None:
        signal = make_signal("DODGE", 0.9)
    return GameEvent(
        type=etype,
        timestamp=timestamp,
        perception=signal,
        trigger_fast=fast,
        trigger_slow=slow,
    )


# ── TTSEngine / ASRHandler mock fixtures ─────────────────────────────

@pytest.fixture
def mock_tts_engine():
    """
    同步调用 on_complete 的 TTSEngine mock。
    speak_async 会立即（同步）触发 on_complete，方便测试队列流转逻辑。
    """
    engine = MagicMock()
    engine.on_audio_data = None

    def _speak(text, on_complete=None):
        if on_complete:
            on_complete()

    engine.speak_async.side_effect = _speak
    return engine


@pytest.fixture
def mock_asr_handler():
    """最简 ASRHandler mock"""
    asr = MagicMock()
    asr._muted = False
    return asr
