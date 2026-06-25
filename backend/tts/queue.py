"""
优先级 TTS 队列：同一时间只播一条，USER_ANSWER 可打断其他，
过期内容自动丢弃，与 ASRHandler 联动实现 mute/unmute。
"""

from __future__ import annotations
import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from backend.tts.engine import TTSEngine
    from backend.asr.handler import ASRHandler

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    USER_ANSWER  = 0   # 最高：用户提问的回答
    FAST_HINT    = 1   # 高：快通道关键提示
    SLOW_ADVICE  = 2   # 中：慢通道建议
    SLOW_SUMMARY = 3   # 低：操作段总结


@dataclass(order=True)
class TTSItem:
    priority:     Priority
    enqueue_time: float
    text:         str   = field(compare=False)
    expire_sec:   float = field(compare=False)


MAX_AGE: dict[Priority, float] = {
    Priority.USER_ANSWER:  30.0,
    Priority.FAST_HINT:     2.0,
    Priority.SLOW_ADVICE:   8.0,
    Priority.SLOW_SUMMARY: 15.0,
}


class TTSQueue:
    """
    优先级播报队列。

    Fix 14：TTS 音频通过 engine.on_audio_data 发往前端，
    队列本身只负责调度顺序和 ASR mute/unmute，不关心音频如何播放。
    """

    def __init__(
        self,
        tts_engine: "TTSEngine",
        asr_handler: Optional["ASRHandler"] = None,
        inter_gap: float = 0.8,
        broadcast_audio: Optional[Callable[[bytes], None]] = None,
    ):
        """
        Args:
            broadcast_audio: 合成完成后发送音频 bytes 的回调（注入到 engine.on_audio_data）
        """
        self._tts   = tts_engine
        self._asr   = asr_handler
        self._gap   = inter_gap

        # Fix 14：将 broadcast_audio 注入 engine
        if broadcast_audio:
            tts_engine.on_audio_data = broadcast_audio

        self._heap: list[TTSItem] = []
        self._lock = threading.Lock()

        self._is_speaking   = False
        self._current_item: Optional[TTSItem] = None

        # WebSocket 广播回调（JSON 事件通知）
        self._on_speak_start: Optional[Callable] = None   # (text, channel)
        self._on_speak_end:   Optional[Callable] = None   # ()

    def set_callbacks(self, on_start=None, on_end=None):
        self._on_speak_start = on_start
        self._on_speak_end   = on_end

    # ── 入队 ──────────────────────────────────────────────────────────

    def push(self, text: str, priority: Priority):
        if not text:
            return
        item = TTSItem(
            priority=priority,
            enqueue_time=time.time(),
            text=text,
            expire_sec=MAX_AGE[priority],
        )
        with self._lock:
            heapq.heappush(self._heap, item)

        if priority == Priority.USER_ANSWER and self._is_speaking:
            self._interrupt()
            threading.Timer(0.05, self._speak_next).start()
        elif not self._is_speaking:
            self._speak_next()

    def clear_by_priority(self, priorities: list[Priority]):
        with self._lock:
            self._heap = [
                item for item in self._heap
                if item.priority not in priorities
            ]
            heapq.heapify(self._heap)

    def clear_and_stop(self):
        with self._lock:
            self._heap.clear()
        self._interrupt()

    # ── 内部调度 ──────────────────────────────────────────────────────

    def _speak_next(self):
        now = time.time()

        with self._lock:
            item = None
            while self._heap:
                candidate = heapq.heappop(self._heap)
                age = now - candidate.enqueue_time
                if age <= candidate.expire_sec:
                    item = candidate
                    break
                else:
                    logger.debug("TTS item expired: %s [%.1fs old]",
                                 candidate.text[:20], age)
            if item is None:
                return

        self._current_item = item
        self._is_speaking  = True

        if self._asr:
            self._asr.mute()

        channel = _priority_to_channel(item.priority)
        if self._on_speak_start:
            self._on_speak_start(item.text, channel)

        logger.info("TTS speak [%s]: %s", channel, item.text)
        self._tts.speak_async(item.text, on_complete=self._on_complete)

    def _on_complete(self):
        self._is_speaking  = False
        self._current_item = None

        if self._asr:
            self._asr.unmute()

        if self._on_speak_end:
            self._on_speak_end()

        threading.Timer(self._gap, self._speak_next).start()

    def _interrupt(self):
        self._tts.stop()
        if self._asr:
            self._asr.unmute()
        self._is_speaking  = False
        self._current_item = None


def _priority_to_channel(p: Priority) -> str:
    return {
        Priority.USER_ANSWER:  "user_answer",
        Priority.FAST_HINT:    "fast",
        Priority.SLOW_ADVICE:  "slow",
        Priority.SLOW_SUMMARY: "slow",
    }.get(p, "slow")
