"""
TTS 引擎封装（edge-tts）。

Fix 14：TTS 音频改为发送给前端播放，不再在服务端用 pygame 播放。
合成完成后通过 on_audio_data 回调将 MP3 bytes 传出，
由 TTSQueue 广播给所有 WebSocket 客户端。
on_complete 基于实际音频时长估算后定时触发（而非等待本地播放结束）。

声音调优（3号）：
- 可用中文声音（部分）：
    zh-CN-YunxiNeural    男声，自然
    zh-CN-YunyangNeural  男声，新闻播报风
    zh-CN-XiaoxiaoNeural 女声，自然活泼
    zh-CN-XiaohanNeural  女声，沉稳
- rate 参数："+20%" 到 "+40%" 适合游戏场景（紧迫感）
- 完整声音列表：python -m edge_tts --list-voices | findstr zh-CN
"""

from __future__ import annotations
import asyncio
import io
import logging
import threading
from typing import Callable, Optional

import edge_tts

logger = logging.getLogger(__name__)

PRELOAD_TEXTS = ["向左闪！", "注意，快闪！", "有机会，打！", "进攻！", "注意！"]


class TTSEngine:
    """
    edge-tts 封装。
    合成 MP3 bytes → 调用 on_audio_data 发往前端 → 定时触发 on_complete。
    """

    def __init__(self, voice: str = "zh-CN-YunxiNeural", rate: str = "+20%"):
        self.voice = voice
        self.rate  = rate

        self._stop_flag        = threading.Event()
        self._completion_timer: Optional[threading.Timer] = None
        self._cache: dict[str, bytes] = {}

        # Fix 14：音频数据回调，由外部设置（TTSQueue 在初始化后注入）
        self.on_audio_data: Optional[Callable[[bytes], None]] = None

    def preload(self, texts: list[str] | None = None):
        """预合成常用短语，存入内存缓存"""
        targets = texts or PRELOAD_TEXTS
        for text in targets:
            try:
                asyncio.run(self._async_preload(text))
            except RuntimeError:
                # 已有 event loop（如在 async 上下文启动时），跳过预缓存
                logger.warning("TTS preload skipped (event loop conflict): %s", text)
        logger.info("TTS preloaded %d phrases", len(self._cache))

    async def _async_preload(self, text: str):
        try:
            data = await self._synthesize(text)
            self._cache[text] = data
        except Exception as e:
            logger.warning("Preload failed for '%s': %s", text, e)

    def speak_async(self, text: str, on_complete: Optional[Callable] = None):
        """
        异步合成并发送（在新线程中执行，不阻塞调用方）。
        合成完成后：
          1. 调用 on_audio_data(mp3_bytes) → 发往前端播放
          2. 根据音频时长估算播放完成时刻，定时触发 on_complete
        """
        self._stop_flag.clear()
        if self._completion_timer:
            self._completion_timer.cancel()
            self._completion_timer = None

        threading.Thread(
            target=self._speak_thread,
            args=(text, on_complete),
            daemon=True,
            name="tts-speak",
        ).start()

    def stop(self):
        """立即中止：取消定时器（前端播放由前端自行停止）"""
        self._stop_flag.set()
        if self._completion_timer:
            self._completion_timer.cancel()
            self._completion_timer = None

    # ── 内部实现 ──────────────────────────────────────────────────────

    def _speak_thread(self, text: str, on_complete: Optional[Callable]):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_data = loop.run_until_complete(self._synthesize_cached(text))
        except Exception as e:
            logger.error("TTS synthesis error: %s", e)
            if on_complete:
                on_complete()
            return

        if self._stop_flag.is_set():
            return

        if not audio_data:
            if on_complete:
                on_complete()
            return

        # 发送到前端
        if self.on_audio_data:
            try:
                self.on_audio_data(audio_data)
            except Exception as e:
                logger.error("TTS on_audio_data callback error: %s", e)

        if self._stop_flag.is_set():
            return

        # 估算播放时长，定时触发 on_complete
        if on_complete:
            duration = self._estimate_duration(audio_data)
            logger.debug("TTS estimated duration: %.2fs for '%s'", duration, text[:20])
            self._completion_timer = threading.Timer(duration, on_complete)
            self._completion_timer.start()

    async def _synthesize_cached(self, text: str) -> bytes:
        if text in self._cache:
            return self._cache[text]
        data = await self._synthesize(text)
        self._cache[text] = data
        return data

    async def _synthesize(self, text: str) -> bytes:
        """调用 edge-tts API 合成音频，返回 MP3 bytes"""
        communicate = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    @staticmethod
    def _estimate_duration(audio_data: bytes) -> float:
        """
        估算 MP3 音频播放时长（秒）。
        优先使用 pydub 精确解析，失败则按 ~24kbps 估算。
        返回值额外加 0.3s 作为缓冲，确保前端播放完毕后再 unmute ASR。
        """
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(io.BytesIO(audio_data))
            return len(seg) / 1000.0 + 0.3  # pydub 返回毫秒
        except Exception:
            # 粗估：edge-tts 中文语音约 24kbps
            return max(1.0, len(audio_data) * 8 / 24000) + 0.3
