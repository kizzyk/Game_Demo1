"""
ASR 持续收音处理器：Whisper + VAD，自动识别用户提问。
TTS 播报期间暂停 VAD，避免回声误触发。

Fix 13：Whisper 识别改为独立线程池，不再阻塞 VAD。
_flush() 立即将音频放入队列后重置 VAD，识别在后台线程完成，
期间新的语音仍可继续被 VAD 捕捉。

VAD 参数调优（5号）：
- SILENCE_THRESHOLD：在真实环境下测量背景噪声振幅，设置在噪声均值 * 2 左右
- SILENCE_END_SEC：1.2 秒是否合适，正常说话停顿会不会被误判为结束
- TTS_MUTE_TAIL_SEC：0.2 秒是否足够消除尾音
- 安静环境 vs 有背景音（游戏声）分别测试
"""

from __future__ import annotations
import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
import whisper

logger = logging.getLogger(__name__)


class ASRHandler:
    """
    持续收音 + VAD + 非阻塞 Whisper 识别。

    架构：
      WebSocket 线程 → process_audio_chunk() → VAD
                                                  ↓ (语音结束)
                                             _transcription_queue
                                                  ↓
                                     _transcription_thread → whisper.transcribe()
                                                                   ↓
                                                           on_utterance(text)
    """

    # VAD 参数（5号调优）
    SILENCE_THRESHOLD  = 300    # 振幅阈值（0~32768），需在真实环境下校准
    SPEECH_MIN_SEC     = 0.5    # 最短有效语音，过滤误触（清嗓子、轻微背景音）
    SILENCE_END_SEC    = 1.2    # 静音多久判定说话结束（秒）
    TTS_MUTE_TAIL_SEC  = 0.2    # TTS 结束后额外静默时间（消余音）

    def __init__(self, model_size: str = "base", language: str = "zh"):
        """
        Args:
            model_size: Whisper 模型大小（5号评估是否升级到 small/medium）
            language:   识别语言
        """
        logger.info("Loading Whisper model: %s ...", model_size)
        self.model    = whisper.load_model(model_size)
        self.language = language
        logger.info("Whisper loaded")

        # 识别完成回调：callable(text: str)
        self.on_utterance: Optional[Callable[[str], None]] = None

        # TTS 联动状态
        self._muted = False

        # VAD 状态（仅在 WebSocket 线程中访问，无需锁）
        self._speaking       = False
        self._audio_buffer:  list[bytes] = []
        self._speech_frames  = 0
        self._silence_frames = 0

        # Fix 13：独立转写线程 + 队列
        # 队列中存放 float32 numpy array，None 为停止信号
        self._transcription_queue: queue.Queue = queue.Queue(maxsize=4)
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop,
            daemon=True,
            name="asr-transcribe",
        )
        self._transcription_thread.start()

    # ── TTS 联动接口 ──────────────────────────────────────────────────

    def mute(self):
        """TTSQueue 开始播报时调用"""
        self._muted = True
        self._reset_vad()
        logger.debug("ASR muted")

    def unmute(self):
        """TTSQueue 播报结束时调用（含 TTS_MUTE_TAIL_SEC 延迟）"""
        threading.Timer(self.TTS_MUTE_TAIL_SEC, self._do_unmute).start()

    def force_unmute(self):
        """视频 seek 时调用，跳过 tail delay 直接 unmute"""
        self._muted = False
        logger.debug("ASR force unmuted")

    def _do_unmute(self):
        self._muted = False
        logger.debug("ASR unmuted")

    # ── 音频处理接口 ──────────────────────────────────────────────────

    def process_audio_chunk(self, audio_bytes: bytes, sample_rate: int = 16000):
        """
        处理前端发来的 PCM 音频块（WebSocket binary frame）。
        约 100ms/块（1600 samples @ 16kHz）。
        本方法在 WebSocket 协程中调用，必须快速返回，不得阻塞。

        Args:
            audio_bytes: PCM 16bit little-endian
            sample_rate: 采样率（默认 16kHz，需与前端一致）
        """
        if self._muted:
            return

        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(audio) == 0:
            return

        amplitude = float(np.abs(audio).mean())

        chunks_per_sec = sample_rate / len(audio)
        silence_limit  = int(self.SILENCE_END_SEC * chunks_per_sec)
        speech_min     = int(self.SPEECH_MIN_SEC  * chunks_per_sec)

        if amplitude > self.SILENCE_THRESHOLD:
            self._speaking = True
            self._silence_frames = 0
            self._speech_frames += 1
            self._audio_buffer.append(audio_bytes)

        elif self._speaking:
            self._silence_frames += 1
            self._audio_buffer.append(audio_bytes)

            if self._silence_frames >= silence_limit:
                if self._speech_frames >= speech_min:
                    self._flush()   # 非阻塞：仅入队
                self._reset_vad()

    # ── 内部实现 ──────────────────────────────────────────────────────

    def _flush(self):
        """
        将缓冲区音频转为 float32 放入转写队列，立即返回。
        VAD 在此之后立即重置，可继续捕捉下一段语音。
        """
        if not self._audio_buffer:
            return

        raw = b"".join(self._audio_buffer)
        arr = (
            np.frombuffer(raw, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )
        logger.debug("ASR queued %.1fs audio for transcription", len(arr) / 16000)

        try:
            self._transcription_queue.put_nowait(arr)
        except queue.Full:
            logger.warning("ASR transcription queue full, dropping audio")

    def _transcription_loop(self):
        """
        独立转写线程：从队列取音频，调用 Whisper，触发回调。
        阻塞在 queue.get()，Whisper 运行期间 VAD 正常处理新音频。
        """
        while True:
            arr = self._transcription_queue.get()
            if arr is None:   # 停止信号
                break
            try:
                result = self.model.transcribe(
                    arr,
                    language=self.language,
                    fp16=False,
                )
                text = result["text"].strip()
                logger.info("ASR result: %s", text)
                if text and self.on_utterance:
                    self.on_utterance(text)
            except Exception as e:
                logger.error("Whisper transcribe error: %s", e)

    def _reset_vad(self):
        self._speaking       = False
        self._audio_buffer   = []
        self._speech_frames  = 0
        self._silence_frames = 0

    def stop(self):
        """关闭转写线程"""
        self._transcription_queue.put(None)
