"""VLM 模拟：无 Anthropic API 时根据操控量 + 用户语音生成短回复。"""

from __future__ import annotations
import asyncio
import logging

from backend.fast.event import EventType, GameEvent
from backend.nitrogen.parser import PerceptionSignal

logger = logging.getLogger(__name__)


def _control_hint(signal: PerceptionSignal) -> str:
    parts = [f"转向{signal.steer:+.2f}"]
    parts.append("油门" if signal.throttle else "油门关")
    parts.append("刹车" if signal.brake else "刹车关")
    return "，".join(parts)


async def call_vlm_mock(
    event: GameEvent,
    user_question: str = "",
    *,
    actions_timeline_text: str = "",
    delay_sec: float = 0.35,
) -> str:
    """模拟 VLM：短暂延迟后返回基于操控量 + 时间线的规则回复。"""
    await asyncio.sleep(delay_sec)
    signal = event.perception
    ctrl = _control_hint(signal)
    timeline_hint = (
        actions_timeline_text[:120] + "…"
        if len(actions_timeline_text) > 120
        else actions_timeline_text
    )

    if event.type == EventType.USER_QUESTION and user_question:
        base = f"收到：{user_question[:16]}。当前{ctrl}。"
        if timeline_hint:
            return base + " 已参考动作时间线。"
        return base + " 保持节奏。"

    if signal.brake:
        return "时间线显示有刹车点，可以再提前一点。"
    if signal.throttle and abs(signal.steer) > 0.3:
        side = "左" if signal.steer < 0 else "右"
        return f"向{side}给油，注意看时间线里的转向段。"
    if signal.throttle:
        return "直线油门段，保持节奏。"
    return "可滑行观察下一段关键动作。"
