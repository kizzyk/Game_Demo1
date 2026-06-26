"""VLM 用户消息构造（Claude / OpenAI 兼容网关共用）。"""

from __future__ import annotations

from backend.fast.event import EventType, GameEvent
from backend.nitrogen.parser import PerceptionSignal

SYSTEM_PROMPT = """你是一个游戏语音教练，正在实时陪伴玩家观看游戏视频录像。
旁边有一个 AI 系统（NitroGen）从视频帧预测操控并输出关键动作时间线。

你的职责：
- 结合当前画面、关键动作 JSON 时间线、实时感知，给出简短有价值的建议或回答
- 1~2 句话，不超过 40 字，口语化
- 不要重复快通道刚说过的内容

约束：不用列表/Markdown；不超过 40 字。"""


def build_task_section(event: GameEvent, user_question: str) -> tuple[str, str]:
    if user_question:
        return (
            f"玩家提问：{user_question}",
            "直接回答玩家问题，结合画面与动作时间线。",
        )
    if event.type == EventType.PATTERN_COMPLETED:
        return (
            "触发原因：玩家刚结束一段操作",
            "总结刚才操作，给一句点评。",
        )
    if event.type == EventType.ATTACK_WINDOW:
        return (
            "触发原因：检测到进攻窗口",
            "说明为何此时可进攻。",
        )
    return (
        f"触发原因：{event.type.value}",
        "给出当前局面下最有价值的一句建议。",
    )


def build_user_text(
    event: GameEvent,
    ctx_summary: str,
    last_fast_text: str,
    actions_timeline_text: str,
    user_question: str = "",
) -> str:
    signal: PerceptionSignal = event.perception
    task_desc, guidance = build_task_section(event, user_question)

    return (
        f"{ctx_summary}\n\n"
        f"{actions_timeline_text}\n\n"
        f"当前帧实时感知（NitroGen）:\n"
        f"- steer={signal.steer:+.2f} throttle={signal.throttle} brake={signal.brake}\n"
        f"- intent={signal.primary_intent} conf={signal.confidence:.0%}\n"
        f"- 方向={signal.move_direction or '无'}\n\n"
        f"快通道刚才已播报：\"{last_fast_text}\"\n\n"
        f"{task_desc}\n{guidance}"
    )
