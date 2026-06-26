# 关键动作时间线 JSON（v1）

视频抽帧 → NitroGen（当前 **mock**）→ 后处理过滤 → 供 VLM 使用的 JSON。

## 生成时机

1. 用户选择视频 → `loadedmetadata`
2. 前端每 **2s** 抽一帧（最多 90 帧）→ `POST /actions/ingest-batch`
3. 后端过滤后写入 `_action_timeline`，`GET /actions/timeline` 可查看

## 格式

```json
{
  "version": 1,
  "source": "mock_nitrogen",
  "duration_sec": 120.0,
  "sample_interval_sec": 2.0,
  "key_actions": [
    {
      "t_sec": 4.0,
      "steer": -0.75,
      "throttle": 1,
      "brake": 0,
      "intent": "NAVIGATE",
      "confidence": 0.8,
      "label": "left_throttle"
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `steer` | [-1, 1] 左右摇杆一维 |
| `throttle` | 0/1 油门 |
| `brake` | 0/1 刹车 |
| `intent` | 派生意图，供快系统兼容 |
| `label` | 简短标签 |

## VLM 使用方式

提问或慢事件触发 VLM 时，将当前时间 ±20s 窗口内的 `key_actions` 摘要拼入 prompt（见 `ActionTimeline.summary_near`）。

## 接实机 NitroGen

替换 `backend/actions/pipeline.py` 中 `mock_predict_from_jpeg` 为 ZMQ 推理即可，**JSON 字段保持不变**。
