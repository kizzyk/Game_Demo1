"""测试 NitroGenClient 辅助方法"""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("zmq", MagicMock())

from backend.nitrogen.client import NitroGenClient
from backend.nitrogen.parser import PerceptionSignal


def test_clear_signal_removes_latest():
    client = NitroGenClient.__new__(NitroGenClient)
    client._signal_lock = __import__("threading").Lock()
    client._latest_signal = PerceptionSignal(
        primary_intent="ATTACK",
        confidence=0.9,
        move_direction=None,
        move_magnitude=0.0,
    )
    client.clear_signal()
    assert client.latest_signal is None
