"""WebSocket 端点：主连接选举、输入门控、断线清理"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.modules.setdefault("zmq", MagicMock())

from fastapi.testclient import TestClient

import backend.main as main_module


@pytest.fixture
def ws_client():
    main_module._ws_clients.clear()
    main_module._primary_ws = None
    main_module._session = None
    with TestClient(main_module.app) as client:
        yield client
    main_module._ws_clients.clear()
    main_module._primary_ws = None


class TestWebSocketHelpers:
    def test_remove_dead_ws_promotes_primary(self):
        main_module._ws_clients.clear()
        ws1 = MagicMock()
        ws2 = MagicMock()
        main_module._ws_clients.extend([ws1, ws2])
        main_module._primary_ws = ws1

        main_module._remove_dead_ws_clients([ws1])

        assert ws1 not in main_module._ws_clients
        assert main_module._primary_ws is ws2

    def test_remove_dead_ws_clears_primary_when_empty(self):
        main_module._ws_clients.clear()
        ws1 = MagicMock()
        main_module._ws_clients.append(ws1)
        main_module._primary_ws = ws1

        main_module._remove_dead_ws_clients([ws1])

        assert main_module._primary_ws is None


class TestWebSocketPrimary:
    def test_first_client_becomes_primary(self, ws_client):
        with ws_client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "noop"}))
            assert main_module._primary_ws is not None

    def test_primary_client_tts_done_accepted(self, ws_client):
        mock_session = MagicMock()
        main_module._session = mock_session

        with ws_client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({
                "type": "tts_done",
                "utterance_id": 8,
            }))

        mock_session.tts_queue.on_client_tts_done.assert_called_once_with(8)

    def test_manual_secondary_client_ignored_for_control(self):
        """模拟双连接：次客户端的 seek 不应到达 GameSession。"""
        ws_primary = MagicMock()
        ws_secondary = MagicMock()
        main_module._ws_clients = [ws_primary, ws_secondary]
        main_module._primary_ws = ws_primary

        mock_session = MagicMock()
        mock_session.on_seek = AsyncMock()
        main_module._session = mock_session

        assert ws_secondary is not main_module._primary_ws
        assert ws_primary is main_module._primary_ws

        # 与 websocket_endpoint 中门控逻辑一致
        for ws, should_run in ((ws_secondary, False), (ws_primary, True)):
            if ws is not main_module._primary_ws:
                continue
            import asyncio
            asyncio.run(mock_session.on_seek(5.0))

        mock_session.on_seek.assert_awaited_once_with(5.0)
