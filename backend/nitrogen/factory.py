"""按配置创建 NitroGen 实机客户端或模拟客户端。"""

from __future__ import annotations
import logging
import os

from backend.config import Config
from backend.nitrogen.mock_client import MockNitroGenClient

logger = logging.getLogger(__name__)


def nitrogen_mock_enabled(cfg: Config) -> bool:
    env = os.getenv("NITROGEN_MOCK")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return cfg.nitrogen_mock


def create_nitrogen_client(cfg: Config):
    if nitrogen_mock_enabled(cfg):
        return MockNitroGenClient()
    from backend.nitrogen.client import NitroGenClient
    addr = os.getenv("NITROGEN_SERVER", cfg.nitrogen_server)
    return NitroGenClient(server_addr=addr)
