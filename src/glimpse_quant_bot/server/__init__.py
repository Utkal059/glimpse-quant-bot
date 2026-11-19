"""Quant Terminal control plane: FastAPI + WebSocket server around the engine.

Install with ``pip install -e ".[server]"`` and run ``glimpse-quant-bot serve``.
The server is paper-first: live trading needs ``GLIMPSE_LIVE_TRADING=true``,
an API key, ``serve --allow-live`` *and* a typed arming step in the UI.
"""

from glimpse_quant_bot.server.api import create_app
from glimpse_quant_bot.server.controller import (
    EngineController,
    Venue,
    glimpse_venue,
    simulated_venue,
)

__all__ = ["EngineController", "Venue", "create_app", "glimpse_venue", "simulated_venue"]
