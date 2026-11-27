"""
ScamShield AI - WebSocket Module.

Provides real-time WebSocket communication for fraud detection updates.
"""

from .endpoints import (
    broadcast_fraud_alert,
    broadcast_scoring_result,
    broadcast_metrics_update,
    broadcast_case_update,
    broadcast_system_event,
    send_to_user,
    init_websocket,
    close_websocket,
    router as websocket_router,
)

__all__ = [
    "broadcast_fraud_alert",
    "broadcast_scoring_result",
    "broadcast_metrics_update",
    "broadcast_case_update",
    "broadcast_system_event",
    "send_to_user",
    "init_websocket",
    "close_websocket",
    "websocket_router",
]