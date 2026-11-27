"""
WebSocket API Endpoints for Real-Time Fraud Detection.

Provides WebSocket endpoints for:
- Live fraud alerts streaming
- Real-time scoring results
- Dashboard metrics updates
- Case/investigation notifications
- System health/status updates
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from upishield_core.api.websocket.manager import (
    ConnectionManager,
    get_websocket_manager,
    init_websocket_manager,
    close_websocket_manager,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/ws", tags=["WebSocket"])


# Pydantic Models for WebSocket Messages


class WSMessage(BaseModel):
    """Base WebSocket message structure."""
    type: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None


class WSSubscribeMessage(WSMessage):
    """Subscribe to channels."""
    type: str = "subscribe"
    channels: List[str]


class WSUnsubscribeMessage(WSMessage):
    """Unsubscribe from channels."""
    type: str = "unsubscribe"
    channels: List[str]


class WSPingMessage(WSMessage):
    """Ping/Pong heartbeat."""
    type: str = "ping"


class WSPongMessage(WSMessage):
    """Pong response."""
    type: str = "pong"


class WSAuthMessage(WSMessage):
    """Authentication message."""
    type: str = "auth"
    token: str
    user_id: Optional[str] = None


class WSAckMessage(WSMessage):
    """Acknowledgment message."""
    type: str = "ack"
    original_type: str
    success: bool
    message: Optional[str] = None


class WSErrorMessage(WSMessage):
    """Error message."""
    type: str = "error"
    code: str
    message: str


# Message types for server -> client
class WSAlertMessage(WSMessage):
    """Real-time fraud alert."""
    type: str = "alert"
    alert: Dict[str, Any]


class WSScoringMessage(WSMessage):
    """Live scoring result."""
    type: str = "scoring"
    result: Dict[str, Any]


class WSMetricsMessage(WSMessage):
    """Dashboard metrics update."""
    type: str = "metrics"
    metrics: Dict[str, Any]


class WSCaseMessage(WSMessage):
    """Case/investigation update."""
    type: str = "case"
    case: Dict[str, Any]


class WSSystemMessage(WSMessage):
    """System notification."""
    type: str = "system"
    event: str
    data: Dict[str, Any]


# WebSocket Endpoint


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token"),
    user_id: Optional[str] = Query(None, description="User ID"),
    client_version: Optional[str] = Query(None, description="Client version"),
    client_platform: Optional[str] = Query(None, description="Client platform"),
):
    """
    Main WebSocket endpoint for real-time updates.

    Query Parameters:
    - token: Optional authentication token
    - user_id: Optional user ID for user-specific messages
    - client_version: Client version for compatibility
    - client_platform: Client platform (web, mobile, desktop)

    Message Protocol:
    - Client -> Server: subscribe, unsubscribe, ping, auth
    - Server -> Client: alert, scoring, metrics, case, system, ack, error, pong
    """
    manager = get_websocket_manager()

    # Prepare client info
    client_info = {}
    if client_version:
        client_info["version"] = client_version
    if client_platform:
        client_info["platform"] = client_platform

    # Connect
    conn_id = await manager.connect(
        websocket,
        user_id=user_id,
        client_info=client_info,
    )

    # Send welcome message
    await manager.send_to_connection(conn_id, {
        "type": "welcome",
        "connection_id": conn_id,
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Connected to ScamShield AI real-time stream",
        "available_channels": ["alerts", "scoring", "metrics", "cases", "system"],
    })

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await _handle_client_message(manager, conn_id, message)
            except json.JSONDecodeError:
                await manager.send_to_connection(conn_id, {
                    "type": "error",
                    "code": "INVALID_JSON",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.error("websocket_message_error", conn_id=conn_id, error=str(e))
                await manager.send_to_connection(conn_id, {
                    "type": "error",
                    "code": "PROCESSING_ERROR",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        logger.info("websocket_client_disconnected", conn_id=conn_id)
    except Exception as e:
        logger.error("websocket_error", conn_id=conn_id, error=str(e))
    finally:
        await manager.disconnect(conn_id)


async def _handle_client_message(
    manager: ConnectionManager,
    conn_id: str,
    message: Dict[str, Any],
):
    """Handle incoming client message."""
    msg_type = message.get("type")
    request_id = message.get("request_id")

    async def send_ack(success: bool, msg: str = ""):
        await manager.send_to_connection(conn_id, {
            "type": "ack",
            "original_type": msg_type,
            "success": success,
            "message": msg,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

    if msg_type == "subscribe":
        channels = message.get("channels", [])
        results = []
        for channel in channels:
            success = await manager.subscribe(conn_id, channel)
            results.append({"channel": channel, "success": success})

        await send_ack(True, f"Subscribed to {sum(1 for r in results if r['success'])}/{len(results)} channels")
        await manager.send_to_connection(conn_id, {
            "type": "subscription_update",
            "subscriptions": results,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

    elif msg_type == "unsubscribe":
        channels = message.get("channels", [])
        results = []
        for channel in channels:
            success = await manager.unsubscribe(conn_id, channel)
            results.append({"channel": channel, "success": success})

        await send_ack(True, f"Unsubscribed from {sum(1 for r in results if r['success'])}/{len(results)} channels")
        await manager.send_to_connection(conn_id, {
            "type": "subscription_update",
            "subscriptions": results,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

    elif msg_type == "ping":
        await manager.send_to_connection(conn_id, {
            "type": "pong",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Update heartbeat
        if conn_id in manager.connection_metadata:
            manager.connection_metadata[conn_id]["missed_heartbeats"] = 0
            manager.connection_metadata[conn_id]["last_heartbeat"] = datetime.utcnow().isoformat()

    elif msg_type == "auth":
        # Handle authentication (if needed)
        token = message.get("token")
        new_user_id = message.get("user_id")

        if new_user_id:
            # Update user association
            old_user_id = manager.connection_metadata[conn_id].get("user_id")
            if old_user_id and old_user_id in manager.user_connections:
                manager.user_connections[old_user_id].discard(conn_id)

            manager.user_connections[new_user_id].add(conn_id)
            manager.connection_metadata[conn_id]["user_id"] = new_user_id

            # Deliver queued messages
            if new_user_id in manager.offline_messages:
                for msg in manager.offline_messages[new_user_id]:
                    await manager.send_to_connection(conn_id, msg)
                del manager.offline_messages[new_user_id]

        await send_ack(True, "Authentication updated")

    elif msg_type == "get_status":
        # Return connection status
        info = manager.get_connection_info(conn_id)
        await manager.send_to_connection(conn_id, {
            "type": "status",
            "connection": info,
            "manager_stats": manager.get_stats(),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

    else:
        await manager.send_to_connection(conn_id, {
            "type": "error",
            "code": "UNKNOWN_MESSAGE_TYPE",
            "message": f"Unknown message type: {msg_type}",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        })


# Helper Functions for Broadcasting


async def broadcast_fraud_alert(alert: Dict[str, Any]):
    """Broadcast a fraud alert to all subscribed connections."""
    manager = get_websocket_manager()
    await manager.broadcast_to_channel("alerts", {
        "type": "alert",
        "alert": alert,
        "timestamp": datetime.utcnow().isoformat(),
    })


async def broadcast_scoring_result(result: Dict[str, Any]):
    """Broadcast a scoring result to all subscribed connections."""
    manager = get_websocket_manager()
    await manager.broadcast_to_channel("scoring", {
        "type": "scoring",
        "result": result,
        "timestamp": datetime.utcnow().isoformat(),
    })


async def broadcast_metrics_update(metrics: Dict[str, Any]):
    """Broadcast dashboard metrics update."""
    manager = get_websocket_manager()
    await manager.broadcast_to_channel("metrics", {
        "type": "metrics",
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat(),
    })


async def broadcast_case_update(case: Dict[str, Any]):
    """Broadcast case/investigation update."""
    manager = get_websocket_manager()
    await manager.broadcast_to_channel("cases", {
        "type": "case",
        "case": case,
        "timestamp": datetime.utcnow().isoformat(),
    })


async def broadcast_system_event(event: str, data: Dict[str, Any]):
    """Broadcast system event."""
    manager = get_websocket_manager()
    await manager.broadcast_to_channel("system", {
        "type": "system",
        "event": event,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })


async def send_to_user(user_id: str, message: Dict[str, Any]):
    """Send a message to a specific user."""
    manager = get_websocket_manager()
    await manager.broadcast_to_user(user_id, message)


# REST Endpoints for WebSocket Management


@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket manager statistics."""
    manager = get_websocket_manager()
    return manager.get_stats()


@router.get("/connections")
async def list_connections():
    """List active WebSocket connections."""
    manager = get_websocket_manager()
    return {"connections": manager.get_active_connections()}


@router.post("/broadcast/alert")
async def broadcast_alert_endpoint(alert: Dict[str, Any]):
    """Broadcast a fraud alert (for testing/integration)."""
    await broadcast_fraud_alert(alert)
    return {"status": "broadcast_sent"}


@router.post("/broadcast/scoring")
async def broadcast_scoring_endpoint(result: Dict[str, Any]):
    """Broadcast a scoring result (for testing/integration)."""
    await broadcast_scoring_result(result)
    return {"status": "broadcast_sent"}


@router.post("/broadcast/metrics")
async def broadcast_metrics_endpoint(metrics: Dict[str, Any]):
    """Broadcast metrics update (for testing/integration)."""
    await broadcast_metrics_update(metrics)
    return {"status": "broadcast_sent"}


@router.post("/broadcast/case")
async def broadcast_case_endpoint(case: Dict[str, Any]):
    """Broadcast case update (for testing/integration)."""
    await broadcast_case_update(case)
    return {"status": "broadcast_sent"}


@router.post("/send/user/{user_id}")
async def send_to_user_endpoint(user_id: str, message: Dict[str, Any]):
    """Send message to specific user (for testing/integration)."""
    sent = await send_to_user(user_id, message)
    return {"status": "sent", "delivered_to": sent}


# Initialization


async def init_websocket():
    """Initialize WebSocket manager on application startup."""
    await init_websocket_manager()
    logger.info("websocket_initialized")


async def close_websocket():
    """Close WebSocket manager on application shutdown."""
    await close_websocket_manager()
    logger.info("websocket_closed")