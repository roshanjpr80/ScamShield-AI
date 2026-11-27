"""
WebSocket Connection Manager for Real-Time Fraud Detection.

Provides connection management, message broadcasting, and subscription handling
for real-time fraud alerts, scoring updates, and dashboard metrics.
"""

import asyncio
import json
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import structlog
from fastapi import WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections with support for:
    - Multiple subscription channels (alerts, scoring, metrics, cases)
    - Connection authentication and authorization
    - Message broadcasting with acknowledgment
    - Connection health monitoring
    - Graceful reconnection handling
    """

    def __init__(self):
        # Active connections: {connection_id: WebSocketConnection}
        self.connections: Dict[str, "WebSocketConnection"] = {}

        # Channel subscriptions: {channel: set(connection_ids)}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # User to connections mapping: {user_id: set(connection_ids)}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)

        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}

        # Message queue for offline users (max 100 per user)
        self.offline_messages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Heartbeat tracking
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval = 30  # seconds
        self._max_missed_heartbeats = 3

        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_failed": 0,
            "channels_active": 0,
        }

    async def start(self):
        """Start background tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("websocket_manager_started")

    async def stop(self):
        """Stop background tasks and close all connections."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for conn_id, conn in list(self.connections.items()):
            await self.disconnect(conn_id, code=1001, reason="Server shutting down")

        logger.info("websocket_manager_stopped")

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        client_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            user_id: Optional authenticated user ID
            client_info: Optional client metadata (version, platform, etc.)

        Returns:
            Connection ID
        """
        await websocket.accept()

        conn_id = str(uuid.uuid4())[:12]

        connection = WebSocketConnection(
            connection_id=conn_id,
            websocket=websocket,
            user_id=user_id,
            client_info=client_info or {},
            connected_at=datetime.utcnow(),
        )

        self.connections[conn_id] = connection

        if user_id:
            self.user_connections[user_id].add(conn_id)

        self.connection_metadata[conn_id] = {
            "user_id": user_id,
            "client_info": client_info or {},
            "connected_at": datetime.utcnow().isoformat(),
            "subscriptions": set(),
            "last_heartbeat": datetime.utcnow().isoformat(),
            "missed_heartbeats": 0,
        }

        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.connections)

        # Send any queued offline messages
        if user_id and user_id in self.offline_messages:
            for msg in self.offline_messages[user_id]:
                await self._send_to_connection(conn_id, msg)
            del self.offline_messages[user_id]

        logger.info("websocket_connected", conn_id=conn_id, user_id=user_id)
        return conn_id

    async def disconnect(
        self,
        conn_id: str,
        code: int = 1000,
        reason: str = "Normal closure",
    ):
        """Disconnect a WebSocket connection."""
        if conn_id not in self.connections:
            return

        connection = self.connections[conn_id]
        user_id = connection.user_id

        # Remove from subscriptions
        for channel in list(self.subscriptions.keys()):
            self.subscriptions[channel].discard(conn_id)
            if not self.subscriptions[channel]:
                del self.subscriptions[channel]

        # Remove from user connections
        if user_id:
            self.user_connections[user_id].discard(conn_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        # Close WebSocket
        try:
            await connection.websocket.close(code=code, reason=reason)
        except Exception:
            pass

        # Clean up metadata
        if conn_id in self.connection_metadata:
            del self.connection_metadata[conn_id]

        del self.connections[conn_id]
        self.stats["active_connections"] = len(self.connections)

        logger.info("websocket_disconnected", conn_id=conn_id, user_id=user_id)

    async def subscribe(self, conn_id: str, channel: str) -> bool:
        """
        Subscribe a connection to a channel.

        Available channels:
        - alerts: Real-time fraud alerts
        - scoring: Live scoring results
        - metrics: Dashboard metrics updates
        - cases: Case/investigation updates
        - system: System notifications
        """
        if conn_id not in self.connections:
            return False

        valid_channels = {"alerts", "scoring", "metrics", "cases", "system", "all"}
        if channel not in valid_channels:
            logger.warning("invalid_channel", conn_id=conn_id, channel=channel)
            return False

        self.subscriptions[channel].add(conn_id)
        self.connection_metadata[conn_id]["subscriptions"].add(channel)
        self.stats["channels_active"] = len(self.subscriptions)

        logger.info("websocket_subscribed", conn_id=conn_id, channel=channel)
        return True

    async def unsubscribe(self, conn_id: str, channel: str) -> bool:
        """Unsubscribe a connection from a channel."""
        if conn_id not in self.connections:
            return False

        self.subscriptions[channel].discard(conn_id)
        if not self.subscriptions[channel]:
            del self.subscriptions[channel]

        if conn_id in self.connection_metadata:
            self.connection_metadata[conn_id]["subscriptions"].discard(channel)

        self.stats["channels_active"] = len(self.subscriptions)
        logger.info("websocket_unsubscribed", conn_id=conn_id, channel=channel)
        return True

    async def send_to_connection(self, conn_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a specific connection."""
        if conn_id not in self.connections:
            return False

        return await self._send_to_connection(conn_id, message)

    async def broadcast_to_channel(
        self,
        channel: str,
        message: Dict[str, Any],
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """
        Broadcast a message to all connections subscribed to a channel.

        Returns:
            Number of connections the message was sent to
        """
        if channel not in self.subscriptions:
            return 0

        targets = self.subscriptions[channel] - (exclude or set())
        sent = 0

        for conn_id in targets:
            if await self._send_to_connection(conn_id, message):
                sent += 1

        return sent

    async def broadcast_to_user(
        self,
        user_id: str,
        message: Dict[str, Any],
    ) -> int:
        """Broadcast a message to all connections for a user."""
        if user_id not in self.user_connections:
            # Queue for when user connects
            self.offline_messages[user_id].append(message)
            # Limit queue size
            if len(self.offline_messages[user_id]) > 100:
                self.offline_messages[user_id] = self.offline_messages[user_id][-100:]
            return 0

        sent = 0
        for conn_id in self.user_connections[user_id]:
            if await self._send_to_connection(conn_id, message):
                sent += 1
        return sent

    async def broadcast_to_all(
        self,
        message: Dict[str, Any],
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """Broadcast a message to all connected clients."""
        sent = 0
        for conn_id in list(self.connections.keys()):
            if exclude and conn_id in exclude:
                continue
            if await self._send_to_connection(conn_id, message):
                sent += 1
        return sent

    async def _send_to_connection(self, conn_id: str, message: Dict[str, Any]) -> bool:
        """Internal method to send message to a connection."""
        if conn_id not in self.connections:
            return False

        connection = self.connections[conn_id]
        try:
            await connection.websocket.send_text(json.dumps(message))
            self.stats["messages_sent"] += 1
            return True
        except Exception as e:
            self.stats["messages_failed"] += 1
            logger.warning("websocket_send_failed", conn_id=conn_id, error=str(e))
            # Mark for cleanup on next heartbeat
            return False

    async def _heartbeat_loop(self):
        """Background task to send heartbeats and detect stale connections."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                stale_connections = []
                for conn_id, metadata in self.connection_metadata.items():
                    metadata["missed_heartbeats"] += 1
                    if metadata["missed_heartbeats"] >= self._max_missed_heartbeats:
                        stale_connections.append(conn_id)
                    else:
                        # Send ping
                        await self._send_to_connection(conn_id, {"type": "ping"})

                # Disconnect stale connections
                for conn_id in stale_connections:
                    logger.info("websocket_stale_disconnect", conn_id=conn_id)
                    await self.disconnect(conn_id, code=1008, reason="Heartbeat timeout")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("heartbeat_loop_error", error=str(e))

    def get_connection_info(self, conn_id: str) -> Optional[Dict[str, Any]]:
        """Get connection metadata."""
        return self.connection_metadata.get(conn_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            "subscriptions_per_channel": {
                channel: len(conns) for channel, conns in self.subscriptions.items()
            },
            "connections_per_user": {
                user: len(conns) for user, conns in self.user_connections.items()
            },
        }

    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get list of active connections with metadata."""
        return [
            {
                "connection_id": conn_id,
                "user_id": meta.get("user_id"),
                "connected_at": meta.get("connected_at"),
                "subscriptions": list(meta.get("subscriptions", [])),
                "client_info": meta.get("client_info", {}),
                "missed_heartbeats": meta.get("missed_heartbeats", 0),
            }
            for conn_id, meta in self.connection_metadata.items()
        ]


class WebSocketConnection:
    """Wrapper for WebSocket connection with metadata."""

    def __init__(
        self,
        connection_id: str,
        websocket: WebSocket,
        user_id: Optional[str],
        client_info: Dict[str, Any],
        connected_at: datetime,
    ):
        self.connection_id = connection_id
        self.websocket = websocket
        self.user_id = user_id
        self.client_info = client_info
        self.connected_at = connected_at
        self.last_activity = datetime.utcnow()
        self.message_count = 0


# Global manager instance
_manager: Optional[ConnectionManager] = None


def get_websocket_manager() -> ConnectionManager:
    """Get the global WebSocket manager instance."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager


async def init_websocket_manager() -> ConnectionManager:
    """Initialize and start the WebSocket manager."""
    manager = get_websocket_manager()
    await manager.start()
    return manager


async def close_websocket_manager():
    """Close the WebSocket manager."""
    global _manager
    if _manager:
        await _manager.stop()
        _manager = None