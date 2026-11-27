"""
Kafka Producer for ScamShield AI Event Streaming.

Provides reliable, high-throughput event publishing for:
- Transaction events
- Fraud alerts
- Scoring results
- Model predictions
- Audit logs
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

import structlog
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Event types for Kafka topics."""
    TRANSACTION = "transaction"
    FRAUD_ALERT = "fraud_alert"
    SCORING_RESULT = "scoring_result"
    MODEL_PREDICTION = "model_prediction"
    CASE_UPDATE = "case_update"
    AUDIT_LOG = "audit_log"
    SYSTEM_METRIC = "system_metric"
    USER_ACTION = "user_action"


class EventPriority(str, Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class KafkaEvent:
    """Standardized Kafka event structure."""
    event_type: EventType
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "scamshield-api"
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class ProducerConfig:
    """Kafka producer configuration."""
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "scamshield-producer"
    acks: str = "all"
    # retries: int = 3  # Not supported in aiokafka 0.15+
    retry_backoff_ms: int = 100
    max_batch_size: int = 16384  # aiokafka 0.15+ uses max_batch_size
    linger_ms: int = 5
    # buffer_memory: int = 33554432  # Not supported in aiokafka 0.15+
    compression_type: str = "gzip"  # snappy not available by default
    # max_in_flight_requests_per_connection: int = 5  # Not supported in aiokafka 0.15+
    enable_idempotence: bool = True
    transactional_id: Optional[str] = None
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_plain_username: Optional[str] = None
    sasl_plain_password: Optional[str] = None


class KafkaProducer:
    """
    High-level Kafka producer with:
    - Automatic connection management
    - Message serialization and partitioning
    - Delivery callbacks and error handling
    - Transactional support for exactly-once semantics
    - Metrics and monitoring
    """

    def __init__(self, config: Optional[ProducerConfig] = None):
        self.config = config or ProducerConfig()
        self._producer: Optional[AIOKafkaProducer] = None
        self._running = False
        self._send_callbacks: List[Callable] = []

        # Metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_failed": 0,
            "bytes_sent": 0,
            "send_latency_ms": [],
            "errors_by_type": {},
        }

    async def start(self):
        """Start the producer."""
        if self._running:
            return

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            client_id=self.config.client_id,
            acks=self.config.acks,
            retry_backoff_ms=self.config.retry_backoff_ms,
            max_batch_size=self.config.max_batch_size,
            linger_ms=self.config.linger_ms,
            compression_type=self.config.compression_type,
            enable_idempotence=self.config.enable_idempotence,
            transactional_id=self.config.transactional_id,
            security_protocol=self.config.security_protocol,
            sasl_mechanism=self.config.sasl_mechanism,
            sasl_plain_username=self.config.sasl_plain_username,
            sasl_plain_password=self.config.sasl_plain_password,
            value_serializer=lambda v: v.encode("utf-8") if isinstance(v, str) else json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )

        await self._producer.start()
        self._running = True

        if self.config.transactional_id:
            await self._producer.init_transactions()

        logger.info("kafka_producer_started", config=self.config.bootstrap_servers)

    async def stop(self):
        """Stop the producer."""
        if not self._running:
            return

        if self._producer:
            await self._producer.stop()
        self._running = False
        logger.info("kafka_producer_stopped")

    async def send(
        self,
        topic: str,
        event: KafkaEvent,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        Send an event to Kafka.

        Args:
            topic: Target topic
            event: KafkaEvent to send
            key: Optional partition key
            partition: Optional explicit partition
            callback: Optional delivery callback

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._running or not self._producer:
            logger.error("producer_not_running")
            return False

        # Use correlation_id as key if no key provided
        if key is None and event.correlation_id:
            key = event.correlation_id

        message = event.to_json()
        start_time = datetime.utcnow()

        try:
            future = self._producer.send_and_wait(
                topic=topic,
                value=message,
                key=key,
                partition=partition,
            )

            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Update metrics
            self.metrics["messages_sent"] += 1
            self.metrics["bytes_sent"] += len(message)
            self.metrics["send_latency_ms"].append(latency_ms)
            if len(self.metrics["send_latency_ms"]) > 1000:
                self.metrics["send_latency_ms"] = self.metrics["send_latency_ms"][-1000:]

            # Call callbacks
            for cb in self._send_callbacks:
                try:
                    await cb(event, True, None)
                except Exception:
                    pass
            if callback:
                try:
                    await callback(event, True, None)
                except Exception:
                    pass

            logger.debug("kafka_message_sent", topic=topic, event_id=event.event_id, latency_ms=latency_ms)
            return True

        except KafkaError as e:
            self.metrics["messages_failed"] += 1
            error_type = type(e).__name__
            self.metrics["errors_by_type"][error_type] = self.metrics["errors_by_type"].get(error_type, 0) + 1

            # Call callbacks
            for cb in self._send_callbacks:
                try:
                    await cb(event, False, str(e))
                except Exception:
                    pass
            if callback:
                try:
                    await callback(event, False, str(e))
                except Exception:
                    pass

            logger.error("kafka_send_failed", topic=topic, event_id=event.event_id, error=str(e))
            return False

        except Exception as e:
            self.metrics["messages_failed"] += 1
            logger.error("kafka_send_error", topic=topic, event_id=event.event_id, error=str(e))
            return False

    async def send_batch(
        self,
        topic: str,
        events: List[KafkaEvent],
        key_func: Optional[Callable[[KafkaEvent], str]] = None,
    ) -> int:
        """
        Send multiple events in a batch.

        Returns:
            Number of successfully sent messages
        """
        if not self._running or not self._producer:
            return 0

        sent = 0
        for event in events:
            key = key_func(event) if key_func else None
            if await self.send(topic, event, key=key):
                sent += 1
        return sent

    async def send_transactional(
        self,
        topic: str,
        events: List[KafkaEvent],
        key_func: Optional[Callable[[KafkaEvent], str]] = None,
    ) -> bool:
        """
        Send events in a transaction for exactly-once semantics.
        """
        if not self._running or not self._producer:
            return False

        if not self.config.transactional_id:
            logger.warning("transactional_id_not_configured")
            return False

        try:
            await self._producer.begin_transaction()
            for event in events:
                key = key_func(event) if key_func else None
                await self._producer.send_and_wait(
                    topic=topic,
                    value=event.to_json(),
                    key=key,
                )
            await self._producer.commit_transaction()
            return True
        except Exception as e:
            try:
                await self._producer.abort_transaction()
            except Exception:
                pass
            logger.error("transactional_send_failed", error=str(e))
            return False

    def add_send_callback(self, callback: Callable):
        """Add a callback for send completion."""
        self._send_callbacks.append(callback)

    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        latency = self.metrics["send_latency_ms"]
        return {
            "messages_sent": self.metrics["messages_sent"],
            "messages_failed": self.metrics["messages_failed"],
            "bytes_sent": self.metrics["bytes_sent"],
            "avg_latency_ms": sum(latency) / len(latency) if latency else 0,
            "p99_latency_ms": sorted(latency)[int(len(latency) * 0.99)] if latency else 0,
            "errors_by_type": self.metrics["errors_by_type"],
            "running": self._running,
        }


# Convenience functions for common event types
async def create_transaction_event(
    transaction_data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> KafkaEvent:
    """Create a transaction event."""
    return KafkaEvent(
        event_type=EventType.TRANSACTION,
        payload=transaction_data,
        correlation_id=correlation_id,
        user_id=user_id,
        priority=EventPriority.NORMAL,
    )


async def create_fraud_alert_event(
    alert_data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> KafkaEvent:
    """Create a fraud alert event."""
    return KafkaEvent(
        event_type=EventType.FRAUD_ALERT,
        payload=alert_data,
        correlation_id=correlation_id,
        user_id=user_id,
        priority=EventPriority.HIGH,
    )


async def create_scoring_result_event(
    scoring_data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> KafkaEvent:
    """Create a scoring result event."""
    return KafkaEvent(
        event_type=EventType.SCORING_RESULT,
        payload=scoring_data,
        correlation_id=correlation_id,
        user_id=user_id,
        priority=EventPriority.NORMAL,
    )


# Global producer instance
_producer: Optional[KafkaProducer] = None


def get_kafka_producer() -> KafkaProducer:
    """Get global Kafka producer instance."""
    global _producer
    if _producer is None:
        _producer = KafkaProducer()
    return _producer


async def init_kafka_producer(config: Optional[ProducerConfig] = None) -> KafkaProducer:
    """Initialize and start global Kafka producer."""
    global _producer
    _producer = KafkaProducer(config)
    await _producer.start()
    return _producer


async def close_kafka_producer():
    """Close global Kafka producer."""
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None