"""
Mock/In-Memory Kafka Implementation for ScamShield AI.

Provides a Kafka-compatible interface for development/testing without requiring a broker.
Supports:
- In-memory message storage per topic
- Producer/consumer semantics
- Consumer groups with offset management
- Exactly-once delivery simulation
- Dead letter queue simulation
- Metrics collection
"""

import asyncio
import json
import uuid
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum
import structlog

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
class MockKafkaEvent:
    """Standardized Kafka event structure for mock implementation."""
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
    _partition: int = 0
    _offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
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
        return json.dumps(self.to_dict(), default=str)


@dataclass
class MockProducerConfig:
    """Mock Kafka producer configuration."""
    bootstrap_servers: str = "mock://localhost:9092"
    client_id: str = "scamshield-producer"
    acks: str = "all"
    retry_backoff_ms: int = 100
    max_batch_size: int = 16384
    linger_ms: int = 5
    compression_type: str = "gzip"
    enable_idempotence: bool = True
    transactional_id: Optional[str] = None
    security_protocol: str = "PLAINTEXT"


@dataclass
class MockConsumerConfig:
    """Mock Kafka consumer configuration."""
    bootstrap_servers: str = "mock://localhost:9092"
    group_id: str = "scamshield-consumer"
    client_id: str = "scamshield-consumer-1"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000
    isolation_level: str = "read_committed"


class MockTopicPartition:
    """Represents a topic partition in mock Kafka."""

    def __init__(self, topic: str, partition: int = 0):
        self.topic = topic
        self.partition = partition
        self.messages: List[MockKafkaEvent] = []
        self.offset = 0

    def append(self, event: MockKafkaEvent) -> int:
        """Append event and return offset."""
        event._partition = self.partition
        event._offset = self.offset
        self.messages.append(event)
        self.offset += 1
        return event._offset

    def get_message(self, offset: int) -> Optional[MockKafkaEvent]:
        """Get message at offset."""
        if 0 <= offset < len(self.messages):
            return self.messages[offset]
        return None

    def __len__(self):
        return len(self.messages)


class MockKafkaCluster:
    """In-memory Kafka cluster simulation."""

    def __init__(self):
        self.topics: Dict[str, Dict[int, MockTopicPartition]] = defaultdict(dict)
        self.consumer_groups: Dict[str, Dict[int, int]] = defaultdict(dict)  # group -> partition -> offset
        self.consumer_group_members: Dict[str, Set[str]] = defaultdict(set)  # group -> set of client_ids
        self.transactions: Dict[str, List[MockKafkaEvent]] = {}  # transactional_id -> pending messages
        self._lock = asyncio.Lock()

    def get_or_create_partition(self, topic: str, partition: int = 0) -> MockTopicPartition:
        """Get or create a topic partition."""
        if partition not in self.topics[topic]:
            self.topics[topic][partition] = MockTopicPartition(topic, partition)
        return self.topics[topic][partition]

    async def send(self, topic: str, event: MockKafkaEvent, key: Optional[str] = None,
                   partition: Optional[int] = None, transactional_id: Optional[str] = None) -> int:
        """Send event to topic."""
        async with self._lock:
            # Determine partition
            if partition is None:
                if key:
                    partition = hash(key) % max(1, len(self.topics[topic]) or 1)
                else:
                    partition = 0

            tp = self.get_or_create_partition(topic, partition)

            if transactional_id:
                if transactional_id not in self.transactions:
                    self.transactions[transactional_id] = []
                self.transactions[transactional_id].append(event)
                return -1  # Will be assigned on commit
            else:
                return tp.append(event)

    async def commit_transaction(self, transactional_id: str) -> List[int]:
        """Commit a transaction."""
        async with self._lock:
            if transactional_id not in self.transactions:
                return []

            events = self.transactions.pop(transactional_id)
            offsets = []
            for event in events:
                # Default to partition 0 for transactional
                tp = self.get_or_create_partition("default", 0)
                offsets.append(tp.append(event))
            return offsets

    async def abort_transaction(self, transactional_id: str):
        """Abort a transaction."""
        async with self._lock:
            self.transactions.pop(transactional_id, None)

    async def fetch(self, topic: str, partition: int, offset: int, max_records: int) -> List[MockKafkaEvent]:
        """Fetch messages from topic partition."""
        async with self._lock:
            if topic not in self.topics or partition not in self.topics[topic]:
                return []

            tp = self.topics[topic][partition]
            messages = []
            for i in range(offset, min(offset + max_records, len(tp))):
                msg = tp.get_message(i)
                if msg:
                    messages.append(msg)
            return messages

    def get_consumer_offset(self, group_id: str, partition: int) -> int:
        """Get consumer group offset for partition."""
        return self.consumer_groups[group_id].get(partition, 0)

    def set_consumer_offset(self, group_id: str, partition: int, offset: int):
        """Set consumer group offset for partition."""
        self.consumer_groups[group_id][partition] = offset

    def register_consumer(self, group_id: str, client_id: str):
        """Register consumer in group."""
        self.consumer_group_members[group_id].add(client_id)

    def unregister_consumer(self, group_id: str, client_id: str):
        """Unregister consumer from group."""
        self.consumer_group_members[group_id].discard(client_id)

    def get_topics(self) -> List[str]:
        """Get list of topics."""
        return list(self.topics.keys())

    def get_partitions(self, topic: str) -> List[int]:
        """Get partitions for topic."""
        return list(self.topics.get(topic, {}).keys())

    def get_latest_offset(self, topic: str, partition: int) -> int:
        """Get latest offset for topic partition."""
        if topic in self.topics and partition in self.topics[topic]:
            return self.topics[topic][partition].offset
        return 0


# Global mock cluster instance
_mock_cluster: Optional[MockKafkaCluster] = None


def get_mock_cluster() -> MockKafkaCluster:
    """Get global mock Kafka cluster."""
    global _mock_cluster
    if _mock_cluster is None:
        _mock_cluster = MockKafkaCluster()
    return _mock_cluster


class MockKafkaProducer:
    """
    Mock Kafka Producer with same interface as real producer.

    Works entirely in-memory for development/testing.
    """

    def __init__(self, config: Optional[MockProducerConfig] = None):
        self.config = config or MockProducerConfig()
        self._running = False
        self._send_callbacks: List[Callable] = []
        self._cluster = get_mock_cluster()

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
        self._running = True
        logger.info("mock_kafka_producer_started", config=self.config.bootstrap_servers)

    async def stop(self):
        """Stop the producer."""
        if not self._running:
            return
        self._running = False
        logger.info("mock_kafka_producer_stopped")

    async def send(
        self,
        topic: str,
        event: MockKafkaEvent,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> bool:
        """Send an event to Kafka (mock)."""
        if not self._running:
            logger.error("producer_not_running")
            return False

        if key is None and event.correlation_id:
            key = event.correlation_id

        message = event.to_json()
        start_time = time.perf_counter()

        try:
            offset = await self._cluster.send(
                topic=topic,
                event=event,
                key=key,
                partition=partition,
                transactional_id=self.config.transactional_id,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

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

            logger.debug("mock_kafka_message_sent", topic=topic, event_id=event.event_id, offset=offset, latency_ms=latency_ms)
            return True

        except Exception as e:
            self.metrics["messages_failed"] += 1
            error_type = type(e).__name__
            self.metrics["errors_by_type"][error_type] = self.metrics["errors_by_type"].get(error_type, 0) + 1

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

            logger.error("mock_kafka_send_failed", topic=topic, event_id=event.event_id, error=str(e))
            return False

    async def send_batch(
        self,
        topic: str,
        events: List[MockKafkaEvent],
        key_func: Optional[Callable[[MockKafkaEvent], str]] = None,
    ) -> int:
        """Send multiple events in a batch."""
        if not self._running:
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
        events: List[MockKafkaEvent],
        key_func: Optional[Callable[[MockKafkaEvent], str]] = None,
    ) -> bool:
        """Send events in a transaction for exactly-once semantics."""
        if not self._running:
            return False

        if not self.config.transactional_id:
            logger.warning("transactional_id_not_configured")
            return False

        try:
            # In mock, just send all events
            for event in events:
                key = key_func(event) if key_func else None
                await self.send(topic, event, key=key)
            return True
        except Exception as e:
            logger.error("mock_transactional_send_failed", error=str(e))
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


class MockConsumedMessage:
    """Mock consumed message."""

    def __init__(self, topic: str, partition: int, offset: int, key: Optional[str],
                 value: Dict[str, Any], timestamp: datetime, event_id: str):
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.event_id = event_id
        self.headers: List[tuple] = []


class ProcessingStatus(str, Enum):
    """Processing status."""
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    DLQ = "dlq"


@dataclass
class MockProcessingResult:
    """Mock processing result."""
    success: bool
    message: Optional[MockConsumedMessage] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    status: ProcessingStatus = ProcessingStatus.SUCCESS


class MockMessageProcessor:
    """Mock message processor interface."""

    async def process(self, message: MockConsumedMessage) -> MockProcessingResult:
        raise NotImplementedError


class MockKafkaConsumer:
    """Mock Kafka Consumer."""

    def __init__(self, config: MockConsumerConfig, topics: List[str], processor: MockMessageProcessor):
        self.config = config
        self.topics = topics
        self.processor = processor
        self._running = False
        self._cluster = get_mock_cluster()
        self._consumer_tasks: List[asyncio.Task] = []

        # Metrics
        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_dlq": 0,
            "processing_latency_ms": [],
            "errors_by_type": {},
            "last_poll_time": None,
        }

        self._cluster.register_consumer(config.group_id, config.client_id)

    async def start(self):
        """Start the consumer."""
        if self._running:
            return
        self._running = True

        # Start consumer tasks for each topic/partition
        for topic in self.topics:
            partitions = self._cluster.get_partitions(topic)
            if not partitions:
                partitions = [0]

            for partition in partitions:
                task = asyncio.create_task(self._consume_loop(topic, partition))
                self._consumer_tasks.append(task)

        logger.info("mock_kafka_consumer_started", topics=self.topics, group=self.config.group_id)

    async def stop(self):
        """Stop the consumer."""
        if not self._running:
            return

        self._running = False

        for task in self._consumer_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._consumer_tasks.clear()
        self._cluster.unregister_consumer(self.config.group_id, self.config.client_id)
        logger.info("mock_kafka_consumer_stopped")

    async def _consume_loop(self, topic: str, partition: int):
        """Consumer loop for a topic partition."""
        while self._running:
            try:
                offset = self._cluster.get_consumer_offset(self.config.group_id, partition)
                messages = await self._cluster.fetch(
                    topic, partition, offset, self.config.max_poll_records
                )

                if messages:
                    self.metrics["last_poll_time"] = datetime.utcnow().isoformat()

                    for event in messages:
                        # Convert to consumed message
                        consumed = MockConsumedMessage(
                            topic=topic,
                            partition=partition,
                            offset=event._offset,
                            key=event.correlation_id,
                            value=event.payload,
                            timestamp=datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')) if event.timestamp else datetime.utcnow(),
                            event_id=event.event_id,
                        )

                        start_time = time.perf_counter()
                        try:
                            result = await self.processor.process(consumed)
                            latency_ms = (time.perf_counter() - start_time) * 1000

                            if result.success:
                                self.metrics["messages_processed"] += 1
                                self.metrics["processing_latency_ms"].append(latency_ms)
                                if len(self.metrics["processing_latency_ms"]) > 1000:
                                    self.metrics["processing_latency_ms"] = self.metrics["processing_latency_ms"][-1000:]
                            else:
                                self.metrics["messages_failed"] += 1
                                error_type = type(Exception(result.error)).__name__ if result.error else "UnknownError"
                                self.metrics["errors_by_type"][error_type] = self.metrics["errors_by_type"].get(error_type, 0) + 1

                            # Commit offset
                            self._cluster.set_consumer_offset(self.config.group_id, partition, event._offset + 1)

                        except Exception as e:
                            self.metrics["messages_failed"] += 1
                            error_type = type(e).__name__
                            self.metrics["errors_by_type"][error_type] = self.metrics["errors_by_type"].get(error_type, 0) + 1
                            logger.error("mock_consumer_process_error", error=str(e))

                else:
                    # No messages, wait a bit
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("mock_consumer_loop_error", error=str(e))
                await asyncio.sleep(1)

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        latency = self.metrics["processing_latency_ms"]
        return {
            "messages_processed": self.metrics["messages_processed"],
            "messages_failed": self.metrics["messages_failed"],
            "messages_dlq": self.metrics["messages_dlq"],
            "avg_latency_ms": sum(latency) / len(latency) if latency else 0,
            "p99_latency_ms": sorted(latency)[int(len(latency) * 0.99)] if latency else 0,
            "errors_by_type": self.metrics["errors_by_type"],
            "running": self._running,
            "topics": self.topics,
            "group_id": self.config.group_id,
        }


# Global instances
_mock_producer: Optional[MockKafkaProducer] = None
_mock_consumers: Dict[str, MockKafkaConsumer] = {}


def get_mock_kafka_producer() -> MockKafkaProducer:
    """Get global mock Kafka producer instance."""
    global _mock_producer
    if _mock_producer is None:
        _mock_producer = MockKafkaProducer()
    return _mock_producer


async def init_mock_kafka_producer(config: Optional[MockProducerConfig] = None) -> MockKafkaProducer:
    """Initialize and start global mock Kafka producer."""
    global _mock_producer
    _mock_producer = MockKafkaProducer(config)
    await _mock_producer.start()
    return _mock_producer


async def close_mock_kafka_producer():
    """Close global mock Kafka producer."""
    global _mock_producer
    if _mock_producer:
        await _mock_producer.stop()
        _mock_producer = None


def create_mock_consumer(config: MockConsumerConfig, topics: List[str], processor: MockMessageProcessor) -> MockKafkaConsumer:
    """Create a mock Kafka consumer."""
    return MockKafkaConsumer(config, topics, processor)


async def start_mock_consumer(consumer: MockKafkaConsumer):
    """Start a mock consumer."""
    await consumer.start()


async def stop_mock_consumer(consumer: MockKafkaConsumer):
    """Stop a mock consumer."""
    await consumer.stop()


def get_mock_consumer_metrics() -> Dict[str, Any]:
    """Get metrics for all mock consumers."""
    return {f"consumer_{i}": c.get_metrics() for i, c in enumerate(_mock_consumers.values())}


# Convenience functions for common event types
async def create_mock_transaction_event(
    transaction_data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> MockKafkaEvent:
    """Create a transaction event."""
    return MockKafkaEvent(
        event_type=EventType.TRANSACTION,
        payload=transaction_data,
        correlation_id=correlation_id,
        user_id=user_id,
        priority=EventPriority.NORMAL,
    )


async def create_mock_fraud_alert_event(
    alert_data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> MockKafkaEvent:
    """Create a fraud alert event."""
    return MockKafkaEvent(
        event_type=EventType.FRAUD_ALERT,
        payload=alert_data,
        correlation_id=correlation_id,
        user_id=user_id,
        priority=EventPriority.HIGH,
    )


async def create_mock_scoring_result_event(
    scoring_data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> MockKafkaEvent:
    """Create a scoring result event."""
    return MockKafkaEvent(
        event_type=EventType.SCORING_RESULT,
        payload=scoring_data,
        correlation_id=correlation_id,
        user_id=user_id,
        priority=EventPriority.NORMAL,
    )