"""
Kafka Consumer for ScamShield AI Event Streaming.

Provides reliable event consumption with:
- Consumer group management
- Message processing with retries
- Dead letter queue for failed messages
- Metrics and monitoring
- Graceful shutdown
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum

import structlog
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

logger = structlog.get_logger(__name__)


class ProcessingStatus(str, Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class ConsumedMessage:
    """Wrapped consumed message with metadata."""
    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: Dict[str, Any]
    timestamp: datetime
    headers: List[tuple] = field(default_factory=list)
    event_id: Optional[str] = None

    @classmethod
    def from_kafka_message(cls, msg) -> "ConsumedMessage":
        """Create from aiokafka message."""
        try:
            value = json.loads(msg.value.decode("utf-8")) if msg.value else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            value = {"raw": msg.value.decode("utf-8", errors="replace") if msg.value else ""}

        event_id = value.get("event_id") if isinstance(value, dict) else None

        return cls(
            topic=msg.topic,
            partition=msg.partition,
            offset=msg.offset,
            key=msg.key.decode("utf-8") if msg.key else None,
            value=value,
            timestamp=datetime.fromtimestamp(msg.timestamp / 1000) if msg.timestamp else datetime.utcnow(),
            headers=msg.headers or [],
            event_id=event_id,
        )


@dataclass
class ConsumerConfig:
    """Kafka consumer configuration."""
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "scamshield-consumer"
    client_id: Optional[str] = None
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000
    fetch_min_bytes: int = 1
    fetch_max_wait_ms: int = 500
    max_partition_fetch_bytes: int = 1048576
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_plain_username: Optional[str] = None
    sasl_plain_password: Optional[str] = None
    isolation_level: str = "read_committed"


@dataclass
class ProcessingResult:
    """Result of message processing."""
    success: bool
    message: ConsumedMessage
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class MessageProcessor:
    """Base class for message processors."""

    async def process(self, message: ConsumedMessage) -> ProcessingResult:
        """Process a message. Override in subclass."""
        raise NotImplementedError

    async def on_error(self, message: ConsumedMessage, error: Exception):
        """Handle processing error. Override for custom error handling."""
        logger.error("message_processing_error", event_id=message.event_id, error=str(error))


class KafkaConsumer:
    """
    High-level Kafka consumer with:
    - Consumer group management
    - Concurrent message processing
    - Automatic offset management
    - Dead letter queue
    - Metrics and monitoring
    """

    def __init__(
        self,
        topics: List[str],
        processor: MessageProcessor,
        config: Optional[ConsumerConfig] = None,
        max_concurrent: int = 10,
        dead_letter_topic: Optional[str] = None,
    ):
        self.topics = topics
        self.processor = processor
        self.config = config or ConsumerConfig()
        self.max_concurrent = max_concurrent
        self.dead_letter_topic = dead_letter_topic

        self._consumer: Optional[AIOKafkaConsumer] = None
        self._running = False
        self._processing_semaphore: Optional[asyncio.Semaphore] = None
        self._tasks: Set[asyncio.Task] = set()

        # Metrics
        self.metrics = {
            "messages_received": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_dead_lettered": 0,
            "processing_latency_ms": [],
            "errors_by_type": {},
            "last_processed_offset": {},
        }

        # Dead letter producer
        self._dlq_producer: Optional["KafkaProducer"] = None

    async def start(self):
        """Start the consumer."""
        if self._running:
            return

        self._processing_semaphore = asyncio.Semaphore(self.max_concurrent)

        # Initialize consumer
        self._consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=self.config.group_id,
            client_id=self.config.client_id,
            auto_offset_reset=self.config.auto_offset_reset,
            enable_auto_commit=self.config.enable_auto_commit,
            auto_commit_interval_ms=self.config.auto_commit_interval_ms,
            max_poll_records=self.config.max_poll_records,
            max_poll_interval_ms=self.config.max_poll_interval_ms,
            session_timeout_ms=self.config.session_timeout_ms,
            heartbeat_interval_ms=self.config.heartbeat_interval_ms,
            fetch_min_bytes=self.config.fetch_min_bytes,
            fetch_max_wait_ms=self.config.fetch_max_wait_ms,
            max_partition_fetch_bytes=self.config.max_partition_fetch_bytes,
            security_protocol=self.config.security_protocol,
            sasl_mechanism=self.config.sasl_mechanism,
            sasl_plain_username=self.config.sasl_plain_username,
            sasl_plain_password=self.config.sasl_plain_password,
            isolation_level=self.config.isolation_level,
            value_deserializer=lambda v: v,  # We'll handle deserialization
        )

        await self._consumer.start()
        self._running = True

        # Initialize DLQ producer if configured
        if self.dead_letter_topic:
            from upishield_core.streaming.kafka_producer import KafkaProducer, ProducerConfig
            self._dlq_producer = KafkaProducer(ProducerConfig(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=f"{self.config.group_id}-dlq",
            ))
            await self._dlq_producer.start()

        # Start processing loop
        asyncio.create_task(self._consume_loop())

        logger.info("kafka_consumer_started", topics=self.topics, group_id=self.config.group_id)

    async def stop(self):
        """Stop the consumer."""
        if not self._running:
            return

        self._running = False

        # Wait for in-flight tasks
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._consumer:
            await self._consumer.stop()

        if self._dlq_producer:
            await self._dlq_producer.stop()

        logger.info("kafka_consumer_stopped")

    async def _consume_loop(self):
        """Main consumption loop."""
        try:
            async for msg in self._consumer:
                if not self._running:
                    break

                consumed_msg = ConsumedMessage.from_kafka_message(msg)
                self.metrics["messages_received"] += 1

                # Process with semaphore for concurrency control
                task = asyncio.create_task(self._process_with_semaphore(consumed_msg))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

        except Exception as e:
            logger.error("consume_loop_error", error=str(e))
            if self._running:
                # Try to restart
                await asyncio.sleep(5)
                if self._running:
                    asyncio.create_task(self._consume_loop())

    async def _process_with_semaphore(self, message: ConsumedMessage):
        """Process message with concurrency control."""
        async with self._processing_semaphore:
            await self._process_message(message)

    async def _process_message(self, message: ConsumedMessage):
        """Process a single message."""
        start_time = datetime.utcnow()

        try:
            result = await self.processor.process(message)

            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics["processing_latency_ms"].append(latency_ms)
            if len(self.metrics["processing_latency_ms"]) > 1000:
                self.metrics["processing_latency_ms"] = self.metrics["processing_latency_ms"][-1000:]

            if result.success:
                self.metrics["messages_processed"] += 1
                self.metrics["last_processed_offset"][f"{message.topic}-{message.partition}"] = message.offset
            else:
                self.metrics["messages_failed"] += 1
                await self._handle_failure(message, result.error or "Processing returned failure")

        except Exception as e:
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics["messages_failed"] += 1
            error_type = type(e).__name__
            self.metrics["errors_by_type"][error_type] = self.metrics["errors_by_type"].get(error_type, 0) + 1
            await self._handle_failure(message, str(e))

    async def _handle_failure(self, message: ConsumedMessage, error: str):
        """Handle message processing failure."""
        logger.warning("message_processing_failed", event_id=message.event_id, error=error)

        # Try to send to dead letter queue
        if self.dead_letter_topic and self._dlq_producer:
            try:
                from upishield_core.streaming.kafka_producer import KafkaEvent, EventType, EventPriority
                dlq_event = KafkaEvent(
                    event_type=EventType.AUDIT_LOG,
                    payload={
                        "original_topic": message.topic,
                        "original_partition": message.partition,
                        "original_offset": message.offset,
                        "original_key": message.key,
                        "original_value": message.value,
                        "error": error,
                        "failed_at": datetime.utcnow().isoformat(),
                        "retry_count": 0,
                    },
                    correlation_id=message.event_id,
                    priority=EventPriority.HIGH,
                )
                await self._dlq_producer.send(self.dead_letter_topic, dlq_event)
                self.metrics["messages_dead_lettered"] += 1
            except Exception as e:
                logger.error("dlq_send_failed", error=str(e))

        # Call processor error handler
        try:
            await self.processor.on_error(message, Exception(error))
        except Exception:
            pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        latency = self.metrics["processing_latency_ms"]
        return {
            "messages_received": self.metrics["messages_received"],
            "messages_processed": self.metrics["messages_processed"],
            "messages_failed": self.metrics["messages_failed"],
            "messages_dead_lettered": self.metrics["messages_dead_lettered"],
            "avg_latency_ms": sum(latency) / len(latency) if latency else 0,
            "p99_latency_ms": sorted(latency)[int(len(latency) * 0.99)] if latency else 0,
            "errors_by_type": self.metrics["errors_by_type"],
            "last_processed_offset": self.metrics["last_processed_offset"],
            "running": self._running,
            "active_tasks": len(self._tasks),
        }

    async def pause(self):
        """Pause consumption."""
        if self._consumer:
            self._consumer.pause(*self.topics)
            logger.info("kafka_consumer_paused")

    async def resume(self):
        """Resume consumption."""
        if self._consumer:
            self._consumer.resume(*self.topics)
            logger.info("kafka_consumer_resumed")

    async def seek_to_beginning(self):
        """Seek to beginning of partitions."""
        if self._consumer:
            await self._consumer.seek_to_beginning()
            logger.info("kafka_consumer_seek_to_beginning")

    async def seek_to_end(self):
        """Seek to end of partitions."""
        if self._consumer:
            await self._consumer.seek_to_end()
            logger.info("kafka_consumer_seek_to_end")


# Convenience processors
class LoggingProcessor(MessageProcessor):
    """Simple processor that logs messages."""

    async def process(self, message: ConsumedMessage) -> ProcessingResult:
        logger.info("consumed_message", topic=message.topic, event_id=message.event_id, payload=message.value)
        return ProcessingResult(success=True, message=message)


class CallbackProcessor(MessageProcessor):
    """Processor that calls a callback function."""

    def __init__(self, callback: Callable[[ConsumedMessage], Any]):
        self.callback = callback

    async def process(self, message: ConsumedMessage) -> ProcessingResult:
        try:
            result = await self.callback(message)
            if isinstance(result, ProcessingResult):
                return result
            return ProcessingResult(success=bool(result), message=message)
        except Exception as e:
            return ProcessingResult(success=False, message=message, error=str(e))


# Global consumer registry
_consumers: Dict[str, KafkaConsumer] = {}


def get_consumer(name: str) -> Optional[KafkaConsumer]:
    """Get a registered consumer by name."""
    return _consumers.get(name)


async def create_consumer(
    name: str,
    topics: List[str],
    processor: MessageProcessor,
    config: Optional[ConsumerConfig] = None,
    max_concurrent: int = 10,
    dead_letter_topic: Optional[str] = None,
) -> KafkaConsumer:
    """Create and register a consumer."""
    consumer = KafkaConsumer(topics, processor, config, max_concurrent, dead_letter_topic)
    _consumers[name] = consumer
    await consumer.start()
    return consumer


async def start_consumer(name: str):
    """Start a registered consumer."""
    if name in _consumers:
        await _consumers[name].start()


async def stop_consumer(name: str):
    """Stop a registered consumer."""
    if name in _consumers:
        await _consumers[name].stop()
        del _consumers[name]


async def stop_all_consumers():
    """Stop all registered consumers."""
    for name in list(_consumers.keys()):
        await stop_consumer(name)


def get_all_consumer_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all consumers."""
    return {name: consumer.get_metrics() for name, consumer in _consumers.items()}