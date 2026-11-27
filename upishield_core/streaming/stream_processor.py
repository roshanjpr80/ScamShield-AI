"""
Stream Processor for ScamShield AI.

Provides real-time stream processing capabilities:
- Feature computation from transaction streams
- Fraud pattern detection
- Alert generation and enrichment
- Metrics aggregation
- Integration with ML models and rule engine
"""

import asyncio
import json
import uuid
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set
from collections import defaultdict, deque
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from upishield_core.streaming.kafka_producer import (
    KafkaProducer,
    KafkaEvent,
    EventType,
    EventPriority,
    get_kafka_producer,
    ProducerConfig,
)
from upishield_core.streaming.kafka_consumer import (
    KafkaConsumer,
    ConsumerConfig,
    ConsumedMessage,
    MessageProcessor,
    ProcessingResult,
    create_consumer,
    stop_all_consumers,
)

# Mock Kafka imports
from upishield_core.streaming.mock_kafka import (
    MockKafkaProducer,
    MockKafkaConsumer,
    MockKafkaEvent,
    MockProducerConfig,
    MockConsumerConfig,
    MockConsumedMessage,
    MockProcessingResult,
    MockMessageProcessor,
    get_mock_kafka_producer,
)

logger = structlog.get_logger(__name__)


class StreamProcessorState(str, Enum):
    """Stream processor states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StreamMetrics:
    """Stream processing metrics."""
    events_processed: int = 0
    events_per_second: float = 0.0
    avg_processing_time_ms: float = 0.0
    errors_count: int = 0
    last_event_time: Optional[datetime] = None
    window_start: datetime = field(default_factory=datetime.utcnow)

    def record_event(self, processing_time_ms: float):
        """Record an event processing."""
        self.events_processed += 1
        self.last_event_time = datetime.utcnow()

        # Calculate events per second
        elapsed = (self.last_event_time - self.window_start).total_seconds()
        if elapsed > 0:
            self.events_per_second = self.events_processed / elapsed

        # Update average processing time
        if self.events_processed == 1:
            self.avg_processing_time_ms = processing_time_ms
        else:
            # Exponential moving average
            self.avg_processing_time_ms = 0.9 * self.avg_processing_time_ms + 0.1 * processing_time_ms

    def record_error(self):
        """Record an error."""
        self.errors_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "events_processed": self.events_processed,
            "events_per_second": round(self.events_per_second, 2),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "errors_count": self.errors_count,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "uptime_seconds": (datetime.utcnow() - self.window_start).total_seconds(),
        }


class WindowedAggregator:
    """Time-windowed aggregator for streaming metrics."""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def add(self, key: str, value: float, timestamp: Optional[datetime] = None):
        """Add a value to the window."""
        ts = timestamp or datetime.utcnow()
        self.windows[key].append((ts, value))
        self._cleanup(key, ts)

    def _cleanup(self, key: str, current_time: datetime):
        """Remove expired entries."""
        cutoff = current_time - timedelta(seconds=self.window_seconds)
        window = self.windows[key]
        while window and window[0][0] < cutoff:
            window.popleft()

    def get_sum(self, key: str) -> float:
        """Get sum of values in window."""
        return sum(v for _, v in self.windows.get(key, []))

    def get_count(self, key: str) -> int:
        """Get count of values in window."""
        return len(self.windows.get(key, []))

    def get_avg(self, key: str) -> float:
        """Get average of values in window."""
        values = [v for _, v in self.windows.get(key, [])]
        return sum(values) / len(values) if values else 0.0

    def get_min(self, key: str) -> float:
        """Get minimum value in window."""
        values = [v for _, v in self.windows.get(key, [])]
        return min(values) if values else 0.0

    def get_max(self, key: str) -> float:
        """Get maximum value in window."""
        values = [v for _, v in self.windows.get(key, [])]
        return max(values) if values else 0.0


class StreamProcessor:
    """
    Main stream processor orchestrating:
    - Transaction event consumption
    - Real-time feature computation
    - Fraud detection pipeline
    - Alert generation
    - Metrics aggregation
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        consumer_group: str = "scamshield-processor",
        input_topics: Optional[List[str]] = None,
        output_topic: str = "fraud-events",
        dead_letter_topic: str = "fraud-events-dlq",
        max_concurrent: int = 20,
        use_mock: bool = True,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.consumer_group = consumer_group
        self.input_topics = input_topics or ["transactions", "scoring-requests"]
        self.output_topic = output_topic
        self.dead_letter_topic = dead_letter_topic
        self.max_concurrent = max_concurrent
        self.use_mock = use_mock

        self.state = StreamProcessorState.STOPPED
        self.metrics = StreamMetrics()
        self.aggregator = WindowedAggregator(window_seconds=60)

        # Components (can be real or mock)
        self._producer: Any = None
        self._consumer: Any = None
        self._processor: Optional["StreamMessageProcessor"] = None
        self._running = False
        self._tasks: Set[asyncio.Task] = set()

        # Feature computation cache
        self._feature_cache: Dict[str, Any] = {}
        self._velocity_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Callbacks
        self._on_fraud_detected: List[Callable] = []
        self._on_alert_generated: List[Callable] = []

    async def start(self):
        """Start the stream processor."""
        if self._running:
            return

        self.state = StreamProcessorState.STARTING
        self._running = True

        if self.use_mock:
            # Initialize mock producer
            self._producer = MockKafkaProducer(MockProducerConfig(
                bootstrap_servers=self.bootstrap_servers,
                client_id=f"{self.consumer_group}-producer",
            ))
            await self._producer.start()

            # Initialize processor
            self._processor = StreamMessageProcessor(self)

            # Initialize mock consumer
            self._consumer = MockKafkaConsumer(
                config=MockConsumerConfig(
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.consumer_group,
                    max_poll_records=500,
                ),
                topics=self.input_topics,
                processor=self._processor,
            )
            await self._consumer.start()
        else:
            # Initialize real producer
            self._producer = KafkaProducer(ProducerConfig(
                bootstrap_servers=self.bootstrap_servers,
                client_id=f"{self.consumer_group}-producer",
            ))
            await self._producer.start()

            # Initialize processor
            self._processor = StreamMessageProcessor(self)

            # Initialize real consumer
            self._consumer = KafkaConsumer(
                topics=self.input_topics,
                processor=self._processor,
                config=ConsumerConfig(
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.consumer_group,
                    max_poll_records=500,
                ),
                max_concurrent=self.max_concurrent,
                dead_letter_topic=self.dead_letter_topic,
            )
            await self._consumer.start()

        # Start metrics reporting
        asyncio.create_task(self._metrics_loop())

        self.state = StreamProcessorState.RUNNING
        logger.info("stream_processor_started",
                   topics=self.input_topics,
                   output_topic=self.output_topic,
                   consumer_group=self.consumer_group,
                   use_mock=self.use_mock)

    async def stop(self):
        """Stop the stream processor."""
        if not self._running:
            return

        self._running = False
        self.state = StreamProcessorState.STOPPED

        if self._consumer:
            await self._consumer.stop()

        if self._producer:
            await self._producer.stop()

        logger.info("stream_processor_stopped")

    async def _metrics_loop(self):
        """Periodic metrics reporting."""
        while self._running:
            await asyncio.sleep(30)
            if self._running:
                metrics = self.get_metrics()
                logger.info("stream_processor_metrics", **metrics)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        base = self.metrics.to_dict()

        # Add consumer metrics
        if self._consumer and hasattr(self._consumer, 'get_metrics'):
            base["consumer"] = self._consumer.get_metrics()

        # Add producer metrics
        if self._producer and hasattr(self._producer, 'get_metrics'):
            base["producer"] = self._producer.get_metrics()

        # Add aggregator metrics
        base["aggregator"] = {
            "transactions_per_minute": self.aggregator.get_count("transactions"),
            "fraud_detected_per_minute": self.aggregator.get_count("fraud_detected"),
            "avg_risk_score": self.aggregator.get_avg("risk_score"),
            "max_risk_score": self.aggregator.get_max("risk_score"),
        }

        base["state"] = self.state.value
        base["use_mock"] = self.use_mock
        return base

    # Feature computation methods
    def compute_velocity_features(self, user_id: str, amount: float, timestamp: datetime) -> Dict[str, float]:
        """Compute velocity features for a user."""
        window = self._velocity_cache[user_id]
        window.append((timestamp, amount))

        # Clean old entries (last 5 minutes)
        cutoff = timestamp - timedelta(minutes=5)
        while window and window[0][0] < cutoff:
            window.popleft()

        if not window:
            return {
                "txn_count_5m": 0,
                "txn_sum_5m": 0.0,
                "txn_avg_5m": 0.0,
                "txn_max_5m": 0.0,
            }

        amounts = [amt for _, amt in window]
        return {
            "txn_count_5m": len(amounts),
            "txn_sum_5m": sum(amounts),
            "txn_avg_5m": sum(amounts) / len(amounts),
            "txn_max_5m": max(amounts),
        }

    def compute_device_features(self, device_id: str, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Compute device-based features."""
        # In production, query from feature store
        return {
            "device_trust_score": 0.8,
            "device_seen_before": True,
            "device_risk_level": "low",
        }

    def compute_merchant_features(self, merchant_id: str) -> Dict[str, Any]:
        """Compute merchant-based features."""
        # In production, query from feature store
        return {
            "merchant_risk_score": 0.1,
            "merchant_category": "retail",
            "merchant_trust_level": "high",
        }

    # Event publishing
    async def publish_fraud_event(self, event: Any) -> bool:
        """Publish a fraud event to output topic."""
        if self._producer:
            return await self._producer.send(self.output_topic, event)
        return False

    async def publish_alert(self, alert_data: Dict[str, Any], correlation_id: Optional[str] = None):
        """Publish a fraud alert."""
        if self.use_mock:
            from upishield_core.streaming.mock_kafka import create_mock_fraud_alert_event
            event = await create_mock_fraud_alert_event(alert_data, correlation_id)
        else:
            from upishield_core.streaming.kafka_producer import create_fraud_alert_event
            event = await create_fraud_alert_event(alert_data, correlation_id)
        await self.publish_fraud_event(event)

        # Trigger callbacks
        for cb in self._on_alert_generated:
            try:
                await cb(alert_data)
            except Exception as e:
                logger.error("alert_callback_error", error=str(e))

    def on_fraud_detected(self, callback: Callable):
        """Register callback for fraud detection."""
        self._on_fraud_detected.append(callback)

    def on_alert_generated(self, callback: Callable):
        """Register callback for alert generation."""
        self._on_alert_generated.append(callback)


class StreamMessageProcessor(MessageProcessor):
    """Message processor for stream processor."""

    def __init__(self, stream_processor: StreamProcessor):
        self.stream_processor = stream_processor

    async def process(self, message: ConsumedMessage) -> ProcessingResult:
        """Process incoming message based on topic."""
        start_time = datetime.utcnow()

        try:
            if message.topic == "transactions":
                return await self._process_transaction(message)
            elif message.topic == "scoring-requests":
                return await self._process_scoring_request(message)
            else:
                logger.warning("unknown_topic", topic=message.topic)
                return ProcessingResult(success=True, message=message)

        except Exception as e:
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ProcessingResult(success=False, message=message, error=str(e), processing_time_ms=latency_ms)

    async def _process_transaction(self, message: Any) -> Any:
        """Process transaction event."""
        start_time = datetime.utcnow()
        payload = message.value

        # Extract key fields
        txn_id = payload.get("txn_id", str(uuid.uuid4())[:12])
        user_id = payload.get("user_id", "unknown")
        amount = float(payload.get("amount", 0))
        device_id = payload.get("device_id", "unknown")
        merchant_id = payload.get("merchant_id", "unknown")
        timestamp_str = payload.get("timestamp", datetime.utcnow().isoformat())

        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except Exception:
            timestamp = datetime.utcnow()

        # Compute features
        velocity = self.stream_processor.compute_velocity_features(user_id, amount, timestamp)
        device_features = self.stream_processor.compute_device_features(device_id, payload)
        merchant_features = self.stream_processor.compute_merchant_features(merchant_id)

        # Simple fraud scoring (in production, call ML model)
        risk_score = self._calculate_risk_score(payload, velocity, device_features, merchant_features)

        # Record metrics
        self.stream_processor.aggregator.add("transactions", 1, timestamp)
        self.stream_processor.aggregator.add("risk_score", risk_score, timestamp)

        if risk_score > 0.7:
            self.stream_processor.aggregator.add("fraud_detected", 1, timestamp)

        # Prepare scoring result
        scoring_result = {
            "txn_id": txn_id,
            "user_id": user_id,
            "risk_score": risk_score,
            "risk_level": "high" if risk_score > 0.7 else ("medium" if risk_score > 0.3 else "low"),
            "decision": "decline" if risk_score > 0.8 else ("review" if risk_score > 0.4 else "approve"),
            "features": {
                "velocity": velocity,
                "device": device_features,
                "merchant": merchant_features,
            },
            "timestamp": timestamp.isoformat(),
        }

        # Publish scoring result
        if self.stream_processor.use_mock:
            from upishield_core.streaming.mock_kafka import create_mock_scoring_result_event
            event = await create_mock_scoring_result_event(scoring_result, correlation_id=txn_id, user_id=user_id)
        else:
            from upishield_core.streaming.kafka_producer import create_scoring_result_event
            event = await create_scoring_result_event(scoring_result, correlation_id=txn_id, user_id=user_id)
        await self.stream_processor.publish_fraud_event(event)

        # Trigger fraud detection callbacks if high risk
        if risk_score > 0.7:
            for cb in self.stream_processor._on_fraud_detected:
                try:
                    await cb(scoring_result)
                except Exception as e:
                    logger.error("fraud_callback_error", error=str(e))

        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.stream_processor.metrics.record_event(latency_ms)

        # Return appropriate result type
        if self.stream_processor.use_mock:
            return MockProcessingResult(success=True, message=message, processing_time_ms=latency_ms)
        else:
            return ProcessingResult(success=True, message=message, processing_time_ms=latency_ms)

    async def _process_scoring_request(self, message: Any) -> Any:
        """Process scoring request event."""
        start_time = datetime.utcnow()
        payload = message.value

        # In production, this would trigger async scoring
        logger.debug("scoring_request_received", payload=payload)

        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.stream_processor.metrics.record_event(latency_ms)

        if self.stream_processor.use_mock:
            return MockProcessingResult(success=True, message=message, processing_time_ms=latency_ms)
        else:
            return ProcessingResult(success=True, message=message, processing_time_ms=latency_ms)

    def _calculate_risk_score(
        self,
        transaction: Dict[str, Any],
        velocity: Dict[str, float],
        device_features: Dict[str, Any],
        merchant_features: Dict[str, Any],
    ) -> float:
        """Calculate fraud risk score (simplified)."""
        score = 0.0

        # Velocity factors
        if velocity["txn_count_5m"] > 10:
            score += 0.3
        elif velocity["txn_count_5m"] > 5:
            score += 0.15

        if velocity["txn_sum_5m"] > 100000:  # 1 lakh INR
            score += 0.2

        # Amount anomaly
        amount = float(transaction.get("amount", 0))
        if amount > 50000:
            score += 0.15
        if amount > 100000:
            score += 0.1

        # Device factors
        if not device_features.get("device_seen_before", True):
            score += 0.2
        if device_features.get("device_risk_level") == "high":
            score += 0.25

        # Merchant factors
        if merchant_features.get("merchant_risk_score", 0) > 0.5:
            score += 0.2

        # Time-based (night transactions higher risk)
        hour = datetime.utcnow().hour
        if hour < 6 or hour > 22:
            score += 0.05

        return min(score, 1.0)


# Global stream processor
_stream_processor: Optional[StreamProcessor] = None


def get_stream_processor() -> Optional[StreamProcessor]:
    """Get global stream processor."""
    return _stream_processor


async def init_stream_processor(
    bootstrap_servers: str = "localhost:9092",
    consumer_group: str = "scamshield-processor",
    input_topics: Optional[List[str]] = None,
    output_topic: str = "fraud-events",
    dead_letter_topic: str = "fraud-events-dlq",
    max_concurrent: int = 20,
    use_mock: Optional[bool] = None,
) -> StreamProcessor:
    """Initialize and start global stream processor."""
    global _stream_processor

    # Auto-detect mock mode from environment if not specified
    if use_mock is None:
        use_mock = os.getenv("USE_MOCK_KAFKA", "true").lower() == "true"

    _stream_processor = StreamProcessor(
        bootstrap_servers=bootstrap_servers,
        consumer_group=consumer_group,
        input_topics=input_topics,
        output_topic=output_topic,
        dead_letter_topic=dead_letter_topic,
        max_concurrent=max_concurrent,
        use_mock=use_mock,
    )
    await _stream_processor.start()
    return _stream_processor


async def close_stream_processor():
    """Close global stream processor."""
    global _stream_processor
    if _stream_processor:
        await _stream_processor.stop()
        _stream_processor = None