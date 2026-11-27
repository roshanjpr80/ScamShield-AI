"""
ScamShield AI - Streaming Module.

Provides real-time event streaming with Kafka integration:
- Kafka Producer for reliable event publishing
- Kafka Consumer with dead letter queue
- Stream Processor for real-time feature computation and fraud detection
- Mock Kafka for development/testing without broker
"""

from .kafka_producer import (
    KafkaProducer,
    KafkaEvent,
    EventType,
    EventPriority,
    ProducerConfig,
    create_transaction_event,
    create_fraud_alert_event,
    create_scoring_result_event,
    get_kafka_producer,
    init_kafka_producer,
    close_kafka_producer,
)

from .kafka_consumer import (
    KafkaConsumer,
    ConsumerConfig,
    ConsumedMessage,
    MessageProcessor,
    ProcessingResult,
    ProcessingStatus,
    LoggingProcessor,
    CallbackProcessor,
    create_consumer,
    start_consumer,
    stop_consumer,
    stop_all_consumers,
    get_consumer,
    get_all_consumer_metrics,
)

from .stream_processor import (
    StreamProcessor,
    StreamProcessorState,
    StreamMetrics,
    WindowedAggregator,
    StreamMessageProcessor,
    get_stream_processor,
    init_stream_processor,
    close_stream_processor,
)

from .mock_kafka import (
    MockKafkaProducer,
    MockKafkaConsumer,
    MockKafkaEvent,
    MockKafkaCluster,
    MockProducerConfig,
    MockConsumerConfig,
    MockConsumedMessage,
    MockProcessingResult,
    MockMessageProcessor,
    EventType,
    EventPriority,
    ProcessingStatus,
    create_mock_transaction_event,
    create_mock_fraud_alert_event,
    create_mock_scoring_result_event,
    get_mock_kafka_producer,
    init_mock_kafka_producer,
    close_mock_kafka_producer,
    create_mock_consumer,
    start_mock_consumer,
    stop_mock_consumer,
    get_mock_consumer_metrics,
    get_mock_cluster,
)

__all__ = [
    # Kafka Producer (Real)
    "KafkaProducer",
    "KafkaEvent",
    "EventType",
    "EventPriority",
    "ProducerConfig",
    "create_transaction_event",
    "create_fraud_alert_event",
    "create_scoring_result_event",
    "get_kafka_producer",
    "init_kafka_producer",
    "close_kafka_producer",
    # Kafka Consumer (Real)
    "KafkaConsumer",
    "ConsumerConfig",
    "ConsumedMessage",
    "MessageProcessor",
    "ProcessingResult",
    "ProcessingStatus",
    "LoggingProcessor",
    "CallbackProcessor",
    "create_consumer",
    "start_consumer",
    "stop_consumer",
    "stop_all_consumers",
    "get_consumer",
    "get_all_consumer_metrics",
    # Stream Processor
    "StreamProcessor",
    "StreamProcessorState",
    "StreamMetrics",
    "WindowedAggregator",
    "StreamMessageProcessor",
    "get_stream_processor",
    "init_stream_processor",
    "close_stream_processor",
    # Mock Kafka (for development/testing)
    "MockKafkaProducer",
    "MockKafkaConsumer",
    "MockKafkaEvent",
    "MockKafkaCluster",
    "MockProducerConfig",
    "MockConsumerConfig",
    "MockConsumedMessage",
    "MockProcessingResult",
    "MockMessageProcessor",
    "create_mock_transaction_event",
    "create_mock_fraud_alert_event",
    "create_mock_scoring_result_event",
    "get_mock_kafka_producer",
    "init_mock_kafka_producer",
    "close_mock_kafka_producer",
    "create_mock_consumer",
    "start_mock_consumer",
    "stop_mock_consumer",
    "get_mock_consumer_metrics",
    "get_mock_cluster",
]