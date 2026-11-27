"""Observability module for UPIShield."""

from upishield_core.observability.metrics import (
    DECISION_LATENCY,
    FEATURE_STORE_LATENCY,
    ML_INFERENCE_LATENCY,
    REGISTRY,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SCORE_COUNT,
    SCORE_LATENCY,
    MetricsCollector,
    count_requests,
    get_content_type,
    get_metrics,
    track_latency,
)
from upishield_core.observability.tracing import (
    SpanAttributes,
    TracingHelper,
    ensure_tracing,
    extract_correlation_id,
    get_tracer,
    init_tracing,
    inject_correlation_id,
    set_correlation_id,
    shutdown_tracing,
    span,
    traced,
)

# Re-export SpanKind for convenience
try:
    from opentelemetry.trace import SpanKind
except ImportError:
    SpanKind = None

__all__ = [
    # Metrics
    "REGISTRY",
    "get_metrics",
    "get_content_type",
    "MetricsCollector",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "SCORE_COUNT",
    "SCORE_LATENCY",
    "ML_INFERENCE_LATENCY",
    "FEATURE_STORE_LATENCY",
    "DECISION_LATENCY",
    "track_latency",
    "count_requests",
    # Tracing
    "init_tracing",
    "get_tracer",
    "span",
    "traced",
    "TracingHelper",
    "inject_correlation_id",
    "extract_correlation_id",
    "set_correlation_id",
    "SpanAttributes",
    "shutdown_tracing",
    "ensure_tracing",
    "SpanKind",
]
