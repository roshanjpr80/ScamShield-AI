"""
OpenTelemetry Tracing for UPIShield.

Distributed tracing for request flows across services.
"""

import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
except ImportError:
    # Newer OpenTelemetry versions may have different import paths
    try:
        from opentelemetry.exporter.jaeger import JaegerExporter
    except ImportError:
        JaegerExporter = None
try:
    from opentelemetry.exporter.zipkin.proto.http import ZipkinExporter
except ImportError:
    ZipkinExporter = None
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
except ImportError:
    FastAPIInstrumentor = None

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
except ImportError:
    HTTPXClientInstrumentor = None

try:
    from opentelemetry.instrumentation.redis import RedisInstrumentor
except ImportError:
    RedisInstrumentor = None

try:
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
except ImportError:
    SQLAlchemyInstrumentor = None
from opentelemetry.propagate import set_global_textmap

try:
    from opentelemetry.propagators.b3 import B3MultiFormat
except ImportError:
    B3MultiFormat = None

try:
    from opentelemetry.propagators.composite import CompositeHTTPPropagator
except ImportError:
    CompositeHTTPPropagator = None

try:
    from opentelemetry.propagators.jaeger import JaegerFormat
except ImportError:
    JaegerFormat = None

from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Global tracer
_tracer: Optional[trace.Tracer] = None
_provider: Optional[TracerProvider] = None


def init_tracing(
    service_name: str = "upishield",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    jaeger_endpoint: Optional[str] = None,
    zipkin_endpoint: Optional[str] = None,
    console_export: bool = False,
    sample_rate: float = 1.0,
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for traces
        service_version: Service version
        otlp_endpoint: OTLP gRPC endpoint (e.g., "http://jaeger:4317")
        jaeger_endpoint: Jaeger Thrift endpoint (e.g., "http://jaeger:14268/api/traces")
        zipkin_endpoint: Zipkin endpoint (e.g., "http://zipkin:9411/api/v2/spans")
        console_export: Export to console (for development)
        sample_rate: Sampling rate (0.0-1.0)

    Returns:
        Configured tracer instance
    """
    global _tracer, _provider

    # Create resource
    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        }
    )

    # Create tracer provider
    _provider = TracerProvider(resource=resource)

    # Configure sampler
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    _provider.sampler = TraceIdRatioBased(sample_rate)

    # Add exporters
    exporters = []

    if otlp_endpoint:
        exporters.append(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))

    if jaeger_endpoint and JaegerExporter:
        exporters.append(
            JaegerExporter(
                agent_host_name=jaeger_endpoint.split("://")[-1].split(":")[0],
                agent_port=int(jaeger_endpoint.split(":")[-1]) if ":" in jaeger_endpoint else 6831,
            )
        )

    if zipkin_endpoint and ZipkinExporter:
        exporters.append(ZipkinExporter(endpoint=zipkin_endpoint))

    if console_export or not exporters:
        exporters.append(ConsoleSpanExporter())

    for exporter in exporters:
        _provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set global tracer provider
    trace.set_tracer_provider(_provider)

    # Set global propagator (support multiple formats)
    propagators = [TraceContextTextMapPropagator()]
    if B3MultiFormat:
        propagators.append(B3MultiFormat())
    if JaegerFormat:
        propagators.append(JaegerFormat())
    if CompositeHTTPPropagator:
        set_global_textmap(CompositeHTTPPropagator(propagators))
    else:
        # Fallback to single propagator
        set_global_textmap(propagators[0])

    # Get tracer
    _tracer = trace.get_tracer(service_name, service_version)

    # Auto-instrument libraries
    _auto_instrument()

    return _tracer


def _auto_instrument():
    """Auto-instrument common libraries."""
    if FastAPIInstrumentor:
        try:
            FastAPIInstrumentor.instrument()
        except Exception:
            pass

    if HTTPXClientInstrumentor:
        try:
            HTTPXClientInstrumentor.instrument()
        except Exception:
            pass

    if RedisInstrumentor:
        try:
            RedisInstrumentor.instrument()
        except Exception:
            pass

    if SQLAlchemyInstrumentor:
        try:
            SQLAlchemyInstrumentor.instrument()
        except Exception:
            pass


def get_tracer() -> trace.Tracer:
    """Get the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = init_tracing()
    return _tracer


# ============================================================
# Span Helpers
# ============================================================


@contextmanager
def span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Dict[str, Any] = None,
    record_exception: bool = True,
):
    """
    Context manager for creating spans.

    Usage:
        with span("my_operation", attributes={"key": "value"}) as s:
            do_work()
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as current_span:
        try:
            yield current_span
        except Exception as e:
            if record_exception:
                current_span.record_exception(e)
                current_span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def traced(name: str = None, kind: SpanKind = SpanKind.INTERNAL, attributes: Dict = None):
    """
    Decorator for tracing functions.

    Usage:
        @traced("my_function", attributes={"custom_attr": "value"})
        async def my_function():
            ...
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with tracer.start_as_current_span(
                span_name, kind=kind, attributes=attributes or {}
            ) as current_span:
                try:
                    result = await func(*args, **kwargs)
                    current_span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    current_span.record_exception(e)
                    current_span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with tracer.start_as_current_span(
                span_name, kind=kind, attributes=attributes or {}
            ) as current_span:
                try:
                    result = func(*args, **kwargs)
                    current_span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    current_span.record_exception(e)
                    current_span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================
# Domain-Specific Span Helpers
# ============================================================


class TracingHelper:
    """Helper methods for common tracing patterns in UPIShield."""

    @staticmethod
    @contextmanager
    def score_transaction(txn_id: str, payer_vpa: str, payee_vpa: str, amount: float):
        """Trace a transaction scoring operation."""
        with span(
            "score_transaction",
            kind=SpanKind.SERVER,
            attributes={
                "txn_id": txn_id,
                "payer_vpa": payer_vpa,
                "payee_vpa": payee_vpa,
                "amount": amount,
                "service": "upishield",
            },
        ) as current_span:
            yield current_span

    @staticmethod
    @contextmanager
    def feature_store_get(keys_count: int, cache_hit: bool):
        """Trace feature store retrieval."""
        with span(
            "feature_store.get",
            kind=SpanKind.CLIENT,
            attributes={
                "keys_count": keys_count,
                "cache_hit": cache_hit,
                "service": "feature_store",
            },
        ) as current_span:
            yield current_span

    @staticmethod
    @contextmanager
    def ml_inference(model_name: str, model_type: str, input_shape: tuple):
        """Trace ML model inference."""
        with span(
            f"ml_inference.{model_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                "model_name": model_name,
                "model_type": model_type,
                "input_shape": str(input_shape),
                "service": "ml_ensemble",
            },
        ) as current_span:
            yield current_span

    @staticmethod
    @contextmanager
    def rule_evaluation(rules_count: int):
        """Trace rule engine evaluation."""
        with span(
            "rule_evaluation",
            kind=SpanKind.INTERNAL,
            attributes={
                "rules_count": rules_count,
                "service": "decision_engine",
            },
        ) as current_span:
            yield current_span

    @staticmethod
    @contextmanager
    def threat_intel_lookup(source: str, target: str):
        """Trace threat intelligence lookup."""
        with span(
            f"threat_intel.{source}",
            kind=SpanKind.CLIENT,
            attributes={
                "source": source,
                "target": target,
                "service": "threat_intel",
            },
        ) as current_span:
            yield current_span

    @staticmethod
    @contextmanager
    def decision_combination(components: List[str]):
        """Trace decision combination logic."""
        with span(
            "decision_combination",
            kind=SpanKind.INTERNAL,
            attributes={
                "components": ",".join(components),
                "service": "decision_engine",
            },
        ) as current_span:
            yield current_span

    @staticmethod
    @contextmanager
    def http_request(method: str, url: str, status_code: int = None):
        """Trace HTTP request."""
        with span(
            f"http.{method}",
            kind=SpanKind.CLIENT,
            attributes={
                "http.method": method,
                "http.url": url,
                "http.status_code": status_code,
            },
        ) as current_span:
            yield current_span

    @staticmethod
    @contextmanager
    def db_query(query: str, table: str = None):
        """Trace database query."""
        with span(
            "db.query",
            kind=SpanKind.CLIENT,
            attributes={
                "db.statement": query[:500],
                "db.table": table or "unknown",
            },
        ) as current_span:
            yield current_span


# ============================================================
# Correlation ID Propagation
# ============================================================

CORRELATION_ID_HEADER = "x-correlation-id"
TRACE_ID_HEADER = "x-trace-id"


def inject_correlation_id(headers: Dict[str, str]) -> Dict[str, str]:
    """Inject correlation ID into headers for downstream calls."""
    span = trace.get_current_span()
    if span and span.get_span_context().trace_id:
        trace_id = format(span.get_span_context().trace_id, "032x")
        headers[TRACE_ID_HEADER] = trace_id
        headers[CORRELATION_ID_HEADER] = trace_id[:16]
    return headers


def extract_correlation_id(headers: Dict[str, str]) -> Optional[str]:
    """Extract correlation ID from incoming headers."""
    return headers.get(CORRELATION_ID_HEADER) or headers.get(TRACE_ID_HEADER)


def set_correlation_id(correlation_id: str):
    """Set correlation ID in current span."""
    span = trace.get_current_span()
    if span:
        span.set_attribute("correlation_id", correlation_id)


# ============================================================
# Custom Span Attributes for UPIShield
# ============================================================


class SpanAttributes:
    """Standardized span attribute keys for UPIShield."""

    # Transaction
    TXN_ID = "upishield.txn_id"
    PAYER_VPA = "upishield.payer_vpa"
    PAYEE_VPA = "upishield.payee_vpa"
    AMOUNT = "upishield.amount"
    TXN_TYPE = "upishield.txn_type"
    DECISION = "upishield.decision"
    RISK_LEVEL = "upishield.risk_level"
    RISK_SCORE = "upishield.risk_score"

    # Features
    FEATURE_COUNT = "upishield.feature_count"
    CACHE_HIT = "upishield.cache_hit"

    # ML
    MODEL_NAME = "upishield.model_name"
    MODEL_VERSION = "upishield.model_version"
    MODEL_SCORE = "upishield.model_score"
    MODEL_CONFIDENCE = "upishield.model_confidence"
    ENSEMBLE_SIZE = "upishield.ensemble_size"

    # Rules
    RULES_TRIGGERED = "upishield.rules_triggered"
    RULE_SCORE = "upishield.rule_score"

    # Threat Intel
    THREAT_SOURCE = "upishield.threat_source"
    THREAT_SCORE = "upishield.threat_score"
    THREAT_LABELS = "upishield.threat_labels"

    # Performance
    LATENCY_MS = "upishield.latency_ms"
    FEATURE_LATENCY_MS = "upishield.feature_latency_ms"
    ML_LATENCY_MS = "upishield.ml_latency_ms"
    RULE_LATENCY_MS = "upishield.rule_latency_ms"
    THREAT_LATENCY_MS = "upishield.threat_latency_ms"

    # Errors
    ERROR_TYPE = "upishield.error_type"
    ERROR_MESSAGE = "upishield.error_message"


def set_span_attributes(span: trace.Span, attributes: Dict[str, Any]):
    """Set multiple span attributes at once."""
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


# ============================================================
# Lifecycle
# ============================================================


async def shutdown_tracing():
    """Shutdown tracing provider gracefully."""
    global _provider
    if _provider:
        _provider.shutdown()
        _provider = None


# Initialize on import (lazy - will init on first use)
def ensure_tracing():
    """Ensure tracing is initialized."""
    if _tracer is None:
        init_tracing()
