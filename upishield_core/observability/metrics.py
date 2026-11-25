"""
Prometheus Metrics for UPIShield.

Standardized metrics for monitoring and alerting.
"""

import time
from functools import wraps
from typing import Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CollectorRegistry

# Custom registry for UPIShield metrics
REGISTRY = CollectorRegistry()

# Request Metrics

REQUEST_COUNT = Counter(
    "upishield_requests_total",
    "Total number of scoring requests",
    ["endpoint", "method", "status"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "upishield_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "method"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

REQUEST_IN_FLIGHT = Gauge(
    "upishield_requests_in_flight",
    "Number of requests currently being processed",
    ["endpoint"],
    registry=REGISTRY,
)

# Scoring Metrics

SCORE_COUNT = Counter(
    "upishield_score_total",
    "Total number of transactions scored",
    ["decision", "risk_level"],
    registry=REGISTRY,
)

SCORE_LATENCY = Histogram(
    "upishield_score_latency_seconds",
    "Scoring latency in seconds",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY,
)

ML_INFERENCE_LATENCY = Histogram(
    "upishield_ml_inference_latency_seconds",
    "ML model inference latency in seconds",
    ["model_name", "model_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    registry=REGISTRY,
)

ML_MODEL_SCORE = Histogram(
    "upishield_ml_model_score",
    "ML model raw score distribution",
    ["model_name"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)

# Feature Store Metrics

FEATURE_STORE_LATENCY = Histogram(
    "upishield_feature_store_latency_seconds",
    "Feature store retrieval latency in seconds",
    ["operation", "cache_hit"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    registry=REGISTRY,
)

FEATURE_STORE_REQUESTS = Counter(
    "upishield_feature_store_requests_total",
    "Total feature store requests",
    ["operation", "status"],
    registry=REGISTRY,
)

L1_CACHE_HITS = Counter(
    "upishield_l1_cache_hits_total",
    "L1 cache hits",
    registry=REGISTRY,
)

L1_CACHE_MISSES = Counter(
    "upishield_l1_cache_misses_total",
    "L1 cache misses",
    registry=REGISTRY,
)

REDIS_CONNECTIONS = Gauge(
    "upishield_redis_connections",
    "Active Redis connections",
    registry=REGISTRY,
)

# Decision Engine Metrics

RULE_EVALUATIONS = Counter(
    "upishield_rule_evaluations_total",
    "Total rule evaluations",
    ["rule_id", "triggered"],
    registry=REGISTRY,
)

RULE_LATENCY = Histogram(
    "upishield_rule_evaluation_latency_seconds",
    "Rule evaluation latency",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    registry=REGISTRY,
)

THREAT_INTEL_LATENCY = Histogram(
    "upishield_threat_intel_latency_seconds",
    "Threat intelligence lookup latency",
    ["source"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY,
)

THREAT_INTEL_SCORE = Histogram(
    "upishield_threat_intel_score",
    "Threat intelligence risk scores",
    ["source"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)

DECISION_LATENCY = Histogram(
    "upishield_decision_latency_seconds",
    "End-to-end decision latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

# ============================================================
# Model Metrics
# ============================================================

MODEL_VERSION = Gauge(
    "upishield_model_version",
    "Currently active model version",
    ["model_name"],
    registry=REGISTRY,
)

MODEL_LOAD_TIME = Histogram(
    "upishield_model_load_time_seconds",
    "Model loading time",
    ["model_name"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

MODEL_ERRORS = Counter(
    "upishield_model_errors_total",
    "Model inference errors",
    ["model_name", "error_type"],
    registry=REGISTRY,
)

# ============================================================
# Business Metrics
# ============================================================

TRANSACTION_AMOUNT = Histogram(
    "upishield_transaction_amount",
    "Transaction amount distribution (INR)",
    buckets=[10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    registry=REGISTRY,
)

TRANSACTION_VELOCITY = Histogram(
    "upishield_transaction_velocity_5m",
    "Transactions per VPA in 5 minutes",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    registry=REGISTRY,
)

FRAUD_RATE = Gauge(
    "upishield_fraud_rate",
    "Current fraud rate (5-minute window)",
    registry=REGISTRY,
)

REVIEW_RATE = Gauge(
    "upishield_review_rate",
    "Current review rate (5-minute window)",
    registry=REGISTRY,
)

# System Metrics

MEMORY_USAGE = Gauge(
    "upishield_memory_usage_bytes",
    "Process memory usage in bytes",
    registry=REGISTRY,
)

CPU_USAGE = Gauge(
    "upishield_cpu_usage_percent",
    "Process CPU usage percentage",
    registry=REGISTRY,
)

THREAD_COUNT = Gauge(
    "upishield_thread_count",
    "Number of active threads",
    registry=REGISTRY,
)


# Decorators for Easy Instrumentation


def track_latency(histogram: Histogram, labels: dict = None):
    """Decorator to track function latency."""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                (
                    histogram.labels(**labels).observe(latency)
                    if labels
                    else histogram.observe(latency)
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                (
                    histogram.labels(**labels).observe(latency)
                    if labels
                    else histogram.observe(latency)
                )

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def count_requests(counter: Counter, labels: dict = None):
    """Decorator to count function calls."""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                counter.labels(**labels).inc() if labels else counter.inc()
                return result
            except Exception:
                error_labels = {**(labels or {}), "status": "error"}
                counter.labels(**error_labels).inc()
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                counter.labels(**labels).inc() if labels else counter.inc()
                return result
            except Exception:
                error_labels = {**(labels or {}), "status": "error"}
                counter.labels(**error_labels).inc()
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Metrics Export


def get_metrics() -> bytes:
    """Get Prometheus metrics in text format."""
    return generate_latest(REGISTRY)


def get_content_type() -> str:
    """Get Prometheus content type."""
    return CONTENT_TYPE_LATEST


class MetricsCollector:
    """Collect and update system metrics periodically."""

    def __init__(self, interval: float = 10.0):
        self.interval = interval
        self._running = False

    async def start(self):
        """Start collecting metrics."""
        self._running = True
        import asyncio

        import psutil

        process = psutil.Process()

        while self._running:
            try:
                MEMORY_USAGE.set(process.memory_info().rss)
                CPU_USAGE.set(process.cpu_percent())
                THREAD_COUNT.set(process.num_threads())
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    def stop(self):
        """Stop collecting metrics."""
        self._running = False
