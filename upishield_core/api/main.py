"""
UPIShield FastAPI Application.

Production-grade API server for real-time UPI fraud detection.
Sub-50ms p99 latency with ONNX Runtime INT8 quantization.
"""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from upishield_core import (
    BatchScoreRequest,
    BatchScoreResponse,
    DecisionEngine,
    FeatureStore,
    FeatureStoreConfig,
    FraudDecision,
    HealthResponse,
    JobResponse,
    JobStatus,
    ModelInfoResponse,
    ModelRegistry,
    ONNXEnsemble,
    ScoreRequest,
    ScoreResponse,
    UPITransaction,
)
from upishield_core.api.rag import (
    FraudKnowledgeBase,
    RAGQuery,
    RAGResponse,
    get_rag,
)
from upishield_core.api.ai.agentic_endpoints import router as agentic_router, close_workflow
from upishield_core.api.ai.rag_endpoints import router as rag_router
from upishield_core.api.websocket.endpoints import router as websocket_router, init_websocket, close_websocket
from upishield_core.streaming import (
    init_stream_processor,
    close_stream_processor,
    init_kafka_producer,
    close_kafka_producer,
    get_stream_processor,
    init_mock_kafka_producer,
    close_mock_kafka_producer,
    get_mock_kafka_producer,
)
from upishield_core.observability import (
    init_tracing,
    shutdown_tracing,
    ensure_tracing,
    get_tracer,
    traced,
    span,
    inject_correlation_id,
    extract_correlation_id,
    set_correlation_id,
    SpanAttributes,
    SpanKind,
)
from opentelemetry import trace

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "upishield_requests_total", "Total number of requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "upishield_request_duration_seconds", "Request latency in seconds", ["method", "endpoint"]
)
MODEL_INFERENCE_LATENCY = Histogram(
    "upishield_model_inference_duration_seconds", "Model inference latency in seconds", ["model"]
)
DECISIONS_COUNTER = Counter("upishield_decisions_total", "Total decisions by type", ["decision"])
ACTIVE_JOBS = Gauge("upishield_active_jobs", "Number of active async jobs")
MODELS_LOADED = Gauge("upishield_models_loaded", "Number of models loaded")
ERROR_COUNTER = Counter("upishield_errors_total", "Total errors", ["endpoint", "error_type"])

# Global state
app_state = {
    "ensemble": None,
    "registry": None,
    "decision_engine": None,
    "feature_store": None,
    "rag": None,
    "start_time": time.time(),
    "jobs": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global app_state

    logger.info("starting_up", version="0.1.0")

    # Initialize OpenTelemetry tracing
    try:
        otlp_endpoint = os.getenv("OTLP_ENDPOINT")
        jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
        zipkin_endpoint = os.getenv("ZIPKIN_ENDPOINT")
        console_export = os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
        sample_rate = float(os.getenv("OTEL_SAMPLE_RATE", "1.0"))

        init_tracing(
            service_name="upishield",
            service_version="0.1.0",
            otlp_endpoint=otlp_endpoint,
            jaeger_endpoint=jaeger_endpoint,
            zipkin_endpoint=zipkin_endpoint,
            console_export=console_export,
            sample_rate=sample_rate,
        )
        logger.info("tracing_initialized", otlp=bool(otlp_endpoint), jaeger=bool(jaeger_endpoint), zipkin=bool(zipkin_endpoint))
    except Exception as e:
        logger.warning("tracing_unavailable", error=str(e))

    # Initialize model registry
    app_state["registry"] = ModelRegistry("./models")

    # Load ensemble models
    try:
        app_state["ensemble"] = ONNXEnsemble("./models")
        app_state["ensemble"].load()
        MODELS_LOADED.set(len(app_state["ensemble"].sessions))
        logger.info("models_loaded", count=len(app_state["ensemble"].sessions))
    except FileNotFoundError as e:
        logger.warning("models_not_found", error=str(e))
        MODELS_LOADED.set(0)

    # Initialize decision engine
    if app_state["ensemble"]:
        app_state["decision_engine"] = DecisionEngine(
            ensemble=app_state["ensemble"],
            registry=app_state["registry"],
        )

    # Initialize feature store (optional)
    try:
        config = FeatureStoreConfig(redis_url="redis://localhost:6379")
        app_state["feature_store"] = FeatureStore(config)
        await app_state["feature_store"].connect()
        logger.info("feature_store_connected")
    except Exception as e:
        logger.warning("feature_store_unavailable", error=str(e))

    # Initialize RAG knowledge base (optional)
    try:
        from upishield_core.ai import get_knowledge_base
        app_state["rag"] = get_knowledge_base()
        success = app_state["rag"].initialize(seed_data=True)
        if success:
            logger.info("rag_initialized", chunks=app_state["rag"].vector_store.count())
    except Exception as e:
        logger.warning("rag_unavailable", error=str(e))

    # Initialize WebSocket manager
    await init_websocket()

    # Initialize Kafka producer and stream processor (optional)
    # Use mock Kafka for development without broker
    import os
    use_mock_kafka = os.getenv("USE_MOCK_KAFKA", "true").lower() == "true"

    try:
        if use_mock_kafka:
            await init_mock_kafka_producer()
            logger.info("mock_kafka_producer_initialized")
        else:
            await init_kafka_producer()
            logger.info("kafka_producer_initialized")

        await init_stream_processor(use_mock=use_mock_kafka)
        logger.info("stream_processor_initialized")
    except Exception as e:
        logger.warning("streaming_unavailable", error=str(e))

    yield

    # Shutdown
    logger.info("shutting_down")
    if app_state["feature_store"]:
        await app_state["feature_store"].close()
    await close_workflow()
    await close_websocket()
    await close_stream_processor()
    import os
    use_mock_kafka = os.getenv("USE_MOCK_KAFKA", "true").lower() == "true"
    if use_mock_kafka:
        await close_mock_kafka_producer()
    else:
        await close_kafka_producer()
    # Shutdown tracing
    await shutdown_tracing()


app = FastAPI(
    title="UPIShield API",
    description="Real-time UPI Fraud Detection - Sub-50ms p99 latency",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agentic_router)
app.include_router(rag_router)
app.include_router(websocket_router)


# Request timing middleware with correlation ID propagation
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.perf_counter()

    # Extract correlation ID from incoming headers
    correlation_id = extract_correlation_id(dict(request.headers))
    if correlation_id:
        set_correlation_id(correlation_id)

    # Add correlation ID to response headers
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)

    REQUEST_COUNT.labels(
        method=request.method, endpoint=request.url.path, status=response.status_code
    ).inc()

    # Inject correlation ID into response headers
    response.headers["X-Process-Time"] = f"{duration:.4f}"
    inject_correlation_id(dict(response.headers))
    # Actually set the headers on the response
    span = trace.get_current_span()
    if span and span.get_span_context().trace_id:
        trace_id = format(span.get_span_context().trace_id, "032x")
        response.headers["x-trace-id"] = trace_id
        response.headers["x-correlation-id"] = trace_id[:16]

    return response


# Health endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check - returns 200 if service is running."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=time.time() - app_state["start_time"],
        models_loaded=len(app_state["ensemble"].sessions) if app_state["ensemble"] else 0,
        redis_connected=app_state["feature_store"] is not None,
        database_connected=False,  # Would check PostgreSQL
    )


@app.get("/ready", response_model=HealthResponse, tags=["Health"])
async def readiness_check():
    """Readiness check - includes model loading status."""
    # Consider both ONNX and sklearn fallback as "loaded"
    models_ready = (
        app_state["ensemble"] and
        (app_state["ensemble"]._loaded or app_state["ensemble"].sklearn_models is not None)
    )
    if not models_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return HealthResponse(
        status="ready",
        version="0.1.0",
        uptime_seconds=time.time() - app_state["start_time"],
        models_loaded=len(app_state["ensemble"].sessions) if app_state["ensemble"] else 0,
        redis_connected=app_state["feature_store"] is not None,
        database_connected=False,
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


# Scoring endpoints
@app.post("/api/v1/score", response_model=ScoreResponse, tags=["Scoring"])
@traced("score_transaction", kind=SpanKind.SERVER, attributes={"service": "scoring", "endpoint": "/api/v1/score"})
async def score_transaction(
    request: ScoreRequest,
    background_tasks: BackgroundTasks,
):
    """
    Score a single UPI transaction for fraud.

    Returns decision (APPROVE/REVIEW/DECLINE), risk level, score, and explanation.
    Target latency: <50ms p99
    """
    if not app_state["decision_engine"]:
        raise HTTPException(status_code=503, detail="Decision engine not initialized")

    start_time = time.perf_counter()

    try:
        # Run decision engine
        context = await app_state["decision_engine"].score(request.transaction)

        processing_time = (time.perf_counter() - start_time) * 1000

        # Build response
        response = ScoreResponse(
            txn_id=request.transaction.txn_id,
            decision=context.decision,
            risk_level=context.risk_level,
            risk_score=context.final_score,
            confidence=context.ml_confidence,
            model_version="ensemble-latest",
            processing_time_ms=processing_time,
            explanation=context.explanation if request.include_explanation else None,
            features=context.features.to_dict() if request.include_features else None,
            triggered_rules=[
                {"rule_id": r.rule_id, "name": r.rule_name, "severity": r.severity}
                for r in context.rule_results
            ],
            threat_intel=[
                {"source": t.source, "risk_score": t.risk_score, "labels": t.labels}
                for t in context.threat_intel_results
                if t.risk_score > 0
            ],
        )

        # Track metrics
        DECISIONS_COUNTER.labels(decision=context.decision.value).inc()
        MODEL_INFERENCE_LATENCY.labels(model="ensemble").observe(processing_time / 1000)

        # Background logging
        background_tasks.add_task(log_decision, request.transaction.txn_id, context)

        return response

    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/api/v1/score", error_type=type(e).__name__).inc()
        logger.error("scoring_failed", txn_id=request.transaction.txn_id, error=str(e))
        raise HTTPException(status_code=500, detail="Scoring failed")


@app.post("/api/v1/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
@traced("score_batch", kind=SpanKind.SERVER, attributes={"service": "scoring", "endpoint": "/api/v1/score/batch"})
async def score_batch(request: BatchScoreRequest, background_tasks: BackgroundTasks):
    """
    Score multiple transactions in a single request.

    More efficient than individual calls for batch processing.
    """
    if not app_state["decision_engine"]:
        raise HTTPException(status_code=503, detail="Decision engine not initialized")

    start_time = time.perf_counter()
    batch_id = str(uuid.uuid4())[:8]

    try:
        contexts = await app_state["decision_engine"].score_batch(request.transactions)

        total_time = (time.perf_counter() - start_time) * 1000

        results = []
        for ctx in contexts:
            response = ScoreResponse(
                txn_id=ctx.transaction.txn_id,
                decision=ctx.decision,
                risk_level=ctx.risk_level,
                risk_score=ctx.final_score,
                confidence=ctx.ml_confidence,
                model_version="ensemble-latest",
                processing_time_ms=ctx.latency_ms,
                explanation=ctx.explanation if request.include_explanation else None,
                features=ctx.features.to_dict() if request.include_features else None,
                triggered_rules=[
                    {"rule_id": r.rule_id, "name": r.rule_name, "severity": r.severity}
                    for r in ctx.rule_results
                ],
                threat_intel=[
                    {"source": t.source, "risk_score": t.risk_score, "labels": t.labels}
                    for t in ctx.threat_intel_results
                    if t.risk_score > 0
                ],
            )
            results.append(response)
            DECISIONS_COUNTER.labels(decision=ctx.decision.value).inc()

        return BatchScoreResponse(
            results=results,
            total_processed=len(results),
            total_time_ms=total_time,
            batch_id=batch_id,
        )

    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/api/v1/score/batch", error_type=type(e).__name__).inc()
        logger.error("batch_scoring_failed", batch_id=batch_id, error=str(e))
        raise HTTPException(status_code=500, detail="Batch scoring failed")


# Async job endpoints
@app.post("/api/v1/score/async", tags=["Scoring"])
async def score_async(request: ScoreRequest):
    """Submit transaction for async scoring. Returns job ID."""
    job_id = str(uuid.uuid4())

    app_state["jobs"][job_id] = {
        "status": JobStatus.PENDING,
        "transaction": request.transaction,
        "include_explanation": request.include_explanation,
        "include_features": request.include_features,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }
    ACTIVE_JOBS.inc()

    # Process in background
    asyncio.create_task(process_async_job(job_id))

    return {"job_id": job_id, "status": JobStatus.PENDING}


async def process_async_job(job_id: str):
    """Background task to process async scoring job."""
    job = app_state["jobs"].get(job_id)
    if not job:
        return

    try:
        job["status"] = JobStatus.PROCESSING

        if app_state["decision_engine"]:
            context = await app_state["decision_engine"].score(job["transaction"])

            response = ScoreResponse(
                txn_id=job["transaction"].txn_id,
                decision=context.decision,
                risk_level=context.risk_level,
                risk_score=context.final_score,
                confidence=context.ml_confidence,
                model_version="ensemble-latest",
                processing_time_ms=context.latency_ms,
                explanation=context.explanation if job["include_explanation"] else None,
                features=context.features.to_dict() if job["include_features"] else None,
                triggered_rules=[
                    {"rule_id": r.rule_id, "name": r.rule_name, "severity": r.severity}
                    for r in context.rule_results
                ],
                threat_intel=[
                    {"source": t.source, "risk_score": t.risk_score, "labels": t.labels}
                    for t in context.threat_intel_results
                    if t.risk_score > 0
                ],
            )
            job["result"] = response
            job["status"] = JobStatus.COMPLETED
            DECISIONS_COUNTER.labels(decision=context.decision.value).inc()
        else:
            job["status"] = JobStatus.FAILED
            job["error"] = "Decision engine not initialized"
    except Exception as e:
        job["status"] = JobStatus.FAILED
        job["error"] = str(e)
        logger.error("async_job_failed", job_id=job_id, error=str(e))
    finally:
        job["completed_at"] = datetime.utcnow()
        ACTIVE_JOBS.dec()


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, tags=["Scoring"])
async def get_job_result(job_id: str):
    """Get result of async scoring job."""
    job = app_state["jobs"].get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job_id,
        status=job["status"],
        result=job["result"],
        error=job["error"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
    )


# Model management endpoints
@app.get("/api/v1/models/info", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info():
    """Get current model information."""
    if not app_state["registry"]:
        raise HTTPException(status_code=503, detail="Model registry not initialized")

    models = {}
    for name, versions in app_state["registry"]._index.items():
        latest = versions[0] if versions else None
        if latest:
            models[name] = {
                "latest_version": latest.version,
                "stage": latest.stage,
                "model_type": latest.model_type,
                "metrics": latest.metrics,
                "latency_ms": latest.latency_ms,
            }

    active_model = "ensemble"
    if app_state["ensemble"]:
        active_model = (
            list(app_state["ensemble"].sessions.keys())[0]
            if app_state["ensemble"].sessions
            else "none"
        )

    return ModelInfoResponse(
        models=models,
        active_model=active_model,
        model_registry_version="1.0",
        last_updated=datetime.utcnow(),
    )


@app.get("/api/v1/models", tags=["Models"])
async def list_models():
    """List all available model versions."""
    if not app_state["registry"]:
        raise HTTPException(status_code=503, detail="Model registry not initialized")

    result = {}
    for name, versions in app_state["registry"]._index.items():
        result[name] = [
            {
                "version": v.version,
                "stage": v.stage,
                "model_type": v.model_type,
                "metrics": v.metrics,
                "training_date": v.training_date,
                "file_size_mb": v.file_size_bytes / 1024 / 1024,
            }
            for v in versions
        ]
    return result


@app.post("/api/v1/models/{model_name}/versions/{version}/promote", tags=["Models"])
async def promote_model(model_name: str, version: str, stage: str = "production"):
    """Promote model version to stage."""
    if not app_state["registry"]:
        raise HTTPException(status_code=503, detail="Model registry not initialized")

    try:
        metadata = app_state["registry"].promote(model_name, version, stage)
        return {
            "name": metadata.name,
            "version": metadata.version,
            "stage": metadata.stage,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Text/SMS scam detection endpoint
class TextDetectionRequest(BaseModel):
    text: str
    include_explanation: bool = True


class TextDetectionResponse(BaseModel):
    is_scam: bool
    category: str
    confidence: float
    risk_score: float
    explanation: Optional[Dict] = None
    processing_time_ms: float


@app.post("/api/v1/detect/sms", response_model=TextDetectionResponse, tags=["Detection"])
@traced("detect_sms_scam", kind=SpanKind.SERVER, attributes={"service": "detection", "endpoint": "/api/v1/detect/sms"})
async def detect_sms_scam(request: TextDetectionRequest):
    """Detect SMS/Phishing scams from text using ONNX TextCNN."""
    start_time = time.perf_counter()

    try:
        from upishield_core.models import get_sms_detector, SmishingCategory

        detector = get_sms_detector()
        prediction = detector.predict_single(request.text)

        processing_time = (time.perf_counter() - start_time) * 1000

        return TextDetectionResponse(
            is_scam=prediction.category != SmishingCategory.LEGITIMATE.value,
            category=prediction.category,
            confidence=prediction.confidence,
            risk_score=prediction.fraud_score,
            explanation=(
                {
                    "category_probs": prediction.category_probs,
                }
                if request.include_explanation
                else None
            ),
            processing_time_ms=processing_time,
        )
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/api/v1/detect/sms", error_type=type(e).__name__).inc()
        logger.error("sms_detection_failed", error=str(e))
        raise HTTPException(status_code=500, detail="SMS detection failed")


@app.post("/api/v1/detect/job-scam", response_model=TextDetectionResponse, tags=["Detection"])
@traced("detect_job_scam", kind=SpanKind.SERVER, attributes={"service": "detection", "endpoint": "/api/v1/detect/job-scam"})
async def detect_job_scam(request: TextDetectionRequest):
    """Detect job scam from text description using ONNX TextCNN."""
    start_time = time.perf_counter()

    try:
        from upishield_core.models import get_job_detector, JobFraudCategory

        detector = get_job_detector()
        prediction = detector.predict_single(request.text)

        processing_time = (time.perf_counter() - start_time) * 1000

        return TextDetectionResponse(
            is_scam=prediction.category != JobFraudCategory.LEGITIMATE.value,
            category=prediction.category,
            confidence=prediction.confidence,
            risk_score=prediction.fraud_score,
            explanation=(
                {
                    "category_probs": prediction.category_probs,
                    "red_flags": prediction.red_flags,
                    "risk_level": prediction.risk_level,
                }
                if request.include_explanation
                else None
            ),
            processing_time_ms=processing_time,
        )
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/api/v1/detect/job-scam", error_type=type(e).__name__).inc()
        logger.error("job_scam_detection_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Job scam detection failed")


# Multi-modal detection endpoint
class MultimodalDetectionRequest(BaseModel):
    transaction: Optional[UPITransaction] = None
    sms_text: Optional[str] = None
    job_description: Optional[str] = None
    image_urls: Optional[List[str]] = None
    include_explanation: bool = True


class MultimodalDetectionResponse(BaseModel):
    transaction_decision: Optional[FraudDecision] = None
    transaction_risk_score: Optional[float] = None
    sms_detection: Optional[TextDetectionResponse] = None
    job_scam_detection: Optional[TextDetectionResponse] = None
    overall_risk_score: float
    overall_decision: FraudDecision
    processing_time_ms: float


@app.post(
    "/api/v1/detect/multimodal", response_model=MultimodalDetectionResponse, tags=["Detection"]
)
@traced("detect_multimodal", kind=SpanKind.SERVER, attributes={"service": "detection", "endpoint": "/api/v1/detect/multimodal"})
async def detect_multimodal(request: MultimodalDetectionRequest):
    """Multi-modal fraud detection combining transaction, text, and image analysis."""
    start_time = time.perf_counter()

    try:
        results = {}
        overall_scores = []

        # Transaction scoring
        if request.transaction and app_state["decision_engine"]:
            context = await app_state["decision_engine"].score(request.transaction)
            results["transaction"] = {
                "decision": context.decision,
                "risk_score": context.final_score,
            }
            overall_scores.append(context.final_score)

        # SMS detection
        if request.sms_text:
            from upishield_core.models import get_sms_detector, SmishingCategory

            detector = get_sms_detector()
            prediction = detector.predict_single(request.sms_text)
            results["sms"] = {
                "is_scam": prediction.category != SmishingCategory.LEGITIMATE.value,
                "category": prediction.category,
                "confidence": prediction.confidence,
                "risk_score": prediction.fraud_score,
                "explanation": ({"category_probs": prediction.category_probs} if request.include_explanation else None),
                "processing_time_ms": 0.0,  # Will be overridden by overall processing time
            }
            overall_scores.append(prediction.fraud_score)

        # Job scam detection
        if request.job_description:
            from upishield_core.models import get_job_detector, JobFraudCategory

            detector = get_job_detector()
            prediction = detector.predict_single(request.job_description)
            results["job_scam"] = {
                "is_scam": prediction.category != JobFraudCategory.LEGITIMATE.value,
                "category": prediction.category,
                "confidence": prediction.confidence,
                "risk_score": prediction.fraud_score,
                "explanation": ({"category_probs": prediction.category_probs, "red_flags": prediction.red_flags, "risk_level": prediction.risk_level} if request.include_explanation else None),
                "processing_time_ms": 0.0,  # Will be overridden by overall processing time
            }
            overall_scores.append(prediction.fraud_score)

        # Compute overall
        overall_risk = max(overall_scores) if overall_scores else 0.0
        overall_decision = (
            FraudDecision.DECLINE
            if overall_risk > 0.8
            else (FraudDecision.REVIEW if overall_risk > 0.45 else FraudDecision.APPROVE)
        )

        processing_time = (time.perf_counter() - start_time) * 1000

        return MultimodalDetectionResponse(
            transaction_decision=results.get("transaction", {}).get("decision"),
            transaction_risk_score=results.get("transaction", {}).get("risk_score"),
            sms_detection=TextDetectionResponse(**results["sms"]) if "sms" in results else None,
            job_scam_detection=(
                TextDetectionResponse(**results["job_scam"]) if "job_scam" in results else None
            ),
            overall_risk_score=overall_risk,
            overall_decision=overall_decision,
            processing_time_ms=processing_time,
        )
    except Exception as e:
        ERROR_COUNTER.labels(
            endpoint="/api/v1/detect/multimodal", error_type=type(e).__name__
        ).inc()
        logger.error("multimodal_detection_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Multi-modal detection failed")


# Streaming endpoints
@app.get("/api/v1/streaming/metrics", tags=["Streaming"])
async def get_streaming_metrics():
    """Get stream processor metrics."""
    import os
    use_mock_kafka = os.getenv("USE_MOCK_KAFKA", "true").lower() == "true"

    processor = get_stream_processor()
    if not processor:
        return {"error": "Stream processor not initialized", "state": "unavailable"}

    result = processor.get_metrics()

    # Add producer metrics
    if use_mock_kafka:
        mock_producer = get_mock_kafka_producer()
        if mock_producer:
            result["producer"] = mock_producer.get_metrics()
    else:
        from upishield_core.streaming import get_kafka_producer
        producer = get_kafka_producer()
        if producer:
            result["producer"] = producer.get_metrics()

    return result


@app.get("/api/v1/streaming/health", tags=["Streaming"])
async def get_streaming_health():
    """Get stream processor health."""
    import os
    use_mock_kafka = os.getenv("USE_MOCK_KAFKA", "true").lower() == "true"

    processor = get_stream_processor()
    if not processor:
        return {"status": "unavailable", "state": "unavailable"}

    health = {
        "status": "healthy" if processor.state.value == "running" else "unhealthy",
        "state": processor.state.value,
        "metrics": processor.get_metrics(),
    }

    if use_mock_kafka:
        mock_producer = get_mock_kafka_producer()
        if mock_producer:
            health["producer"] = mock_producer.get_metrics()
    else:
        from upishield_core.streaming import get_kafka_producer
        producer = get_kafka_producer()
        if producer:
            health["producer"] = producer.get_metrics()

    return health


@app.post("/api/v1/streaming/test-event", tags=["Streaming"])
async def send_test_streaming_event(
    event_type: str = "transaction",
    payload: Optional[Dict] = None,
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """Send a test event to streaming (mock or real)."""
    import os
    use_mock_kafka = os.getenv("USE_MOCK_KAFKA", "true").lower() == "true"

    if not payload:
        payload = {"test": True, "timestamp": datetime.utcnow().isoformat()}

    try:
        if use_mock_kafka:
            from upishield_core.streaming.mock_kafka import (
                MockKafkaEvent,
                EventType,
                EventPriority,
                create_mock_transaction_event,
                create_mock_fraud_alert_event,
                create_mock_scoring_result_event,
            )

            if event_type == "transaction":
                event = await create_mock_transaction_event(payload, correlation_id, user_id)
            elif event_type == "fraud_alert":
                event = await create_mock_fraud_alert_event(payload, correlation_id, user_id)
            elif event_type == "scoring_result":
                event = await create_mock_scoring_result_event(payload, correlation_id, user_id)
            else:
                event = MockKafkaEvent(
                    event_type=EventType(event_type),
                    payload=payload,
                    correlation_id=correlation_id,
                    user_id=user_id,
                )

            mock_producer = get_mock_kafka_producer()
            if not mock_producer:
                return {"error": "Mock Kafka producer not initialized"}

            # Send to the topic that the consumer is listening to
            target_topic = "transactions" if event_type == "transaction" else "scoring-requests"
            await mock_producer.send(target_topic, event)
            return {"status": "sent", "event_id": event.event_id, "mode": "mock", "topic": target_topic}
        else:
            from upishield_core.streaming import (
                KafkaEvent,
                EventType,
                EventPriority,
                create_transaction_event,
                create_fraud_alert_event,
                create_scoring_result_event,
                get_kafka_producer,
            )

            if event_type == "transaction":
                event = await create_transaction_event(payload, correlation_id, user_id)
            elif event_type == "fraud_alert":
                event = await create_fraud_alert_event(payload, correlation_id, user_id)
            elif event_type == "scoring_result":
                event = await create_scoring_result_event(payload, correlation_id, user_id)
            else:
                event = KafkaEvent(
                    event_type=EventType(event_type),
                    payload=payload,
                    correlation_id=correlation_id,
                    user_id=user_id,
                )

            producer = get_kafka_producer()
            if not producer:
                return {"error": "Kafka producer not initialized"}

            # Send to the topic that the consumer is listening to
            target_topic = "transactions" if event_type == "transaction" else "scoring-requests"
            await producer.send(target_topic, event)
            return {"status": "sent", "event_id": event.event_id, "mode": "real", "topic": target_topic}

    except Exception as e:
        logger.error("test_event_failed", error=str(e))
        return {"error": str(e), "mode": "mock" if use_mock_kafka else "real"}


# Analytics endpoints
@app.get("/api/v1/analytics/overview", tags=["Analytics"])
async def get_analytics_overview():
    """Get dashboard overview metrics."""
    # In production, query from database/timeseries
    return {
        "total_transactions_24h": 125847,
        "fraud_detected_24h": 342,
        "fraud_rate": 0.0027,
        "avg_latency_ms": 23.4,
        "p95_latency_ms": 45.2,
        "p99_latency_ms": 48.7,
        "decisions": {
            "approve": 120500,
            "review": 5005,
            "decline": 342,
        },
        "top_fraud_types": [
            {"type": "velocity_abuse", "count": 156},
            {"type": "new_user_high_value", "count": 89},
            {"type": "device_anomaly", "count": 54},
            {"type": "merchant_risk", "count": 43},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/analytics/trends", tags=["Analytics"])
async def get_trends(hours: int = 24):
    """Get fraud trends over time."""
    # In production, query from database
    import random

    now = datetime.utcnow()
    data = []
    for i in range(hours):
        dt = now - timedelta(hours=hours - i)
        data.append(
            {
                "timestamp": dt.isoformat(),
                "transactions": random.randint(4000, 6000),
                "fraud_detected": random.randint(10, 30),
                "avg_latency_ms": round(random.uniform(20, 30), 1),
            }
        )

    return {"trends": data}


@app.get("/api/v1/analytics/geography", tags=["Analytics"])
async def get_geography():
    """Get geographic fraud distribution."""
    # In production, query from database
    return {
        "states": [
            {"state": "Maharashtra", "transactions": 25847, "fraud_rate": 0.0032},
            {"state": "Karnataka", "transactions": 18934, "fraud_rate": 0.0028},
            {"state": "Tamil Nadu", "transactions": 15672, "fraud_rate": 0.0025},
            {"state": "Delhi", "transactions": 12456, "fraud_rate": 0.0035},
            {"state": "Gujarat", "transactions": 11234, "fraud_rate": 0.0021},
            {"state": "Telangana", "transactions": 9876, "fraud_rate": 0.0029},
            {"state": "Uttar Pradesh", "transactions": 8765, "fraud_rate": 0.0023},
            {"state": "West Bengal", "transactions": 7654, "fraud_rate": 0.0026},
        ]
    }


@app.get("/api/v1/analytics/merchant-risk", tags=["Analytics"])
async def get_merchant_risk(limit: int = 20):
    """Get top risky merchants by risk score."""
    # In production, query from database with aggregations
    return {
        "merchants": [
            {"merchant": "crypto_exchange@upi", "risk_score": 0.85, "volume": 12400, "fraud_count": 1054, "category": "crypto"},
            {"merchant": "gaming_site@upi", "risk_score": 0.72, "volume": 8900, "fraud_count": 641, "category": "gaming"},
            {"merchant": "new_merchant@upi", "risk_score": 0.68, "volume": 5600, "fraud_count": 381, "category": "new"},
            {"merchant": "electronics@upi", "risk_score": 0.45, "volume": 23400, "fraud_count": 1053, "category": "electronics"},
            {"merchant": "fashion_store@upi", "risk_score": 0.32, "volume": 18200, "fraud_count": 582, "category": "fashion"},
            {"merchant": "food_delivery@upi", "risk_score": 0.28, "volume": 34500, "fraud_count": 966, "category": "food"},
            {"merchant": "ride_sharing@upi", "risk_score": 0.22, "volume": 28900, "fraud_count": 636, "category": "transport"},
            {"merchant": "grocery_store@upi", "risk_score": 0.15, "volume": 45600, "fraud_count": 684, "category": "grocery"},
            {"merchant": "pharmacy@upi", "risk_score": 0.12, "volume": 12300, "fraud_count": 148, "category": "healthcare"},
            {"merchant": "utility_bills@upi", "risk_score": 0.08, "volume": 56700, "fraud_count": 454, "category": "utilities"},
        ][:limit]
    }


@app.get("/api/v1/analytics/device-risk", tags=["Analytics"])
async def get_device_risk(limit: int = 20):
    """Get device risk profiles by device type."""
    # In production, query from database with aggregations
    return {
        "devices": [
            {"device_type": "Android (Rooted)", "risk_score": 0.78, "count": 2100, "fraud_count": 1638, "os": "android", "is_rooted": True},
            {"device_type": "iOS (Jailbroken)", "risk_score": 0.72, "count": 890, "fraud_count": 641, "os": "ios", "is_jailbroken": True},
            {"device_type": "Android (VPN)", "risk_score": 0.55, "count": 5600, "fraud_count": 3080, "os": "android", "has_vpn": True},
            {"device_type": "Web (Tor)", "risk_score": 0.89, "count": 340, "fraud_count": 303, "os": "web", "has_tor": True},
            {"device_type": "iOS (Normal)", "risk_score": 0.12, "count": 45200, "fraud_count": 5424, "os": "ios", "is_normal": True},
            {"device_type": "Android (Normal)", "risk_score": 0.18, "count": 38900, "fraud_count": 7002, "os": "android", "is_normal": True},
            {"device_type": "Windows (Normal)", "risk_score": 0.25, "count": 12400, "fraud_count": 3100, "os": "windows", "is_normal": True},
            {"device_type": "Mac (Normal)", "risk_score": 0.15, "count": 8700, "fraud_count": 1305, "os": "mac", "is_normal": True},
            {"device_type": "Linux (Normal)", "risk_score": 0.35, "count": 3200, "fraud_count": 1120, "os": "linux", "is_normal": True},
            {"device_type": "Android (Emulator)", "risk_score": 0.68, "count": 1800, "fraud_count": 1224, "os": "android", "is_emulator": True},
        ][:limit]
    }


# Scam library endpoints
@app.get("/api/v1/scams", tags=["Scam Library"])
async def list_scams(limit: int = 20, offset: int = 0, category: str = None):
    """List known scam patterns."""
    # In production, query from database
    scams = [
        {
            "id": "scam_001",
            "name": "Fake Job Offer",
            "category": "job_scam",
            "description": "High-paying remote job requiring upfront payment",
            "indicators": ["upfront_fee", "personal_info_request", "urgent_action"],
            "severity": "high",
            "reported_count": 1234,
            "last_seen": "2024-01-15T10:00:00Z",
        },
        {
            "id": "scam_002",
            "name": "KYC Verification Scam",
            "category": "sms_phishing",
            "description": "Fake SMS claiming KYC expiry, links to phishing site",
            "indicators": ["urgent", "link_click", "personal_info"],
            "severity": "critical",
            "reported_count": 5678,
            "last_seen": "2024-01-14T15:30:00Z",
        },
        {
            "id": "scam_003",
            "name": "QR Code Payment Scam",
            "category": "upi_fraud",
            "description": "Fraudster sends QR code to receive money instead of paying",
            "indicators": ["qr_code", "receive_money", "social_engineering"],
            "severity": "high",
            "reported_count": 2345,
            "last_seen": "2024-01-13T09:00:00Z",
        },
    ]

    if category:
        scams = [s for s in scams if s["category"] == category]

    return {"scams": scams[offset : offset + limit], "total": len(scams)}


@app.get("/api/v1/scams/{scam_id}", tags=["Scam Library"])
async def get_scam(scam_id: str):
    """Get detailed scam information."""
    # In production, query from database
    return {
        "id": scam_id,
        "name": "Fake Job Offer",
        "category": "job_scam",
        "description": "High-paying remote job requiring upfront payment",
        "full_description": "Scammers post fake job listings offering high salaries for remote work. They request upfront payment for 'training materials', 'equipment', or 'background checks'. Once paid, they disappear.",
        "indicators": [
            "upfront_fee",
            "personal_info_request",
            "urgent_action",
            "too_good_to_be_true",
        ],
        "severity": "high",
        "reported_count": 1234,
        "variants": [
            "Data entry job - pay for software",
            "Customer service - pay for headset",
            "Typing job - pay for training",
        ],
        "prevention_tips": [
            "Never pay money to get a job",
            "Verify company independently",
            "Check for official email domains",
            "Report suspicious listings",
        ],
        "last_seen": "2024-01-15T10:00:00Z",
    }


@app.post("/api/v1/scams", tags=["Scam Library"])
async def add_scam(scam: Dict):
    """Add new scam pattern to library."""
    # In production, save to database
    return {"status": "added", "id": f"scam_{uuid.uuid4().hex[:8]}"}


# Cases (Investigation) endpoints
class CaseNote(BaseModel):
    content: str
    type: str = "investigation"


class CaseEvidence(BaseModel):
    type: str  # screenshot, document, log, transaction_record
    title: str
    description: str
    url: str
    metadata: Dict = {}


class Case(BaseModel):
    id: str
    title: str
    description: str
    status: str  # open, in_progress, resolved, closed
    priority: str  # low, medium, high, critical
    assignee: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    alert_ids: List[str] = []
    transaction_ids: List[str] = []
    tags: List[str] = []
    notes: List[Dict] = []
    evidence: List[Dict] = []


class CaseListResponse(BaseModel):
    cases: List[Case]
    total: int


# In-memory storage for cases (in production, use database)
_cases_db: Dict[str, Case] = {}


# Initialize some demo cases
def init_demo_cases():
    global _cases_db
    if not _cases_db:
        now = datetime.utcnow()
        _cases_db = {
            "case_001": Case(
                id="case_001",
                title="Coordinated Fraud Attack - IP Range 203.0.113.0/24",
                description="Multiple high-value transactions from suspicious IP range",
                status="in_progress",
                priority="critical",
                assignee="fraud-lead@company.com",
                created_at=now - timedelta(days=1),
                updated_at=now - timedelta(hours=4),
                alert_ids=["alert_005"],
                transaction_ids=["txn_001", "txn_002", "txn_003"],
                tags=["coordinated-attack", "ip-block"],
                notes=[
                    {"content": "Initial investigation started", "type": "investigation", "author": "fraud-lead@company.com", "created_at": (now - timedelta(days=1)).isoformat()},
                    {"content": "Identified 47 related transactions", "type": "investigation", "author": "fraud-lead@company.com", "created_at": (now - timedelta(hours=4)).isoformat()},
                ],
                evidence=[],
            ),
            "case_002": Case(
                id="case_002",
                title="Suspicious Activity - User user123@upi",
                description="Unusual transaction pattern detected",
                status="open",
                priority="high",
                assignee="analyst@company.com",
                created_at=now - timedelta(hours=5),
                updated_at=now - timedelta(hours=5),
                alert_ids=["alert_003"],
                transaction_ids=["txn_abc123"],
                tags=["velocity", "new-device"],
                notes=[],
                evidence=[],
            ),
            "case_003": Case(
                id="case_003",
                title="QR Code Scam - Merchant fraud",
                description="Multiple reports of QR code fraud at local merchants",
                status="resolved",
                priority="medium",
                assignee="analyst@company.com",
                created_at=now - timedelta(days=3),
                updated_at=now - timedelta(days=1),
                alert_ids=["alert_001"],
                transaction_ids=["txn_500", "txn_501"],
                tags=["qr-code", "merchant-fraud"],
                notes=[
                    {"content": "Merchant accounts identified and blocked", "type": "resolution", "author": "analyst@company.com", "created_at": (now - timedelta(days=1)).isoformat()},
                ],
                evidence=[],
            ),
            "case_004": Case(
                id="case_004",
                title="KYC Phishing Campaign",
                description="Coordinated SMS phishing campaign targeting KYC verification",
                status="closed",
                priority="critical",
                assignee="senior-analyst@company.com",
                created_at=now - timedelta(days=7),
                updated_at=now - timedelta(days=5),
                alert_ids=["alert_002"],
                transaction_ids=[],
                tags=["phishing", "kyc", "sms"],
                notes=[
                    {"content": "Campaign infrastructure identified", "type": "investigation", "author": "senior-analyst@company.com", "created_at": (now - timedelta(days=6)).isoformat()},
                    {"content": "Phishing domains reported to registrar", "type": "action", "author": "senior-analyst@company.com", "created_at": (now - timedelta(days=5)).isoformat()},
                ],
                evidence=[],
            ),
        }


init_demo_cases()


# Alert endpoints
class Alert(BaseModel):
    id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low
    status: str  # active, acknowledged, resolved, silenced
    source: str
    rule_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict = {}


class AlertListResponse(BaseModel):
    alerts: List[Alert]
    total: int


class AlertRule(BaseModel):
    id: str
    name: str
    description: str
    condition: str
    severity: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


class NotificationChannel(BaseModel):
    id: str
    name: str
    type: str  # email, slack, webhook, pagerduty
    config: Dict
    enabled: bool
    created_at: datetime
    updated_at: datetime


# In-memory storage for alerts (in production, use database)
_alerts_db: Dict[str, Alert] = {}
_alert_rules_db: Dict[str, AlertRule] = {}
_notification_channels_db: Dict[str, NotificationChannel] = {}


# Initialize some demo alerts
def init_demo_alerts():
    global _alerts_db, _alert_rules_db, _notification_channels_db

    if not _alerts_db:
        now = datetime.utcnow()
        _alerts_db = {
            "alert_001": Alert(
                id="alert_001",
                title="High Velocity Transaction Pattern Detected",
                description="Multiple rapid transactions from same device in short time window",
                severity="high",
                status="active",
                source="velocity_rule",
                rule_id="rule_001",
                created_at=now - timedelta(minutes=5),
                updated_at=now - timedelta(minutes=5),
            ),
            "alert_002": Alert(
                id="alert_002",
                title="New Device High Value Transaction",
                description="First transaction from new device exceeding 50,000 INR",
                severity="critical",
                status="active",
                source="device_anomaly_rule",
                rule_id="rule_002",
                created_at=now - timedelta(minutes=12),
                updated_at=now - timedelta(minutes=12),
            ),
            "alert_003": Alert(
                id="alert_003",
                title="Suspicious Merchant Category",
                description="Transaction to high-risk merchant category (gambling/crypto)",
                severity="medium",
                status="acknowledged",
                source="merchant_risk_rule",
                rule_id="rule_003",
                created_at=now - timedelta(hours=1),
                updated_at=now - timedelta(minutes=30),
                acknowledged_at=now - timedelta(minutes=30),
            ),
            "alert_004": Alert(
                id="alert_004",
                title="Geographic Anomaly - Impossible Travel",
                description="Transaction from location impossible given previous transaction timestamp",
                severity="high",
                status="resolved",
                source="geo_velocity_rule",
                rule_id="rule_004",
                created_at=now - timedelta(hours=3),
                updated_at=now - timedelta(hours=2),
                resolved_at=now - timedelta(hours=2),
            ),
            "alert_005": Alert(
                id="alert_005",
                title="Repeated Failed PIN Attempts",
                description="More than 3 failed PIN attempts within 10 minutes",
                severity="critical",
                status="active",
                source="pin_retry_rule",
                rule_id="rule_005",
                created_at=now - timedelta(minutes=2),
                updated_at=now - timedelta(minutes=2),
            ),
        }

    if not _alert_rules_db:
        now = datetime.utcnow()
        _alert_rules_db = {
            "rule_001": AlertRule(
                id="rule_001",
                name="Velocity Abuse Detection",
                description="Flag transactions exceeding 5 per minute from same device",
                condition="count(txn) > 5 AND time_window < 60s GROUP BY device_id",
                severity="high",
                enabled=True,
                created_at=now - timedelta(days=30),
                updated_at=now - timedelta(days=1),
            ),
            "rule_002": AlertRule(
                id="rule_002",
                name="New Device High Value",
                description="Flag first transaction > 50,000 INR from unknown device",
                condition="is_new_device AND amount > 50000",
                severity="critical",
                enabled=True,
                created_at=now - timedelta(days=15),
                updated_at=now - timedelta(hours=2),
            ),
            "rule_003": AlertRule(
                id="rule_003",
                name="High Risk Merchant Category",
                description="Flag transactions to gambling, crypto, or adult merchants",
                condition="merchant_category IN ('gambling', 'crypto', 'adult')",
                severity="medium",
                enabled=True,
                created_at=now - timedelta(days=60),
                updated_at=now - timedelta(days=5),
            ),
            "rule_004": AlertRule(
                id="rule_004",
                name="Impossible Travel Detection",
                description="Flag transactions from geographically impossible locations",
                condition="distance(prev_location, curr_location) / time_diff > 1000km/h",
                severity="high",
                enabled=False,
                created_at=now - timedelta(days=45),
                updated_at=now - timedelta(days=10),
            ),
            "rule_005": AlertRule(
                id="rule_005",
                name="PIN Brute Force Prevention",
                description="Alert on repeated failed PIN attempts",
                condition="failed_pin_count > 3 AND time_window < 600s",
                severity="critical",
                enabled=True,
                created_at=now - timedelta(days=90),
                updated_at=now - timedelta(days=1),
            ),
        }

    if not _notification_channels_db:
        now = datetime.utcnow()
        _notification_channels_db = {
            "channel_001": NotificationChannel(
                id="channel_001",
                name="Security Team Slack",
                type="slack",
                config={"webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz", "channel": "#security-alerts"},
                enabled=True,
                created_at=now - timedelta(days=10),
                updated_at=now - timedelta(days=1),
            ),
            "channel_002": NotificationChannel(
                id="channel_002",
                name="Email - Fraud Team",
                type="email",
                config={"recipients": ["fraud-team@company.com", "security@company.com"]},
                enabled=True,
                created_at=now - timedelta(days=5),
                updated_at=now - timedelta(hours=3),
            ),
            "channel_003": NotificationChannel(
                id="channel_003",
                name="PagerDuty Critical",
                type="pagerduty",
                config={"integration_key": "xxx", "service": "fraud-detection"},
                enabled=True,
                created_at=now - timedelta(days=20),
                updated_at=now - timedelta(days=2),
            ),
            "channel_004": NotificationChannel(
                id="channel_004",
                name="Webhook - SIEM",
                type="webhook",
                config={"url": "https://siem.company.com/api/alerts", "auth_token": "xxx"},
                enabled=False,
                created_at=now - timedelta(days=7),
                updated_at=now - timedelta(days=3),
            ),
        }


init_demo_alerts()


# Alert Rules endpoints - must be before parameterized /alerts/{alert_id} route
@app.get("/api/v1/alerts/rules", tags=["Alert Rules"])
async def list_alert_rules():
    """List all alert rules."""
    return {"rules": list(_alert_rules_db.values())}


@app.post("/api/v1/alerts/rules", tags=["Alert Rules"])
async def create_alert_rule(rule: Dict):
    """Create a new alert rule."""
    rule_id = f"rule_{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow()
    new_rule = AlertRule(
        id=rule_id,
        name=rule.get("name", "New Rule"),
        description=rule.get("description", ""),
        condition=rule.get("condition", ""),
        severity=rule.get("severity", "medium"),
        enabled=rule.get("enabled", True),
        created_at=now,
        updated_at=now,
    )
    _alert_rules_db[rule_id] = new_rule
    return new_rule


@app.put("/api/v1/alerts/rules/{rule_id}", tags=["Alert Rules"])
async def update_alert_rule(rule_id: str, rule: Dict):
    """Update an alert rule."""
    existing = _alert_rules_db.get(rule_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Rule not found")

    updated = AlertRule(
        id=rule_id,
        name=rule.get("name", existing.name),
        description=rule.get("description", existing.description),
        condition=rule.get("condition", existing.condition),
        severity=rule.get("severity", existing.severity),
        enabled=rule.get("enabled", existing.enabled),
        created_at=existing.created_at,
        updated_at=datetime.utcnow(),
    )
    _alert_rules_db[rule_id] = updated
    return updated


@app.delete("/api/v1/alerts/rules/{rule_id}", tags=["Alert Rules"])
async def delete_alert_rule(rule_id: str):
    """Delete an alert rule."""
    if rule_id not in _alert_rules_db:
        raise HTTPException(status_code=404, detail="Rule not found")

    del _alert_rules_db[rule_id]
    return {"status": "deleted", "id": rule_id}


@app.post("/api/v1/alerts/rules/{rule_id}/test", tags=["Alert Rules"])
async def test_alert_rule(rule_id: str):
    """Test an alert rule against sample data."""
    rule = _alert_rules_db.get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    # Simulate test result
    return {
        "rule_id": rule_id,
        "test_passed": True,
        "matches": 3,
        "sample_transactions_tested": 100,
        "message": f"Rule '{rule.name}' matched 3 out of 100 sample transactions",
    }


@app.post("/api/v1/alerts/rules/{rule_id}/toggle", tags=["Alert Rules"])
async def toggle_alert_rule(rule_id: str, enabled: bool):
    """Enable or disable an alert rule."""
    rule = _alert_rules_db.get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    rule.enabled = enabled
    rule.updated_at = datetime.utcnow()
    _alert_rules_db[rule_id] = rule
    return rule


# Notification Channels endpoints - must be before parameterized /alerts/{alert_id} route
@app.get("/api/v1/alerts/channels", tags=["Notification Channels"])
async def list_notification_channels():
    """List all notification channels."""
    return {"channels": list(_notification_channels_db.values())}


@app.post("/api/v1/alerts/channels", tags=["Notification Channels"])
async def create_notification_channel(channel: Dict):
    """Create a new notification channel."""
    channel_id = f"channel_{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow()
    new_channel = NotificationChannel(
        id=channel_id,
        name=channel.get("name", "New Channel"),
        type=channel.get("type", "email"),
        config=channel.get("config", {}),
        enabled=channel.get("enabled", True),
        created_at=now,
        updated_at=now,
    )
    _notification_channels_db[channel_id] = new_channel
    return new_channel


@app.put("/api/v1/alerts/channels/{channel_id}", tags=["Notification Channels"])
async def update_notification_channel(channel_id: str, channel: Dict):
    """Update a notification channel."""
    existing = _notification_channels_db.get(channel_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Channel not found")

    updated = NotificationChannel(
        id=channel_id,
        name=channel.get("name", existing.name),
        type=channel.get("type", existing.type),
        config=channel.get("config", existing.config),
        enabled=channel.get("enabled", existing.enabled),
        created_at=existing.created_at,
        updated_at=datetime.utcnow(),
    )
    _notification_channels_db[channel_id] = updated
    return updated


@app.delete("/api/v1/alerts/channels/{channel_id}", tags=["Notification Channels"])
async def delete_notification_channel(channel_id: str):
    """Delete a notification channel."""
    if channel_id not in _notification_channels_db:
        raise HTTPException(status_code=404, detail="Channel not found")

    del _notification_channels_db[channel_id]
    return {"status": "deleted", "id": channel_id}


@app.post("/api/v1/alerts/channels/{channel_id}/test", tags=["Notification Channels"])
async def test_notification_channel(channel_id: str):
    """Test a notification channel."""
    channel = _notification_channels_db.get(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    return {
        "channel_id": channel_id,
        "test_passed": True,
        "message": f"Test notification sent successfully to {channel.name} ({channel.type})",
    }


@app.post("/api/v1/alerts/channels/{channel_id}/toggle", tags=["Notification Channels"])
async def toggle_notification_channel(channel_id: str, enabled: bool):
    """Enable or disable a notification channel."""
    channel = _notification_channels_db.get(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    channel.enabled = enabled
    channel.updated_at = datetime.utcnow()
    _notification_channels_db[channel_id] = channel
    return channel


# Alert endpoints
@app.get("/api/v1/alerts", response_model=AlertListResponse, tags=["Alerts"])
async def list_alerts(
    severity: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List alerts with optional filtering."""
    alerts = list(_alerts_db.values())

    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    if status:
        alerts = [a for a in alerts if a.status == status]

    # Sort by created_at descending
    alerts.sort(key=lambda x: x.created_at, reverse=True)

    return AlertListResponse(
        alerts=alerts[offset : offset + limit],
        total=len(alerts),
    )


@app.get("/api/v1/alerts/{alert_id}", response_model=Alert, tags=["Alerts"])
async def get_alert(alert_id: str):
    """Get alert details."""
    alert = _alerts_db.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@app.post("/api/v1/alerts/{alert_id}/acknowledge", response_model=Alert, tags=["Alerts"])
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    alert = _alerts_db.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = "acknowledged"
    alert.acknowledged_at = datetime.utcnow()
    alert.updated_at = datetime.utcnow()
    _alerts_db[alert_id] = alert
    return alert


@app.post("/api/v1/alerts/{alert_id}/resolve", response_model=Alert, tags=["Alerts"])
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    alert = _alerts_db.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = "resolved"
    alert.resolved_at = datetime.utcnow()
    alert.updated_at = datetime.utcnow()
    _alerts_db[alert_id] = alert
    return alert


@app.post("/api/v1/alerts/{alert_id}/silence", response_model=Alert, tags=["Alerts"])
async def silence_alert(alert_id: str, duration: Optional[int] = None):
    """Silence an alert for a duration (minutes)."""
    alert = _alerts_db.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = "silenced"
    alert.updated_at = datetime.utcnow()
    if duration:
        alert.metadata["silence_until"] = (datetime.utcnow() + timedelta(minutes=duration)).isoformat()
    _alerts_db[alert_id] = alert
    return alert


# Cases (Investigation) endpoints
@app.get("/api/v1/cases", response_model=CaseListResponse, tags=["Cases"])
async def list_cases(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    assignee: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List cases with optional filtering."""
    cases = list(_cases_db.values())

    if status:
        cases = [c for c in cases if c.status == status]
    if priority:
        cases = [c for c in cases if c.priority == priority]
    if assignee:
        cases = [c for c in cases if c.assignee == assignee]

    # Sort by created_at descending
    cases.sort(key=lambda x: x.created_at, reverse=True)

    return CaseListResponse(
        cases=cases[offset : offset + limit],
        total=len(cases),
    )


@app.post("/api/v1/cases", response_model=Case, tags=["Cases"])
async def create_case(case: Dict):
    """Create a new investigation case."""
    case_id = f"case_{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow()
    new_case = Case(
        id=case_id,
        title=case.get("title", "New Case"),
        description=case.get("description", ""),
        status=case.get("status", "open"),
        priority=case.get("priority", "medium"),
        assignee=case.get("assignee"),
        created_at=now,
        updated_at=now,
        alert_ids=case.get("alert_ids", []),
        transaction_ids=case.get("transaction_ids", []),
        tags=case.get("tags", []),
        notes=[],
        evidence=[],
    )
    _cases_db[case_id] = new_case
    return new_case


@app.get("/api/v1/cases/{case_id}", response_model=Case, tags=["Cases"])
async def get_case(case_id: str):
    """Get case details."""
    case = _cases_db.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.patch("/api/v1/cases/{case_id}", response_model=Case, tags=["Cases"])
async def update_case(case_id: str, case: Dict):
    """Update a case."""
    existing = _cases_db.get(case_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Case not found")

    updated = Case(
        id=case_id,
        title=case.get("title", existing.title),
        description=case.get("description", existing.description),
        status=case.get("status", existing.status),
        priority=case.get("priority", existing.priority),
        assignee=case.get("assignee", existing.assignee),
        created_at=existing.created_at,
        updated_at=datetime.utcnow(),
        alert_ids=case.get("alert_ids", existing.alert_ids),
        transaction_ids=case.get("transaction_ids", existing.transaction_ids),
        tags=case.get("tags", existing.tags),
        notes=existing.notes,
        evidence=existing.evidence,
    )
    _cases_db[case_id] = updated
    return updated


@app.post("/api/v1/cases/{case_id}/notes", response_model=Case, tags=["Cases"])
async def add_case_note(case_id: str, note: CaseNote):
    """Add a note to a case."""
    case = _cases_db.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    new_note = {
        "content": note.content,
        "type": note.type,
        "author": "analyst@company.com",  # In production, get from auth
        "created_at": datetime.utcnow().isoformat(),
    }
    case.notes.append(new_note)
    case.updated_at = datetime.utcnow()
    _cases_db[case_id] = case
    return case


@app.post("/api/v1/cases/{case_id}/evidence", response_model=Case, tags=["Cases"])
async def add_case_evidence(case_id: str, evidence: CaseEvidence):
    """Add evidence to a case."""
    case = _cases_db.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    new_evidence = {
        "type": evidence.type,
        "title": evidence.title,
        "description": evidence.description,
        "url": evidence.url,
        "metadata": evidence.metadata,
        "added_at": datetime.utcnow().isoformat(),
    }
    case.evidence.append(new_evidence)
    case.updated_at = datetime.utcnow()
    _cases_db[case_id] = case
    return case


@app.post("/api/v1/cases/{case_id}/close", response_model=Case, tags=["Cases"])
async def close_case(case_id: str, closure: Dict):
    """Close a case with resolution."""
    case = _cases_db.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if case.status == "closed":
        raise HTTPException(status_code=409, detail="Case already closed")

    case.status = "closed"
    case.updated_at = datetime.utcnow()
    case.notes.append({
        "content": f"Case closed: {closure.get('resolution', 'resolved')}. {closure.get('summary', '')}",
        "type": "resolution",
        "author": "analyst@company.com",
        "created_at": datetime.utcnow().isoformat(),
    })
    _cases_db[case_id] = case
    return case


# RAG Knowledge Base endpoints - using new knowledge base from ai module
@app.post("/api/v1/rag/search", response_model=RAGResponse, tags=["Knowledge"])
async def rag_search(request: RAGQuery):
    """
    Search the fraud knowledge base for relevant information.

    Uses semantic search to find relevant fraud detection knowledge,
    scam patterns, and prevention tips.
    """
    if not app_state["rag"] or not app_state["rag"]._initialized:
        # Return empty results if RAG not available, but don't error
        return RAGResponse(results=[], query=request.query, total_chunks_searched=0)

    results = app_state["rag"].search(
        query=request.query,
        top_k=request.top_k,
        category_filter=request.category_filter,
        min_score=request.min_score,
    )

    return RAGResponse(
        results=[r.to_dict() for r in results],
        query=request.query,
        total_chunks_searched=app_state["rag"].vector_store.count(),
    )


# AI Assistant endpoints
class AIMessage(BaseModel):
    """Message in a conversation."""
    role: str  # user, assistant
    content: str


class AIChatRequest(BaseModel):
    """Request for AI-assisted fraud analysis."""
    messages: List[AIMessage]
    session_id: Optional[str] = None
    include_sources: bool = True
    max_tokens: int = 1000


class AIChatResponse(BaseModel):
    """Response from AI assistant."""
    response: str
    sources: Optional[List[Dict]] = None
    confidence: float
    processing_time_ms: float
    session_id: str


class AIChatStreamRequest(BaseModel):
    """Request for streaming AI responses."""
    messages: List[AIMessage]
    session_id: Optional[str] = None
    max_tokens: int = 1000


@app.post("/api/v1/ai/chat", response_model=AIChatResponse, tags=["AI Assistant"])
async def ai_chat(request: AIChatRequest):
    """
    AI-assisted fraud analysis and education.

    Analyzes conversation context and provides:
    - Transaction analysis and recommendations
    - Scam detection and prevention advice
    - Knowledge base answers to fraud questions
    - Investigation insights from similar cases
    """
    start_time = time.perf_counter()

    try:
        # Import the AI assistant logic
        from upishield_core.ai.assistant import AIAssistant

        assistant = AIAssistant(
            rag=get_rag() if app_state["rag"] else None,
            decision_engine=app_state["decision_engine"],
        )

        response = await assistant.process_request(
            messages=request.messages,
            session_id=request.session_id,
        )

        processing_time = (time.perf_counter() - start_time) * 1000

        return AIChatResponse(
            response=response["response"],
            sources=response.get("sources") if request.include_sources else None,
            confidence=response.get("confidence", 0.5),
            processing_time_ms=processing_time,
            session_id=response.get("session_id", request.session_id or str(uuid.uuid4())[:8]),
        )

    except ImportError:
        # Fallback to deterministic response if AI module not available
        return AIChatResponse(
            response="AI assistant not configured. Please use the /api/v1/rag/search endpoint for knowledge base queries.",
            sources=None,
            confidence=0.0,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            session_id=request.session_id or str(uuid.uuid4())[:8],
        )
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/api/v1/ai/chat", error_type=type(e).__name__).inc()
        logger.error("ai_chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")


@app.post("/api/v1/ai/chat/stream")
async def ai_chat_stream(request: AIChatStreamRequest):
    """
    Stream AI responses token-by-token using Server-Sent Events.

    Provides real-time streaming for longer responses and better UX.
    Events: 'response' (token chunks), 'done', 'error'
    """
    from fastapi.responses import StreamingResponse
    import json

    async def generate():
        try:
            from upishield_core.ai.assistant import AIAssistant

            assistant = AIAssistant(
                rag=get_rag() if app_state["rag"] else None,
                decision_engine=app_state["decision_engine"],
            )

            # Simulate streaming response with tokens
            response = await assistant.process_request(
                messages=request.messages,
                session_id=request.session_id,
            )

            # Yield SSE events
            response_text = response["response"]

            # Send response start
            yield "event: response\n"
            data = json.dumps({"type": "start", "content": ""})
            yield f"data: {data}\n\n"

            # Stream tokens (simulated - in production, use actual token streaming)
            tokens = response_text.split()
            for i, token in enumerate(tokens):
                is_last = i == len(tokens) - 1
                token_data = json.dumps({"type": "token", "content": token, "final": is_last})
                yield f"data: {token_data}\n\n"
                await asyncio.sleep(0.01)  # Simulate token generation delay

            # Send sources if available
            if response.get("sources"):
                sources_data = json.dumps({"type": "sources", "sources": response["sources"]})
                yield f"data: {sources_data}\n\n"

            # Send done event
            yield "event: done\n"
            confidence_data = json.dumps({"confidence": response.get("confidence", 0.5)})
            yield f"data: {confidence_data}\n\n"

        except Exception as e:
            yield f"event: error\n"
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# Utility functions
async def log_decision(txn_id: str, context):
    """Log decision for audit trail."""
    logger.info(
        "fraud_decision",
        txn_id=txn_id,
        decision=context.decision.value,
        risk_level=context.risk_level.value,
        final_score=context.final_score,
        latency_ms=context.latency_ms,
    )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    ERROR_COUNTER.labels(endpoint=request.url.path, error_type=type(exc).__name__).inc()
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_id": str(uuid.uuid4())[:8]},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
