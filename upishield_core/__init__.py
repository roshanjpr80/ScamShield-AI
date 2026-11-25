"""
UPIShield Core - Real-time UPI Fraud Detection Inference Engine

Sub-50ms p99 latency fraud detection for India's UPI ecosystem.
Production-grade, battle-tested, open source.

Example:
    >>> from upishield_core import UPIShield
    >>> from upishield_core.api.schemas import UPITransaction
    >>>
    >>> shield = UPIShield(model_dir="./models")
    >>> txn = UPITransaction(txn_id="123", payer_vpa="user@upi", payee_vpa="merchant@upi", amount=50000)
    >>> result = shield.score(txn)
    >>> print(f"Risk: {result.risk_score}/100 | Action: {result.action}")
"""

from upishield_core.api.client import UPIShieldClient, UPIShieldClientConfig, UPIShieldSyncClient
from upishield_core.api.schemas import (
    ScoreRequest,
    ScoreResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ModelInfoResponse,
    MetricsResponse,
    JobResponse,
    JobStatus,
    TransactionStatus,
    TransactionType,
    UPITransaction,
)
from upishield_core.decision.engine import (
    DecisionContext,
    DecisionEngine,
    DecisionEngineSync,
    FraudDecision,
    RiskLevel,
    RuleEngine,
    RuleResult,
    ThreatIntelligenceClient,
    ThreatIntelResult,
)
from upishield_core.features.compute import BatchFeatureComputer, FeatureComputer
from upishield_core.features.schemas import (
    DeviceNetworkFeatures,
    GraphFeatures,
    MerchantFeatures,
    UPIFeatures,
    UserBehavioralFeatures,
    VelocityFeatures,
)
from upishield_core.features.store import FeatureStore, FeatureStoreConfig
from upishield_core.models.ensemble import EnsembleInference, ModelPrediction, ONNXEnsemble
from upishield_core.models.job_fraud_detector import (
    JobFraudCategory,
    JobFraudDetector,
    JobFraudPrediction,
)
from upishield_core.models.registry import ModelMetadata, ModelRegistry
from upishield_core.models.smishing_detector import (
    SmishingCategory,
    SmishingDetector,
    SmishingPrediction,
)
from upishield_core.observability.metrics import (
    REGISTRY,
    MetricsCollector,
    count_requests,
    get_metrics,
    track_latency,
)
from upishield_core.observability.tracing import (
    TracingHelper,
    get_tracer,
    init_tracing,
    span,
    traced,
)

__version__ = "0.1.0"
__author__ = "ScamShield AI Team"
__license__ = "Apache-2.0"

__all__ = [
    # Models
    "ONNXEnsemble",
    "EnsembleInference",
    "ModelPrediction",
    "ModelRegistry",
    "ModelMetadata",
    "JobFraudDetector",
    "JobFraudPrediction",
    "JobFraudCategory",
    "SmishingDetector",
    "SmishingCategory",
    "SmishingPrediction",
    # Features
    "FeatureStore",
    "FeatureStoreConfig",
    "UPIFeatures",
    "UserBehavioralFeatures",
    "DeviceNetworkFeatures",
    "MerchantFeatures",
    "VelocityFeatures",
    "GraphFeatures",
    "FeatureComputer",
    "BatchFeatureComputer",
    # Decision
    "DecisionEngine",
    "DecisionEngineSync",
    "RuleEngine",
    "ThreatIntelligenceClient",
    "DecisionContext",
    "RuleResult",
    "ThreatIntelResult",
    "FraudDecision",
    "RiskLevel",
    # API
    "UPIShieldClient",
    "UPIShieldSyncClient",
    "UPIShieldClientConfig",
    "UPITransaction",
    "TransactionType",
    "TransactionStatus",
    "ScoreRequest",
    "ScoreResponse",
    "BatchScoreRequest",
    "BatchScoreResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "MetricsResponse",
    "JobResponse",
    "JobStatus",
    # Observability
    "REGISTRY",
    "get_metrics",
    "MetricsCollector",
    "track_latency",
    "count_requests",
    "init_tracing",
    "get_tracer",
    "span",
    "traced",
    "TracingHelper",
]


class UPIShield:
    """
    High-level synchronous interface for local inference.

    Usage:
        shield = UPIShield(model_dir="./models", redis_url="redis://localhost:6379")
        result = shield.score(transaction)
    """

    def __init__(
        self,
        model_dir: str = "./models",
        redis_url: str = None,
        config: dict = None,
    ):
        self.model_dir = model_dir
        self.redis_url = redis_url
        self.config = config or {}

        # Initialize components lazily
        self._ensemble: ONNXEnsemble = None
        self._feature_store: FeatureStore = None
        self._decision_engine: DecisionEngineSync = None

    @property
    def ensemble(self) -> ONNXEnsemble:
        if self._ensemble is None:
            self._ensemble = ONNXEnsemble(self.model_dir)
        return self._ensemble

    @property
    def feature_store(self) -> FeatureStore:
        if self._feature_store is None and self.redis_url:
            self._feature_store = FeatureStore(FeatureStoreConfig(redis_url=self.redis_url))
        return self._feature_store

    @property
    def decision_engine(self) -> DecisionEngineSync:
        if self._decision_engine is None:
            self._decision_engine = DecisionEngineSync(
                ensemble=self.ensemble,
                registry=ModelRegistry(self.model_dir),
            )
        return self._decision_engine

    def score(self, transaction: UPITransaction) -> ScoreResponse:
        """Score a UPI transaction synchronously."""
        return self.decision_engine.score(transaction)

    async def score_async(self, transaction: UPITransaction) -> ScoreResponse:
        """Score a UPI transaction asynchronously."""
        # Use the async engine directly
        from upishield_core.decision.engine import DecisionEngine

        async_engine = DecisionEngine(
            ensemble=self.ensemble,
            registry=ModelRegistry(self.model_dir),
        )
        return await async_engine.score(transaction)

    def score_batch(self, transactions: list[UPITransaction]) -> list[ScoreResponse]:
        """Score multiple transactions."""
        return [self.score(txn) for txn in transactions]

    async def score_batch_async(self, transactions: list[UPITransaction]) -> list[ScoreResponse]:
        """Score multiple transactions asynchronously."""
        from upishield_core.decision.engine import DecisionEngine

        async_engine = DecisionEngine(
            ensemble=self.ensemble,
            registry=ModelRegistry(self.model_dir),
        )
        return await async_engine.score_batch(transactions)

    def warm_up(self, num_requests: int = 10):
        """Warm up models with dummy requests."""
        dummy_txn = UPITransaction(
            txn_id="warmup",
            payer_vpa="user@upi",
            payee_vpa="merchant@upi",
            amount=100.0,
            device_id="warmup_device",
            client_ip="127.0.0.1",
        )
        for _ in range(num_requests):
            self.score(dummy_txn)
