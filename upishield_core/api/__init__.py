"""API module for UPIShield."""

from upishield_core.api.client import (
    UPIShieldClient,
    UPIShieldClientConfig,
    UPIShieldSyncClient,
    quick_score,
    quick_score_sync,
)
from upishield_core.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    FraudDecision,
    HealthResponse,
    JobResponse,
    JobStatus,
    MetricsResponse,
    ModelInfoResponse,
    RiskLevel,
    ScoreRequest,
    ScoreResponse,
    TransactionStatus,
    TransactionType,
    UPITransaction,
)

__all__ = [
    # Client
    "UPIShieldClient",
    "UPIShieldSyncClient",
    "UPIShieldClientConfig",
    "quick_score",
    "quick_score_sync",
    # Schemas
    "UPITransaction",
    "TransactionType",
    "TransactionStatus",
    "FraudDecision",
    "RiskLevel",
    "ScoreRequest",
    "ScoreResponse",
    "BatchScoreRequest",
    "BatchScoreResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "JobStatus",
    "JobResponse",
    "MetricsResponse",
]
