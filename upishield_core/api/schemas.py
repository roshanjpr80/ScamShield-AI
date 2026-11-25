"""
API Schemas for UPIShield Core.

Pydantic models for request/response validation.
Compatible with the main ScamShield API schemas.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated


class TransactionType(str, Enum):
    """UPI transaction types."""

    P2P = "P2P"  # Peer to Peer
    P2M = "P2M"  # Peer to Merchant
    M2P = "M2P"  # Merchant to Peer
    COLLECT = "COLLECT"  # Collect request
    REFUND = "REFUND"
    UNKNOWN = "UNKNOWN"


class TransactionStatus(str, Enum):
    """Transaction processing status."""

    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DECLINED = "DECLINED"
    EXPIRED = "EXPIRED"


class FraudDecision(str, Enum):
    """Fraud decision outcomes."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class RiskLevel(str, Enum):
    """Risk level classification."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class UPITransaction(BaseModel):
    """
    UPI Transaction for scoring.

    Matches the main ScamShield transaction schema.
    """

    model_config = ConfigDict(extra="allow")

    txn_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    payer_vpa: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    payee_vpa: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    currency: str = Field(default="INR", max_length=3)
    txn_type: TransactionType = TransactionType.P2M
    status: TransactionStatus = TransactionStatus.PENDING
    timestamp: Optional[datetime] = None
    device_id: Optional[Annotated[str, StringConstraints(max_length=128)]] = None
    client_ip: Optional[Annotated[str, StringConstraints(max_length=45)]] = None
    merchant_category: Optional[str] = None
    merchant_name: Optional[str] = None
    location: Optional[str] = None
    raw_sms: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


class ScoreRequest(BaseModel):
    """Request for single transaction scoring."""

    transaction: UPITransaction
    async_mode: bool = False
    include_explanation: bool = True
    include_features: bool = False


class ScoreResponse(BaseModel):
    """Response from transaction scoring."""

    txn_id: str
    decision: FraudDecision
    risk_level: RiskLevel
    risk_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    model_version: str
    processing_time_ms: float
    explanation: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, float]] = None
    triggered_rules: List[Dict[str, Any]] = Field(default_factory=list)
    threat_intel: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchScoreRequest(BaseModel):
    """Request for batch transaction scoring."""

    transactions: List[UPITransaction]
    include_explanation: bool = True
    include_features: bool = False


class BatchScoreResponse(BaseModel):
    """Response from batch scoring."""

    results: List[ScoreResponse]
    total_processed: int
    total_time_ms: float
    batch_id: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float
    models_loaded: int
    redis_connected: bool
    database_connected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """Model information response."""

    models: Dict[str, Dict[str, Any]]
    active_model: str
    model_registry_version: str
    last_updated: datetime


class JobStatus(str, Enum):
    """Async job status."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobResponse(BaseModel):
    """Async job response."""

    job_id: str
    status: JobStatus
    result: Optional[ScoreResponse] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class MetricsResponse(BaseModel):
    """Metrics response."""

    total_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    decisions: Dict[str, int]
    errors: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FraudCasePriority(str, Enum):
    """Fraud case priority levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CaseListResponse(BaseModel):
    """Paginated list of fraud cases."""

    cases: List["FraudCaseResponse"]
    total: int
    page: int
    page_size: int

    model_config = ConfigDict(from_attributes=True)


class FraudCaseCreate(BaseModel):
    """Create a new fraud case."""

    organization_id: uuid.UUID
    title: str
    created_by: uuid.UUID
    description: Optional[str] = None
    priority: FraudCasePriority = FraudCasePriority.MEDIUM
    transaction_id: Optional[uuid.UUID] = None
    alert_id: Optional[uuid.UUID] = None
    tags: Optional[List[str]] = None

    model_config = ConfigDict(extra="allow")


class FraudCaseUpdate(BaseModel):
    """Update fraud case."""

    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[FraudCasePriority] = None
    tags: Optional[List[str]] = None
    resolution: Optional[str] = None
    updated_by: Optional[uuid.UUID] = None

    model_config = ConfigDict(from_attributes=True)


class ActivityItem(BaseModel):
    """Case activity item."""

    id: uuid.UUID
    type: str
    title: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    user: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class FraudCaseResponse(BaseModel):
    """Fraud case response."""

    id: uuid.UUID
    case_number: str
    title: str
    description: Optional[str] = None
    organization_id: uuid.UUID
    created_by: uuid.UUID
    status: str
    priority: FraudCasePriority
    created_at: datetime
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None
    transaction_id: Optional[uuid.UUID] = None
    alert_id: Optional[uuid.UUID] = None
    tags: List[str] = Field(default_factory=list)
    assignee: Optional[Dict[str, Any]] = None
    assignee_id: Optional[str] = None
    activities: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(from_attributes=True)
