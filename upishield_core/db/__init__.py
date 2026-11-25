"""
Database models and session management for ScamShield AI / UPIShield.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Organization(Base):
    """Organization/tenant for multi-tenancy."""
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="organization")
    transactions = relationship("Transaction", back_populates="organization")
    models = relationship("ModelVersion", back_populates="organization")


class User(Base):
    """User within an organization."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="analyst")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    organization = relationship("Organization", back_populates="users")

    __table_args__ = (UniqueConstraint("organization_id", "email", name="uq_org_email"),)


class Transaction(Base):
    """UPI Transaction record for fraud detection."""
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    txn_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    payer_vpa: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    payee_vpa: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    txn_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    device_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    client_ip: Mapped[Optional[str]] = mapped_column(String(45), index=True)
    merchant_category: Mapped[Optional[str]] = mapped_column(String(50))
    merchant_name: Mapped[Optional[str]] = mapped_column(String(255))
    location: Mapped[Optional[str]] = mapped_column(String(100))
    is_fraud: Mapped[Optional[bool]] = mapped_column(Boolean, default=None, index=True)
    fraud_pattern: Mapped[Optional[str]] = mapped_column(String(50))
    risk_score: Mapped[Optional[float]] = mapped_column(Float)
    risk_level: Mapped[Optional[str]] = mapped_column(String(20))
    decision: Mapped[Optional[str]] = mapped_column(String(20))
    ml_score: Mapped[Optional[float]] = mapped_column(Float)
    rules_score: Mapped[Optional[float]] = mapped_column(Float)
    threat_intel_score: Mapped[Optional[float]] = mapped_column(Float)
    features: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = relationship("Organization", back_populates="transactions")

    __table_args__ = (
        Index("ix_txn_payer_time", "payer_vpa", "timestamp"),
        Index("ix_txn_payee_time", "payee_vpa", "timestamp"),
        Index("ix_txn_fraud_time", "is_fraud", "timestamp"),
    )


class ModelVersion(Base):
    """ML Model version tracking for A/B testing and rollback."""
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), default="onnx")
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="staging")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    organization = relationship("Organization", back_populates="models")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_version"),
        UniqueConstraint("organization_id", "name", "status", name="uq_org_model_status"),
    )


class ScamPattern(Base):
    """Known scam patterns for rule-based detection."""
    __tablename__ = "scam_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    pattern_type: Mapped[str] = mapped_column(String(50), nullable=False)
    pattern: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    category: Mapped[Optional[str]] = mapped_column(String(50))
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Alert(Base):
    """Fraud alerts for investigation."""
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    transaction_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("transactions.id"), index=True)
    txn_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="open")
    assigned_to: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))
    alert_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    __table_args__ = (
        Index("ix_alert_org_status", "organization_id", "status"),
        Index("ix_alert_org_time", "organization_id", "created_at"),
    )


# Database engine and session
_engine = None
_async_engine = None
_session_factory = None
_async_session_factory = None


def init_db(database_url: str = "sqlite:///./scamshield.db", echo: bool = False):
    """Initialize database connection."""
    global _engine, _session_factory, _async_engine, _async_session_factory

    # Synchronous engine for migrations
    _engine = create_engine(database_url, echo=echo)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)

    # Async engine for application
    async_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    _async_engine = create_async_engine(async_url, echo=echo)
    _async_session_factory = async_sessionmaker(bind=_async_engine, class_=AsyncSession, expire_on_commit=False)

    return _engine


def get_session():
    """Get synchronous session (for migrations)."""
    if _session_factory is None:
        init_db()
    return _session_factory()


async def get_async_session() -> AsyncSession:
    """Get async session (for application)."""
    if _async_session_factory is None:
        init_db()
    async with _async_session_factory() as session:
        yield session


def create_tables():
    """Create all tables."""
    if _engine is None:
        init_db()
    Base.metadata.create_all(_engine)


def drop_tables():
    """Drop all tables."""
    if _engine is None:
        init_db()
    Base.metadata.drop_all(_engine)


__all__ = [
    "Base",
    "Organization",
    "User",
    "Transaction",
    "ModelVersion",
    "ScamPattern",
    "Alert",
    "init_db",
    "get_session",
    "get_async_session",
    "create_tables",
    "drop_tables",
]