"""Test configuration for upishield_core."""

import asyncio

# Ensure testing mode
import os
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

os.environ.setdefault("TESTING", "true")


@pytest_asyncio.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_ensemble():
    """Create a mock ONNX ensemble for testing."""
    from upishield_core.models.ensemble import EnsembleInference

    mock = MagicMock(spec=EnsembleInference)
    mock._loaded = True

    # Mock predict_with_uncertainty to return fixed values
    mock.predict_with_uncertainty = MagicMock(
        return_value=(
            0.75,  # weighted_score
            0.85,  # confidence
            {"xgb": 0.8, "lgbm": 0.7, "transformer": 0.75, "gnn": 0.6},  # model_scores
        )
    )

    return mock


@pytest.fixture
def sample_transaction():
    """Create a sample UPI transaction for testing."""
    from datetime import datetime

    from upishield_core.api.schemas import UPITransaction

    return UPITransaction(
        txn_id="TEST_TXN_001",
        payer_vpa="user@upi",
        payee_vpa="merchant@upi",
        amount=5000.0,
        currency="INR",
        txn_type="P2M",
        timestamp=datetime.utcnow(),
        device_id="device_123",
        client_ip="192.168.1.1",
    )


@pytest.fixture
def sample_features():
    """Create sample UPI features for testing."""
    from datetime import datetime

    from upishield_core.features.schemas import (
        DeviceNetworkFeatures,
        GraphFeatures,
        MerchantFeatures,
        UPIFeatures,
        UserBehavioralFeatures,
        VelocityFeatures,
    )

    return UPIFeatures(
        txn_id="TEST_TXN_001",
        payer_vpa="user@upi",
        payee_vpa="merchant@upi",
        amount=5000.0,
        currency="INR",
        txn_type="P2M",
        timestamp=datetime.utcnow(),
        user_behavioral=UserBehavioralFeatures(
            user_id="user@upi",
            avg_txn_amount_7d=3000.0,
            txn_count_1h=2,
            txn_count_24h=10,
            unique_merchants_7d=5,
            favorite_vpa_similarity=0.8,
            days_since_first_txn=100,
            chargeback_rate_30d=0.01,
        ),
        device_network=DeviceNetworkFeatures(
            device_id="device_123",
            device_trust_score=0.8,
            ip_reputation_score=0.9,
            ip_geolocation_mismatch=False,
            is_vpn_proxy=False,
            device_change_recent=False,
            rooted_jailbroken=False,
        ),
        merchant=MerchantFeatures(
            merchant_vpa="merchant@upi",
            merchant_category="retail",
            merchant_risk_score=0.2,
            merchant_age_days=365,
            merchant_txn_volume_30d=1000,
            merchant_chargeback_rate=0.01,
            is_verified_merchant=True,
        ),
        velocity=VelocityFeatures(
            vpa_velocity_5m=1,
            vpa_velocity_1h=5,
            amount_velocity_1h=10000.0,
            unique_merchants_1h=2,
        ),
        graph=GraphFeatures(
            user_merchant_graph_distance=3,
            vpa_cluster_risk=0.3,
            pagerank_score=0.5,
            is_bridge_node=False,
        ),
    )


@pytest.fixture
def mock_feature_computer():
    """Mock FeatureComputer for testing."""
    with patch("upishield_core.decision.engine.FeatureComputer") as mock:
        mock.compute_all = MagicMock()
        yield mock


# Common test utilities
def assert_valid_score(score: float):
    """Assert that a score is valid (0-1 range)."""
    assert 0.0 <= score <= 1.0


def assert_valid_decision(decision):
    """Assert that a decision is valid."""
    from upishield_core.api.schemas import FraudDecision

    assert decision in [FraudDecision.APPROVE, FraudDecision.REVIEW, FraudDecision.DECLINE]


def assert_valid_risk_level(level):
    """Assert that a risk level is valid."""
    from upishield_core.api.schemas import RiskLevel

    assert level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
