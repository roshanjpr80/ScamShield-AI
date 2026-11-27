"""
Feast Feature Store Integration for ScamShield AI.

Provides feature definitions, online/offline store management,
and feature retrieval for real-time fraud detection.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import json

try:
    from feast import (
        FeatureStore,
        Entity,
        FeatureView,
        Field,
        FileSource,
        PushSource,
        RequestSource,
        FeatureService,
        ValueType,
        RepoConfig,
    )
    from feast.types import Float32, Float64, Int64, Int32, String, Bool
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logging.warning("Feast not available - feature store will use mock implementation")

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Categories of features."""
    TRANSACTION = "transaction"
    USER = "user"
    MERCHANT = "merchant"
    DEVICE = "device"
    BEHAVIORAL = "behavioral"
    NETWORK = "network"
    TEMPORAL = "temporal"
    DOCUMENT = "document"


@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    name: str
    category: FeatureCategory
    dtype: str  # feast type as string
    description: str
    default_value: Any = None
    transformation: Optional[str] = None  # SQL or Python transformation
    tags: Dict[str, str] = field(default_factory=dict)


# Transaction Features
TRANSACTION_FEATURES = [
    FeatureDefinition("txn_amount", FeatureCategory.TRANSACTION, "Float64", "Transaction amount in INR"),
    FeatureDefinition("txn_amount_log", FeatureCategory.TRANSACTION, "Float64", "Log of transaction amount"),
    FeatureDefinition("txn_amount_zscore", FeatureCategory.TRANSACTION, "Float64", "Z-score of amount vs user history"),
    FeatureDefinition("txn_currency", FeatureCategory.TRANSACTION, "String", "Transaction currency"),
    FeatureDefinition("txn_type", FeatureCategory.TRANSACTION, "String", "Transaction type (P2P, P2M, QR, etc.)"),
    FeatureDefinition("txn_channel", FeatureCategory.TRANSACTION, "String", "Channel (UPI, IMPS, NEFT, etc.)"),
    FeatureDefinition("txn_hour", FeatureCategory.TEMPORAL, "Int32", "Hour of transaction (0-23)"),
    FeatureDefinition("txn_day_of_week", FeatureCategory.TEMPORAL, "Int32", "Day of week (0-6)"),
    FeatureDefinition("txn_is_weekend", FeatureCategory.TEMPORAL, "Bool", "Is weekend transaction"),
    FeatureDefinition("txn_is_night", FeatureCategory.TEMPORAL, "Bool", "Is night transaction (10PM-6AM)"),
    FeatureDefinition("sender_vpa", FeatureCategory.TRANSACTION, "String", "Sender VPA"),
    FeatureDefinition("receiver_vpa", FeatureCategory.TRANSACTION, "String", "Receiver VPA"),
    FeatureDefinition("vpa_domain_match", FeatureCategory.TRANSACTION, "Bool", "Sender and receiver VPA domain match"),
    FeatureDefinition("vpa_is_known_merchant", FeatureCategory.MERCHANT, "Bool", "Receiver is known merchant"),
]

# User Features
USER_FEATURES = [
    FeatureDefinition("user_id", FeatureCategory.USER, "String", "Unique user identifier"),
    FeatureDefinition("user_age_days", FeatureCategory.USER, "Int32", "Account age in days"),
    FeatureDefinition("user_kyc_level", FeatureCategory.USER, "Int32", "KYC level (0=none, 1=basic, 2=full)"),
    FeatureDefinition("user_risk_score", FeatureCategory.USER, "Float64", "User risk score (0-1)"),
    FeatureDefinition("user_txn_count_7d", FeatureCategory.BEHAVIORAL, "Int64", "Transaction count last 7 days"),
    FeatureDefinition("user_txn_count_30d", FeatureCategory.BEHAVIORAL, "Int64", "Transaction count last 30 days"),
    FeatureDefinition("user_txn_volume_7d", FeatureCategory.BEHAVIORAL, "Float64", "Transaction volume last 7 days"),
    FeatureDefinition("user_txn_volume_30d", FeatureCategory.BEHAVIORAL, "Float64", "Transaction volume last 30 days"),
    FeatureDefinition("user_avg_txn_amount", FeatureCategory.BEHAVIORAL, "Float64", "Average transaction amount"),
    FeatureDefinition("user_max_txn_amount", FeatureCategory.BEHAVIORAL, "Float64", "Maximum transaction amount"),
    FeatureDefinition("user_unique_merchants_7d", FeatureCategory.BEHAVIORAL, "Int32", "Unique merchants last 7 days"),
    FeatureDefinition("user_unique_vpas_7d", FeatureCategory.BEHAVIORAL, "Int32", "Unique VPAs last 7 days"),
    FeatureDefinition("user_fraud_count_30d", FeatureCategory.BEHAVIORAL, "Int32", "Fraud count last 30 days"),
    FeatureDefinition("user_chargeback_count_30d", FeatureCategory.BEHAVIORAL, "Int32", "Chargeback count last 30 days"),
    FeatureDefinition("user_device_count_30d", FeatureCategory.DEVICE, "Int32", "Unique devices last 30 days"),
    FeatureDefinition("user_ip_count_30d", FeatureCategory.NETWORK, "Int32", "Unique IPs last 30 days"),
]

# Merchant Features
MERCHANT_FEATURES = [
    FeatureDefinition("merchant_id", FeatureCategory.MERCHANT, "String", "Merchant identifier"),
    FeatureDefinition("merchant_category", FeatureCategory.MERCHANT, "String", "Merchant category code"),
    FeatureDefinition("merchant_risk_score", FeatureCategory.MERCHANT, "Float64", "Merchant risk score (0-1)"),
    FeatureDefinition("merchant_txn_count_7d", FeatureCategory.MERCHANT, "Int64", "Transaction count last 7 days"),
    FeatureDefinition("merchant_txn_volume_7d", FeatureCategory.MERCHANT, "Float64", "Volume last 7 days"),
    FeatureDefinition("merchant_fraud_rate_30d", FeatureCategory.MERCHANT, "Float64", "Fraud rate last 30 days"),
    FeatureDefinition("merchant_chargeback_rate_30d", FeatureCategory.MERCHANT, "Float64", "Chargeback rate last 30 days"),
    FeatureDefinition("merchant_avg_txn_amount", FeatureCategory.MERCHANT, "Float64", "Average transaction amount"),
    FeatureDefinition("merchant_unique_users_7d", FeatureCategory.MERCHANT, "Int32", "Unique users last 7 days"),
    FeatureDefinition("merchant_is_new", FeatureCategory.MERCHANT, "Bool", "Merchant onboarded < 30 days"),
]

# Device Features
DEVICE_FEATURES = [
    FeatureDefinition("device_id", FeatureCategory.DEVICE, "String", "Device fingerprint"),
    FeatureDefinition("device_type", FeatureCategory.DEVICE, "String", "Device type (mobile, desktop, tablet)"),
    FeatureDefinition("device_os", FeatureCategory.DEVICE, "String", "Operating system"),
    FeatureDefinition("device_is_rooted", FeatureCategory.DEVICE, "Bool", "Device is rooted/jailbroken"),
    FeatureDefinition("device_is_emulator", FeatureCategory.DEVICE, "Bool", "Running in emulator"),
    FeatureDefinition("device_fraud_count_30d", FeatureCategory.DEVICE, "Int32", "Fraud count on device last 30 days"),
    FeatureDefinition("device_user_count_30d", FeatureCategory.DEVICE, "Int32", "Unique users on device last 30 days"),
    FeatureDefinition("device_age_days", FeatureCategory.DEVICE, "Int32", "Days since first seen"),
]

# Network Features
NETWORK_FEATURES = [
    FeatureDefinition("ip_address", FeatureCategory.NETWORK, "String", "Client IP address"),
    FeatureDefinition("ip_country", FeatureCategory.NETWORK, "String", "IP country code"),
    FeatureDefinition("ip_asn", FeatureCategory.NETWORK, "Int32", "IP ASN number"),
    FeatureDefinition("ip_is_vpn", FeatureCategory.NETWORK, "Bool", "IP is VPN/proxy"),
    FeatureDefinition("ip_is_tor", FeatureCategory.NETWORK, "Bool", "IP is Tor exit node"),
    FeatureDefinition("ip_fraud_count_30d", FeatureCategory.NETWORK, "Int32", "Fraud count on IP last 30 days"),
    FeatureDefinition("ip_user_count_30d", FeatureCategory.NETWORK, "Int32", "Unique users on IP last 30 days"),
    FeatureDefinition("ip_distance_km", FeatureCategory.NETWORK, "Float64", "Distance from user's usual location"),
]

# Document Features (from CV pipeline)
DOCUMENT_FEATURES = [
    FeatureDefinition("doc_type", FeatureCategory.DOCUMENT, "String", "Document type (aadhaar, pan, etc.)"),
    FeatureDefinition("doc_ela_score", FeatureCategory.DOCUMENT, "Float64", "ELA anomaly score"),
    FeatureDefinition("doc_noise_score", FeatureCategory.DOCUMENT, "Float64", "Noise inconsistency score"),
    FeatureDefinition("doc_copymove_score", FeatureCategory.DOCUMENT, "Float64", "Copy-move forgery score"),
    FeatureDefinition("doc_color_score", FeatureCategory.DOCUMENT, "Float64", "Color inconsistency score"),
    FeatureDefinition("doc_edge_score", FeatureCategory.DOCUMENT, "Float64", "Edge inconsistency score"),
    FeatureDefinition("doc_ocr_confidence", FeatureCategory.DOCUMENT, "Float64", "OCR extraction confidence"),
    FeatureDefinition("doc_tampering_verdict", FeatureCategory.DOCUMENT, "String", "Tampering verdict"),
    FeatureDefinition("doc_risk_level", FeatureCategory.DOCUMENT, "String", "Risk level"),
]

ALL_FEATURES = (
    TRANSACTION_FEATURES + USER_FEATURES + MERCHANT_FEATURES +
    DEVICE_FEATURES + NETWORK_FEATURES + DOCUMENT_FEATURES
)


class FeastFeatureStore:
    """
    Feast Feature Store wrapper for ScamShield AI.

    Provides:
    - Feature registration and management
    - Online feature retrieval for real-time inference
    - Offline feature retrieval for training
    - Feature service definitions
    - Materialization jobs
    """

    def __init__(
        self,
        repo_path: str = "./feature_repo",
        online_store_type: str = "sqlite",
    ):
        self.repo_path = Path(repo_path)
        self.online_store_type = online_store_type
        self._store: Optional[FeatureStore] = None
        self._initialized = False

    def initialize(self):
        """Initialize Feast feature store."""
        if not FEAST_AVAILABLE:
            logger.warning("Feast not available, using mock store")
            self._initialized = True
            return

        # Create repo config
        config = RepoConfig(
            project="scamshield",
            provider="local",
            registry=str(self.repo_path / "registry.db"),
            online_store={
                "type": self.online_store_type,
                "path": str(self.repo_path / "online_store.db"),
            },
            offline_store={
                "type": "file",
            },
            entity_key_serialization_version=2,
        )

        self.repo_path.mkdir(parents=True, exist_ok=True)
        self._store = FeatureStore(config=config)
        self._initialized = True
        logger.info(f"Feast feature store initialized at {self.repo_path}")

    def apply(self):
        """Apply feature definitions to store."""
        if not FEAST_AVAILABLE or not self._initialized:
            logger.warning("Feast not available or not initialized")
            return

        # Define entities
        entities = [
            Entity(name="user_id", value_type=ValueType.STRING, description="User identifier"),
            Entity(name="merchant_id", value_type=ValueType.STRING, description="Merchant identifier"),
            Entity(name="device_id", value_type=ValueType.STRING, description="Device identifier"),
            Entity(name="transaction_id", value_type=ValueType.STRING, description="Transaction identifier"),
        ]

        # Create feature views
        feature_views = self._create_feature_views()

        # Apply to store
        self._store.apply(entities + feature_views)
        logger.info("Feature definitions applied to Feast store")

    def _create_feature_views(self) -> List[FeatureView]:
        """Create feature views from definitions."""
        if not FEAST_AVAILABLE:
            return []

        views = []

        # Transaction feature view
        txn_fields = [
            Field(name=f.name, dtype=self._get_feast_type(f.dtype))
            for f in TRANSACTION_FEATURES
        ]
        views.append(FeatureView(
            name="transaction_features",
            entities=["transaction_id"],
            ttl=timedelta(days=30),
            schema=txn_fields,
            online=True,
            source=FileSource(
                path=str(self.repo_path / "data" / "transactions.parquet"),
                timestamp_field="event_timestamp",
                created_timestamp_column="created_timestamp",
            ),
            tags={"category": "transaction"},
        ))

        # User feature view
        user_fields = [
            Field(name=f.name, dtype=self._get_feast_type(f.dtype))
            for f in USER_FEATURES
        ]
        views.append(FeatureView(
            name="user_features",
            entities=["user_id"],
            ttl=timedelta(days=90),
            schema=user_fields,
            online=True,
            source=FileSource(
                path=str(self.repo_path / "data" / "users.parquet"),
                timestamp_field="event_timestamp",
                created_timestamp_column="created_timestamp",
            ),
            tags={"category": "user"},
        ))

        # Merchant feature view
        merchant_fields = [
            Field(name=f.name, dtype=self._get_feast_type(f.dtype))
            for f in MERCHANT_FEATURES
        ]
        views.append(FeatureView(
            name="merchant_features",
            entities=["merchant_id"],
            ttl=timedelta(days=90),
            schema=merchant_fields,
            online=True,
            source=FileSource(
                path=str(self.repo_path / "data" / "merchants.parquet"),
                timestamp_field="event_timestamp",
                created_timestamp_column="created_timestamp",
            ),
            tags={"category": "merchant"},
        ))

        # Device feature view
        device_fields = [
            Field(name=f.name, dtype=self._get_feast_type(f.dtype))
            for f in DEVICE_FEATURES
        ]
        views.append(FeatureView(
            name="device_features",
            entities=["device_id"],
            ttl=timedelta(days=90),
            schema=device_fields,
            online=True,
            source=FileSource(
                path=str(self.repo_path / "data" / "devices.parquet"),
                timestamp_field="event_timestamp",
                created_timestamp_column="created_timestamp",
            ),
            tags={"category": "device"},
        ))

        # Document feature view
        doc_fields = [
            Field(name=f.name, dtype=self._get_feast_type(f.dtype))
            for f in DOCUMENT_FEATURES
        ]
        views.append(FeatureView(
            name="document_features",
            entities=["transaction_id"],
            ttl=timedelta(days=30),
            schema=doc_fields,
            online=True,
            source=FileSource(
                path=str(self.repo_path / "data" / "documents.parquet"),
                timestamp_field="event_timestamp",
                created_timestamp_column="created_timestamp",
            ),
            tags={"category": "document"},
        ))

        return views

    def _get_feast_type(self, dtype_str: str):
        """Convert dtype string to Feast type."""
        type_map = {
            "Float64": Float64,
            "Float32": Float32,
            "Int64": Int64,
            "Int32": Int32,
            "String": String,
            "Bool": Bool,
        }
        return type_map.get(dtype_str, String)

    def create_feature_services(self):
        """Create feature services for different use cases."""
        if not FEAST_AVAILABLE or not self._initialized:
            return

        # Fraud detection service - all features
        fraud_service = FeatureService(
            name="fraud_detection",
            features=[
                "transaction_features",
                "user_features",
                "merchant_features",
                "device_features",
                "document_features",
            ],
            tags={"use_case": "fraud_detection"},
        )

        # Training service - subset for model training
        training_service = FeatureService(
            name="model_training",
            features=[
                "transaction_features",
                "user_features",
                "merchant_features",
            ],
            tags={"use_case": "training"},
        )

        # Real-time scoring service - minimal features
        scoring_service = FeatureService(
            name="realtime_scoring",
            features=[
                "transaction_features:txn_amount",
                "transaction_features:txn_hour",
                "transaction_features:txn_is_night",
                "user_features:user_risk_score",
                "user_features:user_txn_count_7d",
                "merchant_features:merchant_risk_score",
            ],
            tags={"use_case": "realtime"},
        )

        self._store.apply([fraud_service, training_service, scoring_service])
        logger.info("Feature services created")

    def get_online_features(
        self,
        entity_rows: List[Dict[str, str]],
        features: Optional[List[str]] = None,
        feature_service: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get online features for real-time inference."""
        if not FEAST_AVAILABLE or not self._initialized:
            return self._mock_online_features(entity_rows)

        if feature_service:
            return self._store.get_online_features(
                entity_rows=entity_rows,
                features=[],
                feature_service=feature_service,
            ).to_dict()

        return self._store.get_online_features(
            entity_rows=entity_rows,
            features=features or [],
        ).to_dict()

    def get_historical_features(
        self,
        entity_df: Union[pd.DataFrame, str],
        features: Optional[List[str]] = None,
        feature_service: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get historical features for training."""
        if not FEAST_AVAILABLE or not self._initialized:
            return self._mock_historical_features(entity_df)

        if feature_service:
            return self._store.get_historical_features(
                entity_df=entity_df,
                features=[],
                feature_service=feature_service,
            ).to_df()

        return self._store.get_historical_features(
            entity_df=entity_df,
            features=features or [],
        ).to_df()

    def materialize(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_views: Optional[List[str]] = None,
    ):
        """Materialize features to online store."""
        if not FEAST_AVAILABLE or not self._initialized:
            logger.info("Mock materialization completed")
            return

        self._store.materialize(start_date, end_date, feature_views)
        logger.info(f"Materialized features from {start_date} to {end_date}")

    def materialize_incremental(self, end_date: datetime, feature_views: Optional[List[str]] = None):
        """Incremental materialization."""
        if not FEAST_AVAILABLE or not self._initialized:
            return

        self._store.materialize_incremental(end_date, feature_views)
        logger.info(f"Incremental materialization to {end_date}")

    def push_features(
        self,
        push_source_name: str,
        df: pd.DataFrame,
        to_online: bool = True,
        to_offline: bool = True,
    ):
        """Push features to Feast."""
        if not FEAST_AVAILABLE or not self._initialized:
            logger.info(f"Mock push to {push_source_name}: {len(df)} rows")
            return

        self._store.push(push_source_name, df, to_online=to_online, to_offline=to_offline)

    def list_feature_views(self) -> List[str]:
        """List all feature views."""
        if not FEAST_AVAILABLE or not self._initialized:
            return [fv.name for fv in self._create_feature_views()]

        return [fv.name for fv in self._store.list_feature_views()]

    def list_feature_services(self) -> List[str]:
        """List all feature services."""
        if not FEAST_AVAILABLE or not self._initialized:
            return ["fraud_detection", "model_training", "realtime_scoring"]

        return [fs.name for fs in self._store.list_feature_services()]

    def _mock_online_features(self, entity_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Mock online feature retrieval."""
        results = []
        for row in entity_rows:
            result = dict(row)
            # Add mock feature values
            result.update({
                "txn_amount": 1000.0,
                "user_risk_score": 0.1,
                "merchant_risk_score": 0.05,
                "doc_ela_score": 0.02,
            })
            results.append(result)
        return results

    def _mock_historical_features(self, entity_df: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """Mock historical feature retrieval."""
        if isinstance(entity_df, str):
            return pd.DataFrame()
        df = entity_df.copy()
        # Add mock features
        for f in ALL_FEATURES:
            if f.dtype in ["Float64", "Float32"]:
                df[f.name] = 0.0
            elif f.dtype in ["Int64", "Int32"]:
                df[f.name] = 0
            elif f.dtype == "Bool":
                df[f.name] = False
            else:
                df[f.name] = ""
        return df

    def export_feature_definitions(self, output_path: str):
        """Export feature definitions to JSON."""
        definitions = []
        for f in ALL_FEATURES:
            definitions.append({
                "name": f.name,
                "category": f.category.value,
                "dtype": f.dtype,
                "description": f.description,
                "default_value": f.default_value,
                "transformation": f.transformation,
                "tags": f.tags,
            })
        with open(output_path, 'w') as f:
            json.dump(definitions, f, indent=2)
        logger.info(f"Exported {len(definitions)} feature definitions to {output_path}")


# Global store instance
_default_store: Optional[FeastFeatureStore] = None


def get_feature_store(
    repo_path: str = "./feature_repo",
    online_store_type: str = "sqlite",
) -> FeastFeatureStore:
    """Get or create default feature store."""
    global _default_store
    if _default_store is None:
        _default_store = FeastFeatureStore(repo_path, online_store_type)
        _default_store.initialize()
    return _default_store