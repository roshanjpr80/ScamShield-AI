"""
Feature Materialization Pipeline for ScamShield AI.

Materializes features from offline store (Parquet/Feast) to online store (Redis/SQLite/Memory)
for real-time feature serving in the Decision Engine.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from upishield_core.features.feast_store import FeastFeatureStore, get_feature_store
from upishield_core.features.online_store import OnlineFeatureStore, OnlineStoreConfig, StoreBackend, FeatureVector

logger = logging.getLogger(__name__)


class FeatureMaterializer:
    """
    Materializes features from offline store to online store.

    Supports:
    - Full materialization (historical backfill)
    - Incremental materialization (daily/hourly updates)
    - Push-based real-time updates
    - Feature freshness monitoring
    """

    def __init__(
        self,
        feast_store: Optional[FeastFeatureStore] = None,
        online_store: Optional[OnlineFeatureStore] = None,
    ):
        self.feast_store = feast_store or get_feature_store()
        self.online_store = online_store or OnlineFeatureStore()
        self._stats = {
            "total_materialized": 0,
            "last_materialization": None,
            "errors": 0,
        }

    def materialize_user_features(
        self,
        user_df: pd.DataFrame,
        entity_column: str = "user_id",
        ttl_seconds: int = 86400 * 90,  # 90 days
    ) -> int:
        """
        Materialize user features to online store.

        Args:
            user_df: DataFrame with user features
            entity_column: Column name for user ID
            ttl_seconds: Time-to-live for features

        Returns:
            Number of features materialized
        """
        count = 0
        for _, row in user_df.iterrows():
            entity_id = str(row[entity_column])
            features = row.drop(entity_column).to_dict()

            # Convert numpy types to Python types
            features = self._convert_types(features)

            vector = FeatureVector(
                entity_id=entity_id,
                features=features,
                timestamp=datetime.now(),
                ttl_seconds=ttl_seconds,
            )

            if self.online_store.write_features(entity_id, features, ttl_seconds):
                count += 1

        self._stats["total_materialized"] += count
        self._stats["last_materialization"] = datetime.now()
        logger.info(f"Materialized {count} user feature vectors")
        return count

    def materialize_merchant_features(
        self,
        merchant_df: pd.DataFrame,
        entity_column: str = "merchant_id",
        ttl_seconds: int = 86400 * 90,
    ) -> int:
        """Materialize merchant features to online store."""
        count = 0
        for _, row in merchant_df.iterrows():
            entity_id = str(row[entity_column])
            features = row.drop(entity_column).to_dict()
            features = self._convert_types(features)

            if self.online_store.write_features(entity_id, features, ttl_seconds):
                count += 1

        self._stats["total_materialized"] += count
        logger.info(f"Materialized {count} merchant feature vectors")
        return count

    def materialize_device_features(
        self,
        device_df: pd.DataFrame,
        entity_column: str = "device_id",
        ttl_seconds: int = 86400 * 90,
    ) -> int:
        """Materialize device features to online store."""
        count = 0
        for _, row in device_df.iterrows():
            entity_id = str(row[entity_column])
            features = row.drop(entity_column).to_dict()
            features = self._convert_types(features)

            if self.online_store.write_features(entity_id, features, ttl_seconds):
                count += 1

        self._stats["total_materialized"] += count
        logger.info(f"Materialized {count} device feature vectors")
        return count

    def materialize_all(
        self,
        data_dir: str = "./feature_repo/data",
        ttl_seconds: int = 86400 * 90,
    ) -> Dict[str, int]:
        """
        Materialize all feature views from Parquet files.

        Args:
            data_dir: Directory containing Parquet files
            ttl_seconds: Default TTL for all features

        Returns:
            Dict with counts per feature type
        """
        data_path = Path(data_dir)
        results = {}

        # Define file mappings
        file_mappings = [
            ("users.parquet", "user_id", "user"),
            ("merchants.parquet", "merchant_id", "merchant"),
            ("devices.parquet", "device_id", "device"),
            ("transactions.parquet", "transaction_id", "transaction"),
        ]

        for filename, entity_col, feature_type in file_mappings:
            filepath = data_path / filename
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath)
                    if feature_type == "user":
                        count = self.materialize_user_features(df, entity_col, ttl_seconds)
                    elif feature_type == "merchant":
                        count = self.materialize_merchant_features(df, entity_col, ttl_seconds)
                    elif feature_type == "device":
                        count = self.materialize_device_features(df, entity_col, ttl_seconds)
                    else:
                        # Transaction features - use generic method
                        count = self._materialize_generic(df, entity_col, ttl_seconds)
                    results[feature_type] = count
                except Exception as e:
                    logger.error(f"Failed to materialize {filename}: {e}")
                    self._stats["errors"] += 1
                    results[feature_type] = 0
            else:
                logger.warning(f"File not found: {filepath}")
                results[feature_type] = 0

        logger.info(f"Full materialization complete: {results}")
        return results

    def _materialize_generic(
        self,
        df: pd.DataFrame,
        entity_column: str,
        ttl_seconds: int,
    ) -> int:
        """Generic materialization for any entity type."""
        count = 0
        for _, row in df.iterrows():
            entity_id = str(row[entity_column])
            features = row.drop(entity_column).to_dict()
            features = self._convert_types(features)

            if self.online_store.write_features(entity_id, features, ttl_seconds):
                count += 1
        return count

    def _convert_types(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy/pandas types to Python native types for serialization."""
        converted = {}
        for key, value in features.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                converted[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                converted[key] = float(value)
            elif isinstance(value, np.bool_):
                converted[key] = bool(value)
            elif isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            elif pd.isna(value):
                converted[key] = None
            else:
                converted[key] = value
        return converted

    def push_real_time_features(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, Any],
        ttl_seconds: int = 86400,
    ) -> bool:
        """
        Push real-time feature updates to online store.

        Use this for streaming updates from Kafka/Feast push source.

        Args:
            entity_type: Type of entity (user, merchant, device, transaction)
            entity_id: Entity identifier
            features: Feature dictionary
            ttl_seconds: TTL for the features

        Returns:
            True if successful
        """
        # Create compound key with entity type
        compound_id = f"{entity_type}:{entity_id}"

        features = self._convert_types(features)

        success = self.online_store.write_features(
            compound_id,
            features,
            ttl_seconds,
        )

        if success:
            logger.debug(f"Pushed real-time features for {compound_id}")
        else:
            logger.error(f"Failed to push real-time features for {compound_id}")
            self._stats["errors"] += 1

        return success

    def get_materialization_stats(self) -> Dict[str, Any]:
        """Get materialization statistics."""
        return {
            **self._stats,
            "online_store_stats": self.online_store.get_stats(),
            "online_store_health": self.online_store.health_check(),
        }

    def verify_materialization(
        self,
        entity_type: str,
        entity_id: str,
        expected_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Verify features were materialized correctly.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            expected_features: List of expected feature names

        Returns:
            Verification result
        """
        compound_id = f"{entity_type}:{entity_id}"
        features = self.online_store.read_features(compound_id)

        if features is None:
            return {
                "verified": False,
                "reason": "Features not found in online store",
                "entity_id": compound_id,
            }

        missing_features = []
        if expected_features:
            missing_features = [f for f in expected_features if f not in features]

        return {
            "verified": len(missing_features) == 0,
            "entity_id": compound_id,
            "feature_count": len(features),
            "missing_features": missing_features,
            "sample_features": dict(list(features.items())[:10]),
            "freshness": self.online_store.get_freshness(compound_id),
        }


def create_sample_data(data_dir: str = "./feature_repo/data"):
    """Create sample feature data for testing."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Sample users
    n_users = 1000
    users = pd.DataFrame({
        "user_id": [f"user_{i:06d}" for i in range(n_users)],
        "avg_txn_amount_7d": np.random.lognormal(8, 1, n_users),
        "avg_txn_amount_30d": np.random.lognormal(8, 1, n_users),
        "txn_count_1h": np.random.poisson(2, n_users),
        "txn_count_24h": np.random.poisson(20, n_users),
        "txn_count_7d": np.random.poisson(50, n_users),
        "unique_merchants_7d": np.random.poisson(10, n_users),
        "unique_merchants_30d": np.random.poisson(30, n_users),
        "favorite_vpa_similarity": np.random.beta(5, 2, n_users),
        "days_since_first_txn": np.random.randint(1, 3650, n_users),
        "chargeback_rate_30d": np.random.beta(1, 100, n_users),
        "dispute_rate_30d": np.random.beta(1, 50, n_users),
        "user_risk_score": np.random.beta(2, 8, n_users),
    })
    users.to_parquet(data_path / "users.parquet", index=False)

    # Sample merchants
    n_merchants = 500
    categories = ["ecommerce", "food_delivery", "ride_hailing", "bill_payment", "recharge",
                  "peer_transfer", "investment", "gaming", "crypto", "gambling"]
    merchants = pd.DataFrame({
        "merchant_id": [f"merch_{i:06d}" for i in range(n_merchants)],
        "merchant_category": np.random.choice(categories, n_merchants),
        "merchant_risk_score": np.random.beta(2, 5, n_merchants),
        "merchant_age_days": np.random.randint(1, 1825, n_merchants),
        "merchant_txn_volume_30d": np.random.lognormal(10, 2, n_merchants).astype(int),
        "merchant_unique_users_30d": np.random.poisson(100, n_merchants),
        "merchant_chargeback_rate": np.random.beta(1, 50, n_merchants),
        "merchant_dispute_rate": np.random.beta(1, 30, n_merchants),
        "is_verified_merchant": np.random.choice([True, False], n_merchants, p=[0.3, 0.7]),
    })
    merchants.to_parquet(data_path / "merchants.parquet", index=False)

    # Sample devices
    n_devices = 2000
    device_types = ["android", "ios", "web"]
    os_versions = ["Android 14", "Android 13", "iOS 17", "iOS 16", "Windows 11", "macOS 14"]
    devices = pd.DataFrame({
        "device_id": [f"dev_{i:08x}" for i in range(n_devices)],
        "device_type": np.random.choice(device_types, n_devices),
        "device_trust_score": np.random.beta(5, 2, n_devices),
        "os_version": np.random.choice(os_versions, n_devices),
        "ip_address": [f"192.168.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}" for _ in range(n_devices)],
        "ip_reputation_score": np.random.beta(3, 3, n_devices),
        "ip_geolocation_mismatch": np.random.choice([True, False], n_devices, p=[0.1, 0.9]),
        "is_vpn_proxy": np.random.choice([True, False], n_devices, p=[0.05, 0.95]),
        "is_tor": np.random.choice([True, False], n_devices, p=[0.01, 0.99]),
        "sim_change_recent": np.random.choice([True, False], n_devices, p=[0.05, 0.95]),
        "device_change_recent": np.random.choice([True, False], n_devices, p=[0.1, 0.9]),
        "rooted_jailbroken": np.random.choice([True, False], n_devices, p=[0.02, 0.98]),
    })
    devices.to_parquet(data_path / "devices.parquet", index=False)

    # Sample transactions
    n_transactions = 5000
    transactions = pd.DataFrame({
        "transaction_id": [f"txn_{i:010d}" for i in range(n_transactions)],
        "user_id": np.random.choice(users["user_id"], n_transactions),
        "merchant_id": np.random.choice(merchants["merchant_id"], n_transactions),
        "device_id": np.random.choice(devices["device_id"], n_transactions),
        "amount": np.random.lognormal(8, 1.5, n_transactions),
        "txn_hour": np.random.randint(0, 24, n_transactions),
        "txn_day_of_week": np.random.randint(0, 7, n_transactions),
        "is_weekend": np.random.choice([True, False], n_transactions, p=[2/7, 5/7]),
        "is_night": np.random.choice([True, False], n_transactions, p=[0.3, 0.7]),
        "velocity_5m": np.random.poisson(2, n_transactions),
        "velocity_1h": np.random.poisson(10, n_transactions),
    })
    transactions.to_parquet(data_path / "transactions.parquet", index=False)

    logger.info(f"Created sample data in {data_dir}")
    return data_path


def run_materialization_pipeline(
    data_dir: str = "./feature_repo/data",
    online_store_backend: str = "memory",
    ttl_seconds: int = 86400 * 90,
) -> Dict[str, Any]:
    """
    Run complete materialization pipeline.

    Args:
        data_dir: Directory with Parquet files
        online_store_backend: Backend type (memory, sqlite, redis)
        ttl_seconds: Feature TTL

    Returns:
        Materialization results and stats
    """
    # Initialize stores
    feast_store = get_feature_store()
    feast_store.initialize()
    feast_store.apply()
    feast_store.create_feature_services()

    online_config = OnlineStoreConfig(
        backend=StoreBackend(online_store_backend),
        default_ttl_seconds=ttl_seconds,
    )
    online_store = OnlineFeatureStore(online_config)

    # Create materializer
    materializer = FeatureMaterializer(feast_store, online_store)

    # Create sample data if not exists
    data_path = Path(data_dir)
    if not (data_path / "users.parquet").exists():
        create_sample_data(data_dir)

    # Run materialization
    results = materializer.materialize_all(data_dir, ttl_seconds)

    # Verify a sample
    verification = materializer.verify_materialization(
        "user",
        "user_000000",
        ["user_risk_score", "avg_txn_amount_7d", "txn_count_7d"],
    )

    stats = materializer.get_materialization_stats()

    return {
        "materialization_results": results,
        "verification": verification,
        "stats": stats,
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Running feature materialization pipeline...")
    result = run_materialization_pipeline()

    print("\n=== Materialization Results ===")
    for k, v in result["materialization_results"].items():
        print(f"  {k}: {v} vectors")

    print("\n=== Verification ===")
    print(f"  Verified: {result['verification']['verified']}")
    print(f"  Feature count: {result['verification']['feature_count']}")

    print("\n=== Stats ===")
    print(f"  Total materialized: {result['stats']['total_materialized']}")
    print(f"  Online store health: {result['stats']['online_store_health']['healthy']}")
    print(f"  Cache hit rate: {result['stats']['online_store_stats'].get('cache_hit_rate', 0):.2%}")