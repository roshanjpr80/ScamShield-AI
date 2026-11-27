"""
Redis-backed Feature Store for Real-time UPI Fraud Detection.

Optimized for <5ms p99 feature retrieval using:
- Redis MGET pipeline (single round-trip)
- L1 in-process cache (hot features)
- Connection pooling
- Async operations
"""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from redis.asyncio import ConnectionPool, Redis

from upishield_core.api.schemas import UPITransaction
from upishield_core.features.schemas import (
    DeviceNetworkFeatures,
    GraphFeatures,
    MerchantFeatures,
    UPIFeatures,
    UserBehavioralFeatures,
    VelocityFeatures,
)


@dataclass
class FeatureStoreConfig:
    """Configuration for FeatureStore."""

    redis_url: str = "redis://localhost:6379"
    max_connections: int = 200
    socket_timeout: float = 2.0
    socket_connect_timeout: float = 1.0
    l1_cache_size: int = 10000
    l1_ttl_seconds: int = 300
    key_prefix: str = "upishield:features"
    velocity_ttl_1m: int = 60
    velocity_ttl_5m: int = 300
    velocity_ttl_1h: int = 3600
    velocity_ttl_24h: int = 86400


class LRUCache:
    """Thread-safe LRU cache for L1 feature caching."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()


class FeatureStore:
    """
    High-performance feature store for UPI fraud detection.

    Features:
    - Sub-5ms p99 latency for 50+ features
    - Redis Cluster support
    - L1 in-process cache for hot keys
    - Automatic velocity counter management
    - Batch feature computation support
    """

    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        self.config = config or FeatureStoreConfig()
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[Redis] = None
        self._l1_cache = LRUCache(self.config.l1_cache_size)
        self._connected = False

    async def connect(self) -> None:
        """Initialize Redis connection pool."""
        if self._connected:
            return

        self._pool = ConnectionPool.from_url(
            self.config.redis_url,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            decode_responses=True,
        )
        self._redis = Redis(connection_pool=self._pool)

        # Test connection
        await self._redis.ping()
        self._connected = True

    async def close(self) -> None:
        """Close Redis connections."""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
        self._connected = False

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
# Feature Retrieval (Hot Path - Optimized)
 
    async def get_features(self, transaction: UPITransaction) -> UPIFeatures:
        """
        Fetch all features for a transaction in parallel.

        Target: <5ms p99
        """
        if not self._connected:
            await self.connect()

        # Build cache key for L1
        cache_key = self._build_cache_key(transaction)

        # L1 cache check
        cached = await self._l1_cache.get(cache_key)
        if cached:
            return cached

        # Build all Redis keys
        keys = self._build_all_keys(transaction)

        # Single round-trip MGET
        pipe = self._redis.pipeline()
        for key in keys:
            pipe.get(key)
        values = await pipe.execute()

        # Parse into feature objects
        features = self._parse_features(transaction, keys, values)

        # Update L1 cache
        await self._l1_cache.set(cache_key, features)

        return features

    def _build_cache_key(self, transaction: UPITransaction) -> str:
        """Build L1 cache key from transaction."""
        # Include amount bucket to reduce cache misses on similar amounts
        amount_bucket = int(transaction.amount / 1000) * 1000
        key_str = f"{transaction.payer_vpa}:{transaction.payee_vpa}:{amount_bucket}"
        return f"{self.config.key_prefix}:cache:{hashlib.md5(key_str.encode()).hexdigest()[:16]}"

    def _build_all_keys(self, transaction: UPITransaction) -> List[str]:
        """Build all Redis keys needed for feature retrieval."""
        prefix = self.config.key_prefix
        payer = transaction.payer_vpa
        payee = transaction.payee_vpa
        device = transaction.device_id
        ip = transaction.client_ip

        keys = [
            # User behavioral
            f"{prefix}:user:{payer}:avg_amount_7d",
            f"{prefix}:user:{payer}:avg_amount_30d",
            f"{prefix}:user:{payer}:txn_count_1h",
            f"{prefix}:user:{payer}:txn_count_24h",
            f"{prefix}:user:{payer}:txn_count_7d",
            f"{prefix}:user:{payer}:unique_merchants_7d",
            f"{prefix}:user:{payer}:unique_merchants_30d",
            f"{prefix}:user:{payer}:fav_vpa_sim",
            f"{prefix}:user:{payer}:typical_hours",
            f"{prefix}:user:{payer}:chargeback_rate_30d",
            f"{prefix}:user:{payer}:dispute_rate_30d",
            f"{prefix}:user:{payer}:days_since_first",
            # Device/Network
            f"{prefix}:device:{device}:trust_score",
            f"{prefix}:device:{device}:type",
            f"{prefix}:device:{device}:os_version",
            f"{prefix}:device:{device}:app_version",
            f"{prefix}:ip:{ip}:reputation",
            f"{prefix}:ip:{ip}:geo_mismatch",
            f"{prefix}:ip:{ip}:asn",
            f"{prefix}:ip:{ip}:is_vpn",
            f"{prefix}:ip:{ip}:is_tor",
            f"{prefix}:user:{payer}:sim_change_recent",
            f"{prefix}:user:{payer}:device_change_recent",
            f"{prefix}:device:{device}:rooted",
            # Merchant
            f"{prefix}:merchant:{payee}:category",
            f"{prefix}:merchant:{payee}:risk_score",
            f"{prefix}:merchant:{payee}:age_days",
            f"{prefix}:merchant:{payee}:volume_30d",
            f"{prefix}:merchant:{payee}:unique_users_30d",
            f"{prefix}:merchant:{payee}:chargeback_rate",
            f"{prefix}:merchant:{payee}:dispute_rate",
            f"{prefix}:merchant:{payee}:refund_rate",
            f"{prefix}:merchant:{payee}:verified",
            f"{prefix}:merchant:{payee}:kyc_status",
            # Velocity
            f"{prefix}:velocity:{payer}:1m",
            f"{prefix}:velocity:{payer}:5m",
            f"{prefix}:velocity:{payer}:15m",
            f"{prefix}:velocity:{payer}:1h",
            f"{prefix}:velocity:{payer}:24h",
            f"{prefix}:velocity:{payer}:amount_1m",
            f"{prefix}:velocity:{payer}:amount_5m",
            f"{prefix}:velocity:{payer}:amount_1h",
            f"{prefix}:velocity:{payer}:amount_24h",
            f"{prefix}:velocity:{payer}:unique_merchants_1h",
            f"{prefix}:velocity:{payer}:unique_merchants_24h",
            f"{prefix}:velocity:{payer}:failed_1h",
            f"{prefix}:velocity:{payer}:failed_rate_1h",
            # Graph
            f"{prefix}:graph:{payer}:{payee}:distance",
            f"{prefix}:graph:{payer}:{payee}:common_neighbors",
            f"{prefix}:graph:{payer}:cluster_id",
            f"{prefix}:graph:{payer}:cluster_risk",
            f"{prefix}:graph:{payer}:cluster_size",
            f"{prefix}:graph:{payer}:pagerank",
            f"{prefix}:graph:{payer}:community_id",
            f"{prefix}:graph:{payer}:is_bridge",
        ]

        return keys

    def _parse_features(
        self, transaction: UPITransaction, keys: List[str], values: List[Optional[str]]
    ) -> UPIFeatures:
        """Parse Redis values into typed feature objects."""
        kv = dict(zip(keys, values))
        prefix = self.config.key_prefix

        def get(key: str, default: Any = None) -> Any:
            full_key = f"{prefix}:{key}" if not key.startswith(prefix) else key
            val = kv.get(full_key)
            if val is None:
                return default
            # Try to parse as JSON for complex types
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val

        def get_float(key: str, default: float = 0.0) -> float:
            val = get(key)
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def get_int(key: str, default: int = 0) -> int:
            val = get(key)
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        def get_bool(key: str, default: bool = False) -> bool:
            val = get(key)
            if val is None:
                return default
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val)

        # Build feature objects
        user_behavioral = UserBehavioralFeatures(
            user_id=transaction.payer_vpa,
            avg_txn_amount_7d=get_float(f"user:{transaction.payer_vpa}:avg_amount_7d"),
            avg_txn_amount_30d=get_float(f"user:{transaction.payer_vpa}:avg_amount_30d"),
            txn_count_1h=get_int(f"user:{transaction.payer_vpa}:txn_count_1h"),
            txn_count_24h=get_int(f"user:{transaction.payer_vpa}:txn_count_24h"),
            txn_count_7d=get_int(f"user:{transaction.payer_vpa}:txn_count_7d"),
            unique_merchants_7d=get_int(f"user:{transaction.payer_vpa}:unique_merchants_7d"),
            unique_merchants_30d=get_int(f"user:{transaction.payer_vpa}:unique_merchants_30d"),
            favorite_vpa_similarity=get_float(f"user:{transaction.payer_vpa}:fav_vpa_sim"),
            typical_txn_hours=get(f"user:{transaction.payer_vpa}:typical_hours", []),
            chargeback_rate_30d=get_float(f"user:{transaction.payer_vpa}:chargeback_rate_30d"),
            dispute_rate_30d=get_float(f"user:{transaction.payer_vpa}:dispute_rate_30d"),
            days_since_first_txn=get_int(f"user:{transaction.payer_vpa}:days_since_first"),
        )

        device_network = DeviceNetworkFeatures(
            device_id=transaction.device_id,
            device_trust_score=get_float(f"device:{transaction.device_id}:trust_score"),
            device_type=get(f"device:{transaction.device_id}:type", "unknown"),
            os_version=get(f"device:{transaction.device_id}:os_version", "unknown"),
            app_version=get(f"device:{transaction.device_id}:app_version", "unknown"),
            ip_address=transaction.client_ip,
            ip_reputation_score=get_float(f"ip:{transaction.client_ip}:reputation"),
            ip_geolocation_mismatch=get_bool(f"ip:{transaction.client_ip}:geo_mismatch"),
            ip_asn=get(f"ip:{transaction.client_ip}:asn", ""),
            is_vpn_proxy=get_bool(f"ip:{transaction.client_ip}:is_vpn"),
            is_tor=get_bool(f"ip:{transaction.client_ip}:is_tor"),
            sim_change_recent=get_bool(f"user:{transaction.payer_vpa}:sim_change_recent"),
            device_change_recent=get_bool(f"user:{transaction.payer_vpa}:device_change_recent"),
            rooted_jailbroken=get_bool(f"device:{transaction.device_id}:rooted"),
        )

        merchant = MerchantFeatures(
            merchant_vpa=transaction.payee_vpa,
            merchant_category=get(f"merchant:{transaction.payee_vpa}:category", "unknown"),
            merchant_risk_score=get_float(f"merchant:{transaction.payee_vpa}:risk_score"),
            merchant_age_days=get_int(f"merchant:{transaction.payee_vpa}:age_days"),
            merchant_txn_volume_30d=get_int(f"merchant:{transaction.payee_vpa}:volume_30d"),
            merchant_unique_users_30d=get_int(f"merchant:{transaction.payee_vpa}:unique_users_30d"),
            merchant_chargeback_rate=get_float(f"merchant:{transaction.payee_vpa}:chargeback_rate"),
            merchant_dispute_rate=get_float(f"merchant:{transaction.payee_vpa}:dispute_rate"),
            merchant_refund_rate=get_float(f"merchant:{transaction.payee_vpa}:refund_rate"),
            is_verified_merchant=get_bool(f"merchant:{transaction.payee_vpa}:verified"),
            kyc_status=get(f"merchant:{transaction.payee_vpa}:kyc_status", "unknown"),
        )

        velocity = VelocityFeatures(
            vpa_velocity_1m=get_int(f"velocity:{transaction.payer_vpa}:1m"),
            vpa_velocity_5m=get_int(f"velocity:{transaction.payer_vpa}:5m"),
            vpa_velocity_15m=get_int(f"velocity:{transaction.payer_vpa}:15m"),
            vpa_velocity_1h=get_int(f"velocity:{transaction.payer_vpa}:1h"),
            vpa_velocity_24h=get_int(f"velocity:{transaction.payer_vpa}:24h"),
            amount_velocity_1m=get_float(f"velocity:{transaction.payer_vpa}:amount_1m"),
            amount_velocity_5m=get_float(f"velocity:{transaction.payer_vpa}:amount_5m"),
            amount_velocity_1h=get_float(f"velocity:{transaction.payer_vpa}:amount_1h"),
            amount_velocity_24h=get_float(f"velocity:{transaction.payer_vpa}:amount_24h"),
            unique_merchants_1h=get_int(f"velocity:{transaction.payer_vpa}:unique_merchants_1h"),
            unique_merchants_24h=get_int(f"velocity:{transaction.payer_vpa}:unique_merchants_24h"),
            failed_txn_count_1h=get_int(f"velocity:{transaction.payer_vpa}:failed_1h"),
            failed_txn_rate_1h=get_float(f"velocity:{transaction.payer_vpa}:failed_rate_1h"),
        )

        graph = GraphFeatures(
            user_merchant_graph_distance=get_int(
                f"graph:{transaction.payer_vpa}:{transaction.payee_vpa}:distance", -1
            ),
            user_merchant_common_neighbors=get_int(
                f"graph:{transaction.payer_vpa}:{transaction.payee_vpa}:common_neighbors"
            ),
            vpa_cluster_id=get_int(f"graph:{transaction.payer_vpa}:cluster_id", -1),
            vpa_cluster_risk=get_float(f"graph:{transaction.payer_vpa}:cluster_risk"),
            vpa_cluster_size=get_int(f"graph:{transaction.payer_vpa}:cluster_size"),
            pagerank_score=get_float(f"graph:{transaction.payer_vpa}:pagerank"),
            community_id=get_int(f"graph:{transaction.payer_vpa}:community_id", -1),
            is_bridge_node=get_bool(f"graph:{transaction.payer_vpa}:is_bridge"),
        )

        # Create main feature object
        features = UPIFeatures(
            txn_id=transaction.txn_id,
            payer_vpa=transaction.payer_vpa,
            payee_vpa=transaction.payee_vpa,
            amount=transaction.amount,
            currency=transaction.currency,
            txn_type=transaction.txn_type,
            timestamp=transaction.timestamp,
            user_behavioral=user_behavioral,
            device_network=device_network,
            merchant=merchant,
            velocity=velocity,
            graph=graph,
        )

        # Compute derived features
        features.is_night_txn = (
            transaction.timestamp.hour in [0, 1, 2, 3, 4, 5] if transaction.timestamp else False
        )
        features.is_high_value = transaction.amount > 50000
        if user_behavioral.avg_txn_amount_7d > 0:
            features.amount_zscore_7d = (
                transaction.amount - user_behavioral.avg_txn_amount_7d
            ) / max(user_behavioral.avg_txn_amount_7d, 1)
        features.velocity_anomaly_score = min(velocity.vpa_velocity_5m / 20.0, 1.0)

        return features

  # Feature Updates (Async, Fire-and-Forget)
   
    async def update_transaction_features(
        self, transaction: UPITransaction, is_fraud: bool = False
    ) -> None:
        """
        Update features after a transaction completes.

        Fire-and-forget: doesn't block the scoring path.
        """
        if not self._connected:
            await self.connect()

        pipe = self._redis.pipeline()
        payer = transaction.payer_vpa
        payee = transaction.payee_vpa
        prefix = self.config.key_prefix
        now = time.time()

        # Update velocity counters
        pipe.incr(f"{prefix}:velocity:{payer}:1m")
        pipe.incr(f"{prefix}:velocity:{payer}:5m")
        pipe.incr(f"{prefix}:velocity:{payer}:15m")
        pipe.incr(f"{prefix}:velocity:{payer}:1h")
        pipe.incr(f"{prefix}:velocity:{payer}:24h")

        pipe.incrbyfloat(f"{prefix}:velocity:{payer}:amount_1m", transaction.amount)
        pipe.incrbyfloat(f"{prefix}:velocity:{payer}:amount_5m", transaction.amount)
        pipe.incrbyfloat(f"{prefix}:velocity:{payer}:amount_1h", transaction.amount)
        pipe.incrbyfloat(f"{prefix}:velocity:{payer}:amount_24h", transaction.amount)

        # Set TTLs
        pipe.expire(f"{prefix}:velocity:{payer}:1m", self.config.velocity_ttl_1m)
        pipe.expire(f"{prefix}:velocity:{payer}:5m", self.config.velocity_ttl_5m)
        pipe.expire(f"{prefix}:velocity:{payer}:15m", self.config.velocity_ttl_5m)
        pipe.expire(f"{prefix}:velocity:{payer}:1h", self.config.velocity_ttl_1h)
        pipe.expire(f"{prefix}:velocity:{payer}:24h", self.config.velocity_ttl_24h)
        pipe.expire(f"{prefix}:velocity:{payer}:amount_1m", self.config.velocity_ttl_1m)
        pipe.expire(f"{prefix}:velocity:{payer}:amount_5m", self.config.velocity_ttl_5m)
        pipe.expire(f"{prefix}:velocity:{payer}:amount_1h", self.config.velocity_ttl_1h)
        pipe.expire(f"{prefix}:velocity:{payer}:amount_24h", self.config.velocity_ttl_24h)

        # Update user behavioral (running averages)
        # These would be computed by batch jobs, but we can increment counters
        pipe.incr(f"{prefix}:user:{payer}:txn_count_1h")
        pipe.incr(f"{prefix}:user:{payer}:txn_count_24h")
        pipe.expire(f"{prefix}:user:{payer}:txn_count_1h", 3600)
        pipe.expire(f"{prefix}:user:{payer}:txn_count_24h", 86400)

        # Track unique merchants (HyperLogLog for cardinality estimation)
        pipe.pfadd(f"{prefix}:user:{payer}:unique_merchants_hll", payee)

        # Update merchant stats
        pipe.incr(f"{prefix}:merchant:{payee}:volume_30d")
        pipe.pfadd(f"{prefix}:merchant:{payee}:unique_users_hll", payer)
        pipe.expire(f"{prefix}:merchant:{payee}:volume_30d", 2592000)  # 30 days

        # Track failed transactions
        if not is_fraud:  # Assuming we only know fraud later via webhook
            # This would be called separately for failed transactions
            pass

        await pipe.execute()

    async def update_fraud_label(self, transaction: UPITransaction, is_fraud: bool) -> None:
        """Update fraud labels for model retraining."""
        if not self._connected:
            await self.connect()

        key = f"{self.config.key_prefix}:labels:{transaction.txn_id}"
        await self._redis.set(
            key,
            json.dumps(
                {
                    "txn_id": transaction.txn_id,
                    "is_fraud": is_fraud,
                    "timestamp": time.time(),
                }
            ),
            ex=2592000,
        )  # 30 days
# Batch Operations
  
    async def get_features_batch(self, transactions: List[UPITransaction]) -> List[UPIFeatures]:
        """Fetch features for multiple transactions in parallel."""
        tasks = [self.get_features(txn) for txn in transactions]
        return await asyncio.gather(*tasks)

    async def warm_cache(self, transactions: List[UPITransaction]) -> int:
        """Pre-warm cache for expected transactions."""
        count = 0
        for txn in transactions:
            cache_key = self._build_cache_key(txn)
            cached = await self._l1_cache.get(cache_key)
            if not cached:
                await self.get_features(txn)
                count += 1
        return count

   # Health & Monitoring
   
    async def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring."""
        if not self._connected:
            return {"status": "disconnected"}

        start = time.perf_counter()
        await self._redis.ping()
        latency_ms = (time.perf_counter() - start) * 1000

        info = await self._redis.info("memory")

        return {
            "status": "healthy",
            "redis_latency_ms": round(latency_ms, 2),
            "redis_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
            "l1_cache_size": self.config.l1_cache_size,
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get feature store statistics."""
        if not self._connected:
            return {}

        info = await self._redis.info()
        keyspace = await self._redis.info("keyspace")

        total_keys = 0
        for db_info in keyspace.values():
            if isinstance(db_info, dict):
                total_keys += db_info.get("keys", 0)

        return {
            "total_keys": total_keys,
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
            "used_memory_peak_mb": round(info.get("used_memory_peak", 0) / 1024 / 1024, 2),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0)
            / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1),
        }
