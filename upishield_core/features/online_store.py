"""
Online Feature Store Implementation for ScamShield AI.

Provides high-performance online feature serving with:
- Redis-backed storage for low-latency reads
- Local caching for frequently accessed features
- Batch feature retrieval
- Feature freshness tracking
- Fallback to offline store
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from collections import OrderedDict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using in-memory store")

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

logger = logging.getLogger(__name__)


class StoreBackend(Enum):
    """Supported storage backends."""
    REDIS = "redis"
    SQLITE = "sqlite"
    MEMORY = "memory"


@dataclass
class FeatureVector:
    """Feature vector with metadata."""
    entity_id: str
    features: Dict[str, Any]
    timestamp: datetime
    ttl_seconds: int = 86400  # 24 hours default
    version: int = 1


@dataclass
class OnlineStoreConfig:
    """Configuration for online store."""
    backend: StoreBackend = StoreBackend.MEMORY
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_key_prefix: str = "scamshield:features:"
    sqlite_path: str = "./online_store.db"
    cache_size: int = 10000
    cache_ttl_seconds: int = 300
    default_ttl_seconds: int = 86400
    enable_compression: bool = True
    batch_size: int = 1000


class LRUCache:
    """Thread-safe LRU cache with TTL."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time())

    def delete(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __len__(self):
        return len(self._cache)


class OnlineFeatureStore:
    """
    High-performance online feature store for real-time fraud detection.

    Features:
    - Multiple backend support (Redis, SQLite, In-memory)
    - LRU caching for hot features
    - Batch read/write operations
    - Feature versioning and TTL
    - Freshness monitoring
    - Graceful degradation
    """

    def __init__(self, config: Optional[OnlineStoreConfig] = None):
        self.config = config or OnlineStoreConfig()
        self._backend = self._initialize_backend()
        self._cache = LRUCache(self.config.cache_size, self.config.cache_ttl_seconds)
        self._stats = {
            "reads": 0,
            "writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }
        self._lock = threading.RLock()

    def _initialize_backend(self):
        """Initialize storage backend."""
        if self.config.backend == StoreBackend.REDIS and REDIS_AVAILABLE:
            return RedisBackend(self.config)
        elif self.config.backend == StoreBackend.SQLITE and SQLITE_AVAILABLE:
            return SQLiteBackend(self.config)
        else:
            logger.info("Using in-memory backend")
            return MemoryBackend(self.config)

    def write_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        version: int = 1,
    ) -> bool:
        """
        Write feature vector for an entity.

        Args:
            entity_id: Unique entity identifier
            features: Dictionary of feature names to values
            ttl_seconds: Time-to-live in seconds
            version: Feature schema version

        Returns:
            True if successful
        """
        ttl = ttl_seconds or self.config.default_ttl_seconds
        vector = FeatureVector(
            entity_id=entity_id,
            features=features,
            timestamp=datetime.now(),
            ttl_seconds=ttl,
            version=version,
        )

        try:
            key = self._make_key(entity_id)
            self._backend.write(key, vector)
            self._cache.set(key, vector)
            self._stats["writes"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to write features for {entity_id}: {e}")
            self._stats["errors"] += 1
            return False

    def write_batch(
        self,
        feature_vectors: List[FeatureVector],
    ) -> int:
        """Write multiple feature vectors efficiently."""
        success_count = 0
        for vector in feature_vectors:
            if self.write_features(
                vector.entity_id,
                vector.features,
                vector.ttl_seconds,
                vector.version,
            ):
                success_count += 1
        return success_count

    def read_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Read feature vector for an entity.

        Args:
            entity_id: Unique entity identifier
            feature_names: Optional list of features to retrieve (None = all)

        Returns:
            Dictionary of features or None if not found
        """
        self._stats["reads"] += 1
        key = self._make_key(entity_id)

        # Check cache first
        cached = self._cache.get(key)
        if cached:
            self._stats["cache_hits"] += 1
            vector = cached
        else:
            self._stats["cache_misses"] += 1
            vector = self._backend.read(key)
            if vector:
                self._cache.set(key, vector)

        if not vector:
            return None

        # Check TTL
        if self._is_expired(vector):
            self.delete_features(entity_id)
            return None

        # Filter features if requested
        if feature_names:
            return {k: v for k, v in vector.features.items() if k in feature_names}

        return vector.features

    def read_batch(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Read features for multiple entities."""
        results = {}
        for entity_id in entity_ids:
            results[entity_id] = self.read_features(entity_id, feature_names)
        return results

    def read_multi_entity(
        self,
        entity_rows: List[Dict[str, str]],
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read features for multiple entities with different entity keys.

        Args:
            entity_rows: List of dicts with entity keys (e.g., [{"user_id": "123"}, {"merchant_id": "456"}])
            feature_names: Optional list of features to retrieve

        Returns:
            List of feature dicts with entity keys included
        """
        results = []
        for row in entity_rows:
            # Find the entity key
            entity_key = None
            entity_value = None
            for k, v in row.items():
                if k.endswith("_id"):
                    entity_key = k
                    entity_value = v
                    break

            if not entity_value:
                results.append({**row, "error": "No entity ID found"})
                continue

            features = self.read_features(entity_value, feature_names)
            if features:
                result = {**row, **features}
            else:
                result = {**row, "error": "Features not found"}
            results.append(result)

        return results

    def delete_features(self, entity_id: str) -> bool:
        """Delete features for an entity."""
        key = self._make_key(entity_id)
        try:
            self._backend.delete(key)
            self._cache.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete features for {entity_id}: {e}")
            return False

    def exists(self, entity_id: str) -> bool:
        """Check if features exist for entity."""
        key = self._make_key(entity_id)
        cached = self._cache.get(key)
        if cached and not self._is_expired(cached):
            return True
        return self._backend.exists(key)

    def get_freshness(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get feature freshness information."""
        key = self._make_key(entity_id)
        vector = self._cache.get(key) or self._backend.read(key)
        if not vector:
            return None

        age_seconds = (datetime.now() - vector.timestamp).total_seconds()
        return {
            "entity_id": entity_id,
            "timestamp": vector.timestamp.isoformat(),
            "age_seconds": age_seconds,
            "ttl_seconds": vector.ttl_seconds,
            "expired": age_seconds > vector.ttl_seconds,
            "version": vector.version,
            "feature_count": len(vector.features),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            cache_hit_rate = 0
            total_reads = self._stats["cache_hits"] + self._stats["cache_misses"]
            if total_reads > 0:
                cache_hit_rate = self._stats["cache_hits"] / total_reads

            return {
                **self._stats,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self._cache),
                "backend": self.config.backend.value,
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        start = time.time()
        try:
            # Test write/read
            test_key = "_health_check"
            test_vector = FeatureVector(
                entity_id=test_key,
                features={"test": 1},
                timestamp=datetime.now(),
                ttl_seconds=60,
            )
            self._backend.write(test_key, test_vector)
            read_back = self._backend.read(test_key)
            self._backend.delete(test_key)

            latency_ms = (time.time() - start) * 1000
            healthy = read_back is not None and read_back.features.get("test") == 1

            return {
                "healthy": healthy,
                "latency_ms": latency_ms,
                "backend": self.config.backend.value,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "backend": self.config.backend.value,
                "timestamp": datetime.now().isoformat(),
            }

    def _make_key(self, entity_id: str) -> str:
        """Create storage key."""
        return f"{self.config.redis_key_prefix}{entity_id}"

    def _is_expired(self, vector: FeatureVector) -> bool:
        """Check if feature vector is expired."""
        age = (datetime.now() - vector.timestamp).total_seconds()
        return age > vector.ttl_seconds

    def close(self):
        """Close connections."""
        if hasattr(self._backend, 'close'):
            self._backend.close()


# ============================================================
# BACKEND IMPLEMENTATIONS
# ============================================================

class MemoryBackend:
    """In-memory storage backend."""

    def __init__(self, config: OnlineStoreConfig):
        self.config = config
        self._store: Dict[str, FeatureVector] = {}
        self._lock = threading.RLock()

    def write(self, key: str, vector: FeatureVector):
        with self._lock:
            self._store[key] = vector

    def read(self, key: str) -> Optional[FeatureVector]:
        with self._lock:
            return self._store.get(key)

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def close(self):
        pass


class SQLiteBackend:
    """SQLite storage backend for persistence."""

    def __init__(self, config: OnlineStoreConfig):
        self.config = config
        self._conn = None
        self._initialize_db()

    def _initialize_db(self):
        self._conn = sqlite3.connect(
            self.config.sqlite_path,
            check_same_thread=False,
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_vectors (
                key TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                features BLOB NOT NULL,
                timestamp TEXT NOT NULL,
                ttl_seconds INTEGER NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_id ON feature_vectors(entity_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON feature_vectors(timestamp)
        """)
        self._conn.commit()

    def write(self, key: str, vector: FeatureVector):
        features_blob = pickle.dumps(vector.features)
        self._conn.execute("""
            INSERT OR REPLACE INTO feature_vectors
            (key, entity_id, features, timestamp, ttl_seconds, version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            key,
            vector.entity_id,
            features_blob,
            vector.timestamp.isoformat(),
            vector.ttl_seconds,
            vector.version,
        ))
        self._conn.commit()

    def read(self, key: str) -> Optional[FeatureVector]:
        cursor = self._conn.execute(
            "SELECT entity_id, features, timestamp, ttl_seconds, version FROM feature_vectors WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        entity_id, features_blob, timestamp_str, ttl_seconds, version = row
        features = pickle.loads(features_blob)
        timestamp = datetime.fromisoformat(timestamp_str)

        return FeatureVector(
            entity_id=entity_id,
            features=features,
            timestamp=timestamp,
            ttl_seconds=ttl_seconds,
            version=version,
        )

    def delete(self, key: str):
        self._conn.execute("DELETE FROM feature_vectors WHERE key = ?", (key,))
        self._conn.commit()

    def exists(self, key: str) -> bool:
        cursor = self._conn.execute("SELECT 1 FROM feature_vectors WHERE key = ?", (key,))
        return cursor.fetchone() is not None

    def close(self):
        if self._conn:
            self._conn.close()


class RedisBackend:
    """Redis storage backend for distributed deployment."""

    def __init__(self, config: OnlineStoreConfig):
        self.config = config
        self._client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # Test connection
        self._client.ping()
        logger.info(f"Connected to Redis at {config.redis_host}:{config.redis_port}")

    def write(self, key: str, vector: FeatureVector):
        data = {
            "entity_id": vector.entity_id,
            "features": vector.features,
            "timestamp": vector.timestamp.isoformat(),
            "ttl_seconds": vector.ttl_seconds,
            "version": vector.version,
        }
        serialized = pickle.dumps(data)
        self._client.setex(key, vector.ttl_seconds, serialized)

    def read(self, key: str) -> Optional[FeatureVector]:
        data = self._client.get(key)
        if not data:
            return None

        try:
            obj = pickle.loads(data)
            return FeatureVector(
                entity_id=obj["entity_id"],
                features=obj["features"],
                timestamp=datetime.fromisoformat(obj["timestamp"]),
                ttl_seconds=obj["ttl_seconds"],
                version=obj["version"],
            )
        except Exception as e:
            logger.error(f"Failed to deserialize feature vector: {e}")
            return None

    def delete(self, key: str):
        self._client.delete(key)

    def exists(self, key: str) -> bool:
        return self._client.exists(key) > 0

    def close(self):
        self._client.close()


# ============================================================
# FEATURE SERVER (for HTTP/gRPC serving)
# ============================================================

class FeatureServer:
    """
    HTTP server for online feature serving.
    Can be run as a standalone service.
    """

    def __init__(self, store: OnlineFeatureStore, host: str = "0.0.0.0", port: int = 8080):
        self.store = store
        self.host = host
        self.port = port
        self._app = None

    def create_app(self):
        """Create FastAPI app."""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            from typing import List, Optional
        except ImportError:
            logger.error("FastAPI not available")
            return None

        app = FastAPI(title="ScamShield Feature Server")

        class FeatureRequest(BaseModel):
            entity_id: str
            feature_names: Optional[List[str]] = None

        class BatchFeatureRequest(BaseModel):
            entity_ids: List[str]
            feature_names: Optional[List[str]] = None

        class MultiEntityRequest(BaseModel):
            entity_rows: List[Dict[str, str]]
            feature_names: Optional[List[str]] = None

        class WriteRequest(BaseModel):
            entity_id: str
            features: Dict[str, Any]
            ttl_seconds: Optional[int] = None

        @app.get("/health")
        async def health():
            return self.store.health_check()

        @app.get("/stats")
        async def stats():
            return self.store.get_stats()

        @app.post("/features")
        async def read_features(request: FeatureRequest):
            features = self.store.read_features(request.entity_id, request.feature_names)
            if features is None:
                raise HTTPException(status_code=404, detail="Features not found")
            return {"entity_id": request.entity_id, "features": features}

        @app.post("/features/batch")
        async def read_batch(request: BatchFeatureRequest):
            results = self.store.read_batch(request.entity_ids, request.feature_names)
            return {"results": results}

        @app.post("/features/multi-entity")
        async def read_multi_entity(request: MultiEntityRequest):
            results = self.store.read_multi_entity(request.entity_rows, request.feature_names)
            return {"results": results}

        @app.post("/features/write")
        async def write_features(request: WriteRequest):
            success = self.store.write_features(
                request.entity_id,
                request.features,
                request.ttl_seconds,
            )
            if not success:
                raise HTTPException(status_code=500, detail="Write failed")
            return {"success": True}

        @app.get("/features/{entity_id}/freshness")
        async def get_freshness(entity_id: str):
            freshness = self.store.get_freshness(entity_id)
            if freshness is None:
                raise HTTPException(status_code=404, detail="Features not found")
            return freshness

        self._app = app
        return app

    def run(self):
        """Run the server."""
        import uvicorn
        if self._app is None:
            self.create_app()
        uvicorn.run(self._app, host=self.host, port=self.port)


# FACTORY FUNCTIONS

_default_store: Optional[OnlineFeatureStore] = None


def get_online_store(config: Optional[OnlineStoreConfig] = None) -> OnlineFeatureStore:
    """Get or create default online feature store."""
    global _default_store
    if _default_store is None:
        _default_store = OnlineFeatureStore(config)
    return _default_store


def create_feature_server(
    config: Optional[OnlineStoreConfig] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> FeatureServer:
    """Create feature server with store."""
    store = get_online_store(config)
    return FeatureServer(store, host, port)