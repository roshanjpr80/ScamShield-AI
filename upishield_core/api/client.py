"""
UPIShield API Client.

Async/Sync HTTP client for integrating with UPIShield API gateway.
Supports: scoring, batch scoring, health checks, model info.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from httpx import AsyncClient, Limits, Timeout

from upishield_core.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ModelInfoResponse,
    ScoreRequest,
    ScoreResponse,
    UPITransaction,
)


@dataclass
class UPIShieldClientConfig:
    """Configuration for UPIShield API client."""

    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 10.0
    connect_timeout: float = 5.0
    max_connections: int = 100
    max_keepalive_connections: int = 20
    retry_attempts: int = 3
    retry_backoff: float = 0.5
    enable_compression: bool = True


class UPIShieldClient:
    """
    Async client for UPIShield API.

    Usage:
        async with UPIShieldClient(config) as client:
            result = await client.score(transaction)
    """

    def __init__(self, config: UPIShieldClientConfig = None):
        self.config = config or UPIShieldClientConfig()
        self._client: Optional[AsyncClient] = None
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "UPIShield-Python/1.0",
        }
        if self.config.api_key:
            self._headers["Authorization"] = f"Bearer {self.config.api_key}"

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._client:
            return

        limits = Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_keepalive_connections,
        )
        timeout = Timeout(
            timeout=self.config.timeout,
            connect=self.config.connect_timeout,
        )

        self._client = AsyncClient(
            base_url=self.config.base_url,
            headers=self._headers,
            timeout=timeout,
            limits=limits,
            http2=True,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        json_data: Any = None,
        params: Dict = None,
    ) -> httpx.Response:
        """Make HTTP request with retry logic."""
        if not self._client:
            await self.connect()

        last_exception = None
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self._client.request(
                    method=method,
                    url=path,
                    json=json_data,
                    params=params,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise  # Don't retry client errors
                last_exception = e
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_exception = e

            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(self.config.retry_backoff * (attempt + 1))

        raise last_exception

   # Scoring Endpoints
 
    async def score(self, transaction: UPITransaction) -> ScoreResponse:
        """
        Score a single transaction.

        Returns ScoreResponse with decision, risk level, and explanation.
        """
        request = ScoreRequest(transaction=transaction)
        response = await self._request("POST", "/api/v1/score", json_data=request.model_dump())
        return ScoreResponse(**response.json())

    async def score_batch(self, transactions: List[UPITransaction]) -> BatchScoreResponse:
        """
        Score multiple transactions in a single request.

        More efficient than individual calls for batch processing.
        """
        request = BatchScoreRequest(transactions=transactions)
        response = await self._request(
            "POST", "/api/v1/score/batch", json_data=request.model_dump()
        )
        return BatchScoreResponse(**response.json())

    async def score_async(self, transaction: UPITransaction) -> str:
        """
        Submit transaction for async scoring.

        Returns job ID - poll /api/v1/jobs/{job_id} for result.
        """
        request = ScoreRequest(transaction=transaction, async_mode=True)
        response = await self._request("POST", "/api/v1/score", json_data=request.model_dump())
        return response.json().get("job_id")

    async def get_job_result(self, job_id: str) -> ScoreResponse:
        """Get result of async scoring job."""
        response = await self._request("GET", f"/api/v1/jobs/{job_id}")
        return ScoreResponse(**response.json())

   # Health & Monitoring

    async def health(self) -> HealthResponse:
        """Check API health."""
        response = await self._request("GET", "/health")
        return HealthResponse(**response.json())

    async def readiness(self) -> HealthResponse:
        """Check API readiness (includes model loading)."""
        response = await self._request("GET", "/ready")
        return HealthResponse(**response.json())

    async def metrics(self) -> str:
        """Get Prometheus metrics."""
        response = await self._request("GET", "/metrics")
        return response.text

  # Model Management
 
    async def get_model_info(self) -> ModelInfoResponse:
        """Get current model information."""
        response = await self._request("GET", "/api/v1/models/info")
        return ModelInfoResponse(**response.json())

    async def list_models(self) -> Dict[str, Any]:
        """List all available model versions."""
        response = await self._request("GET", "/api/v1/models")
        return response.json()

    async def promote_model(self, model_name: str, version: str, stage: str = "production") -> Dict:
        """Promote model version to stage."""
        response = await self._request(
            "POST",
            f"/api/v1/models/{model_name}/versions/{version}/promote",
            json_data={"stage": stage},
        )
        return response.json()

   # Utility Methods
   
    async def warm_up(self, num_requests: int = 10) -> float:
        """Warm up the API with dummy requests."""
        dummy_txn = UPITransaction(
            txn_id="warmup_001",
            payer_vpa="user@upi",
            payee_vpa="merchant@upi",
            amount=100.0,
            device_id="warmup_device",
            client_ip="127.0.0.1",
        )

        start = time.perf_counter()
        tasks = [self.score(dummy_txn) for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        return (time.perf_counter() - start) * 1000


class UPIShieldSyncClient:
    """
    Synchronous wrapper for UPIShieldClient.

    Usage:
        with UPIShieldSyncClient(config) as client:
            result = client.score(transaction)
    """

    def __init__(self, config: UPIShieldClientConfig = None):
        self._async_client = UPIShieldClient(config)
        self._loop = None

    def __enter__(self):
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_client.connect())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._loop.run_until_complete(self._async_client.close())
        return False

    def _run_async(self, coro):
        """Run async coroutine in event loop."""
        return self._loop.run_until_complete(coro)

    # Sync versions of all async methods
    def score(self, transaction: UPITransaction) -> ScoreResponse:
        return self._run_async(self._async_client.score(transaction))

    def score_batch(self, transactions: List[UPITransaction]) -> BatchScoreResponse:
        return self._run_async(self._async_client.score_batch(transactions))

    def health(self) -> HealthResponse:
        return self._run_async(self._async_client.health())

    def readiness(self) -> HealthResponse:
        return self._run_async(self._async_client.readiness())

    def metrics(self) -> str:
        return self._run_async(self._async_client.metrics())

    def get_model_info(self) -> ModelInfoResponse:
        return self._run_async(self._async_client.get_model_info())

    def list_models(self) -> Dict[str, Any]:
        return self._run_async(self._async_client.list_models())

    def promote_model(self, model_name: str, version: str, stage: str = "production") -> Dict:
        return self._run_async(self._async_client.promote_model(model_name, version, stage))

    def warm_up(self, num_requests: int = 10) -> float:
        return self._run_async(self._async_client.warm_up(num_requests))


# High-level convenience functions


async def quick_score(
    transaction: UPITransaction,
    base_url: str = "http://localhost:8000",
    api_key: str = None,
) -> ScoreResponse:
    """Quick one-off scoring without context manager."""
    config = UPIShieldClientConfig(base_url=base_url, api_key=api_key)
    async with UPIShieldClient(config) as client:
        return await client.score(transaction)


def quick_score_sync(
    transaction: UPITransaction,
    base_url: str = "http://localhost:8000",
    api_key: str = None,
) -> ScoreResponse:
    """Quick one-off synchronous scoring."""
    config = UPIShieldClientConfig(base_url=base_url, api_key=api_key)
    with UPIShieldSyncClient(config) as client:
        return client.score(transaction)
