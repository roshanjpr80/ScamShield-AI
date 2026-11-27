"""
UPIShield CLI.

Command-line interface for model management, scoring, and administration.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from upishield_core.api.client import UPIShieldClient, UPIShieldClientConfig, quick_score_sync
from upishield_core.api.schemas import TransactionType, UPITransaction
from upishield_core.features.store import FeatureStore, FeatureStoreConfig
from upishield_core.models.export import (
    benchmark_model,
    compare_models,
    export_pipeline,
)
from upishield_core.models.registry import ModelRegistry

app = typer.Typer(
    name="upishield",
    help="UPIShield - Real-time UPI Fraud Detection CLI",
    add_completion=False,
)
console = Console()


# Model Commands

model_app = typer.Typer(name="model", help="Model management commands")
app.add_typer(model_app)


@model_app.command("register")
def model_register(
    model_path: Path = typer.Argument(..., help="Path to ONNX model file"),
    name: str = typer.Option(..., "--name", "-n", help="Model name"),
    version: str = typer.Option(..., "--version", "-v", help="Semantic version"),
    model_type: str = typer.Option(
        ..., "--type", "-t", help="Model type: xgb, lgbm, transformer, gnn"
    ),
    description: str = typer.Option(..., "--description", "-d", help="Model description"),
    metrics: Path = typer.Option(..., "--metrics", "-m", help="Path to metrics JSON file"),
    training_data_hash: str = typer.Option(..., "--data-hash", help="Training data hash"),
    author: str = typer.Option("unknown", "--author", help="Author name"),
    git_commit: str = typer.Option("unknown", "--git-commit", help="Git commit hash"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
    registry_dir: Path = typer.Option(
        Path("./models"), "--registry", "-r", help="Registry directory"
    ),
    stage: str = typer.Option("staging", "--stage", "-s", help="Stage: staging or production"),
):
    """Register a new model version."""
    registry = ModelRegistry(str(registry_dir))

    with open(metrics) as f:
        metrics_data = json.load(f)

    metadata = registry.register(
        model_path=str(model_path),
        name=name,
        version=version,
        model_type=model_type,
        description=description,
        metrics=metrics_data,
        training_data_hash=training_data_hash,
        author=author,
        git_commit=git_commit,
        tags=tags.split(",") if tags else [],
        stage=stage,
    )

    console.print(f"[green]✓[/green] Registered {name} v{version} ({model_type})")
    console.print(f"  Stage: {metadata.stage}")
    console.print(f"  Path: {registry.get_model_path(name, version)}")


@model_app.command("list")
def model_list(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Filter by model name"),
    registry_dir: Path = typer.Option(
        Path("./models"), "--registry", "-r", help="Registry directory"
    ),
):
    """List registered models."""
    registry = ModelRegistry(str(registry_dir))

    if name:
        versions = registry.list_versions(name)
        if not versions:
            console.print(f"[yellow]No versions found for {name}[/yellow]")
            return

        table = Table(title=f"Versions for {name}")
        table.add_column("Version")
        table.add_column("Stage")
        table.add_column("Type")
        table.add_column("Metrics")
        table.add_column("Date")
        table.add_column("Size (MB)")

        for v in versions:
            metrics_str = ", ".join(f"{k}: {v:.3f}" for k, v in v.metrics.items())
            table.add_row(
                v.version,
                v.stage,
                v.model_type,
                metrics_str,
                v.training_date[:10],
                f"{v.file_size_bytes / 1024 / 1024:.2f}",
            )
        console.print(table)
    else:
        # List all models
        if not registry._index:
            console.print("[yellow]No models registered[/yellow]")
            return

        table = Table(title="All Models")
        table.add_column("Name")
        table.add_column("Latest Version")
        table.add_column("Stage")
        table.add_column("Type")
        table.add_column("Versions")

        for model_name, versions in registry._index.items():
            latest = versions[0]
            table.add_row(
                model_name,
                latest.version,
                latest.stage,
                latest.model_type,
                str(len(versions)),
            )
        console.print(table)


@model_app.command("promote")
def model_promote(
    name: str = typer.Argument(..., help="Model name"),
    version: str = typer.Argument(..., help="Version to promote"),
    stage: str = typer.Option("production", "--stage", "-s", help="Target stage"),
    registry_dir: Path = typer.Option(
        Path("./models"), "--registry", "-r", help="Registry directory"
    ),
):
    """Promote model version to stage."""
    registry = ModelRegistry(str(registry_dir))
    metadata = registry.promote(name, version, stage)
    console.print(f"[green]✓[/green] Promoted {name} v{version} to {stage}")


@model_app.command("rollback")
def model_rollback(
    name: str = typer.Argument(..., help="Model name"),
    registry_dir: Path = typer.Option(
        Path("./models"), "--registry", "-r", help="Registry directory"
    ),
):
    """Rollback to previous production version."""
    registry = ModelRegistry(str(registry_dir))
    metadata = registry.rollback(name)
    console.print(f"[green]✓[/green] Rolled back {name} to v{metadata.version}")


@model_app.command("card")
def model_card(
    name: str = typer.Argument(..., help="Model name"),
    version: str = typer.Argument(..., help="Version"),
    registry_dir: Path = typer.Option(
        Path("./models"), "--registry", "-r", help="Registry directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (default: stdout)"
    ),
):
    """Generate model card."""
    registry = ModelRegistry(str(registry_dir))
    card = registry.export_model_card(name, version)

    if output:
        output.write_text(card)
        console.print(f"[green]✓[/green] Model card written to {output}")
    else:
        console.print(card)


@model_app.command("benchmark")
def model_benchmark(
    model_path: Path = typer.Argument(..., help="Path to ONNX model"),
    runs: int = typer.Option(1000, "--runs", "-n", help="Number of benchmark runs"),
    warmup: int = typer.Option(100, "--warmup", "-w", help="Warmup runs"),
    threads: int = typer.Option(1, "--threads", "-t", help="Number of threads"),
):
    """Benchmark ONNX model latency."""
    console.print(f"[blue]Benchmarking {model_path}...[/blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmark...", total=None)
        results = benchmark_model(
            model_path,
            num_runs=runs,
            warmup_runs=warmup,
            num_threads=threads,
        )

    table = Table(title=f"Benchmark Results: {model_path.name}")
    table.add_column("Metric")
    table.add_column("Value")

    for key, value in results.items():
        table.add_row(key, f"{value:.2f} ms" if "ms" in key else str(value))

    console.print(table)


@model_app.command("compare")
def model_compare(
    original: Path = typer.Argument(..., help="Original model path"),
    quantized: Path = typer.Argument(..., help="Quantized model path"),
    samples: int = typer.Option(1000, "--samples", "-n", help="Number of test samples"),
):
    """Compare original vs quantized model."""
    console.print("[blue]Comparing models...[/blue]")

    results = compare_models(original, quantized, num_samples=samples)

    table = Table(title="Model Comparison")
    table.add_column("Metric")
    table.add_column("Original")
    table.add_column("Quantized")

    # Accuracy
    acc = results["accuracy"]
    table.add_row("Cosine Similarity", "-", f"{acc['cosine_similarity']:.6f}")
    table.add_row("MSE", "-", f"{acc['mse']:.6f}")
    table.add_row("MAE", "-", f"{acc['mae']:.6f}")
    table.add_row("Max Diff", "-", f"{acc['max_diff']:.6f}")

    # Latency
    table.add_row("", "", "")
    table.add_row("Speedup", "-", f"{results['speedup']:.2f}x")
    table.add_row("Size Reduction", "-", f"{results['size_reduction_pct']:.1f}%")

    orig_lat = results["latency_original"]
    quant_lat = results["latency_quantized"]

    table.add_row("Mean Latency", f"{orig_lat['mean_ms']:.2f} ms", f"{quant_lat['mean_ms']:.2f} ms")
    table.add_row("P50 Latency", f"{orig_lat['p50_ms']:.2f} ms", f"{quant_lat['p50_ms']:.2f} ms")
    table.add_row("P95 Latency", f"{orig_lat['p95_ms']:.2f} ms", f"{quant_lat['p95_ms']:.2f} ms")
    table.add_row("P99 Latency", f"{orig_lat['p99_ms']:.2f} ms", f"{quant_lat['p99_ms']:.2f} ms")

    console.print(table)


@model_app.command("export")
def model_export(
    model_path: Path = typer.Argument(..., help="Path to PyTorch model (.pt/.pth)"),
    output_dir: Path = typer.Option(
        Path("./models/exported"), "--output", "-o", help="Output directory"
    ),
    name: str = typer.Option("model", "--name", "-n", help="Model name"),
    quantization: str = typer.Option(
        "dynamic", "--quant", "-q", help="Quantization: dynamic, static, none"
    ),
    calibrate: Optional[Path] = typer.Option(
        None, "--calibrate", "-c", help="Calibration data (.npy) for static quantization"
    ),
    optimize: bool = typer.Option(
        True, "--optimize/--no-optimize", help="Apply ONNX Runtime optimization"
    ),
):
    """Export PyTorch model to ONNX with quantization."""
    import torch

    console.print("[blue]Loading PyTorch model...[/blue]")
    model = torch.load(model_path, map_location="cpu", weights_only=True)

    calibration_data = None
    if calibrate and calibrate.exists():
        import numpy as np

        calibration_data = np.load(calibrate)

    console.print("[blue]Running export pipeline...[/blue]")
    paths = export_pipeline(
        model,
        output_dir,
        name,
        calibration_data=calibration_data,
        quantization=quantization,
        optimize=optimize,
    )

    console.print("[green]✓[/green] Export complete!")
    for key, path in paths.items():
        console.print(f"  {key}: {path}")


# ============================================================
# Scoring Commands
# ============================================================

score_app = typer.Typer(name="score", help="Transaction scoring commands")
app.add_typer(score_app)


@score_app.command("single")
def score_single(
    txn_id: str = typer.Option(..., "--txn-id", help="Transaction ID"),
    payer_vpa: str = typer.Option(..., "--payer", help="Payer VPA"),
    payee_vpa: str = typer.Option(..., "--payee", help="Payee VPA"),
    amount: float = typer.Option(..., "--amount", "-a", help="Transaction amount"),
    txn_type: str = typer.Option("P2M", "--type", "-t", help="Transaction type"),
    device_id: str = typer.Option("unknown", "--device", "-d", help="Device ID"),
    client_ip: str = typer.Option("127.0.0.1", "--ip", help="Client IP"),
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="API URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key"),
    explain: bool = typer.Option(True, "--explain/--no-explain", help="Include explanation"),
):
    """Score a single transaction."""
    transaction = UPITransaction(
        txn_id=txn_id,
        payer_vpa=payer_vpa,
        payee_vpa=payee_vpa,
        amount=amount,
        txn_type=TransactionType(txn_type),
        device_id=device_id,
        client_ip=client_ip,
    )

    console.print(f"[blue]Scoring transaction {txn_id}...[/blue]")
    result = quick_score_sync(transaction, base_url=url, api_key=api_key)

    console.print(f"[green]Decision:[/green] {result.decision}")
    console.print(f"[green]Risk Level:[/green] {result.risk_level}")
    console.print(f"[green]Risk Score:[/green] {result.risk_score:.4f}")
    console.print(f"[green]Confidence:[/green] {result.confidence:.4f}")
    console.print(f"[green]Processing Time:[/green] {result.processing_time_ms:.2f} ms")

    if explain and result.explanation:
        console.print("\n[blue]Explanation:[/blue]")
        syntax = Syntax(json.dumps(result.explanation, indent=2), "json", theme="monokai")
        console.print(syntax)


@score_app.command("batch")
def score_batch(
    file: Path = typer.Argument(..., help="JSON file with transactions"),
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="API URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Score transactions from JSON file."""
    with open(file) as f:
        data = json.load(f)

    transactions = [UPITransaction(**txn) for txn in data.get("transactions", [])]

    console.print(f"[blue]Scoring {len(transactions)} transactions...[/blue]")

    config = UPIShieldClientConfig(base_url=url, api_key=api_key)
    with UPIShieldClient(config) as client:
        result = asyncio.run(client.score_batch(transactions))

    console.print(f"[green]Processed:[/green] {result.total_processed}")
    console.print(f"[green]Total Time:[/green] {result.total_time_ms:.2f} ms")

    # Summary
    decisions = {}
    for r in result.results:
        decisions[r.decision] = decisions.get(r.decision, 0) + 1

    table = Table(title="Decisions")
    table.add_column("Decision")
    table.add_column("Count")
    for decision, count in decisions.items():
        table.add_row(decision, str(count))
    console.print(table)

    if output:
        output.write_text(
            json.dumps(
                {
                    "results": [r.model_dump() for r in result.results],
                    "total_processed": result.total_processed,
                    "total_time_ms": result.total_time_ms,
                },
                indent=2,
                default=str,
            )
        )
        console.print(f"[green]✓[/green] Results written to {output}")


# Feature Store Commands

feature_app = typer.Typer(name="features", help="Feature store commands")
app.add_typer(feature_app)


@feature_app.command("health")
def feature_health(
    redis_url: str = typer.Option("redis://localhost:6379", "--redis", "-r", help="Redis URL"),
):
    """Check feature store health."""

    async def check():
        config = FeatureStoreConfig(redis_url=redis_url)
        store = FeatureStore(config)
        await store.connect()
        health = await store.health_check()
        await store.close()
        return health

    health = asyncio.run(check())

    table = Table(title="Feature Store Health")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in health.items():
        table.add_row(k, str(v))
    console.print(table)


@feature_app.command("stats")
def feature_stats(
    redis_url: str = typer.Option("redis://localhost:6379", "--redis", "-r", help="Redis URL"),
):
    """Get feature store statistics."""

    async def check():
        config = FeatureStoreConfig(redis_url=redis_url)
        store = FeatureStore(config)
        await store.connect()
        stats = await store.get_stats()
        await store.close()
        return stats

    stats = asyncio.run(check())

    table = Table(title="Feature Store Stats")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)

# Server Commands

server_app = typer.Typer(name="server", help="API server commands")
app.add_typer(server_app)


@server_app.command("run")
def server_run(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Run UPIShield API server."""
    import uvicorn

    console.print(f"[blue]Starting server on {host}:{port}...[/blue]")
    uvicorn.run(
        "upishield_core.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@server_app.command("health")
def server_health(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL"),
):
    """Check server health."""

    async def check():
        config = UPIShieldClientConfig(base_url=url)
        async with UPIShieldClient(config) as client:
            return await client.health()

    health = asyncio.run(check())

    table = Table(title="Server Health")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in health.model_dump().items():
        table.add_row(k, str(v))
    console.print(table)


# Development Commands

dev_app = typer.Typer(name="dev", help="Development utilities")
app.add_typer(dev_app)


@dev_app.command("generate-test-data")
def generate_test_data(
    count: int = typer.Option(100, "--count", "-n", help="Number of transactions"),
    output: Path = typer.Option(
        Path("test_transactions.json"), "--output", "-o", help="Output file"
    ),
    fraud_ratio: float = typer.Option(0.1, "--fraud-ratio", help="Ratio of fraud transactions"),
):
    """Generate test transaction data."""
    import random
    from datetime import datetime, timedelta

    vpases = [f"user{i}@upi" for i in range(50)]
    merchant_vpases = [f"merchant{i}@upi" for i in range(20)]

    transactions = []
    for i in range(count):
        is_fraud = random.random() < fraud_ratio

        if is_fraud:
            # Fraud patterns
            amount = random.choice([9999, 19999, 49999, 99999, 199999])
            velocity = random.randint(10, 50)
        else:
            amount = round(random.uniform(10, 50000), 2)
            velocity = random.randint(1, 5)

        txn = {
            "txn_id": f"TXN{datetime.now().strftime('%Y%m%d')}{i:06d}",
            "payer_vpa": random.choice(vpases),
            "payee_vpa": random.choice(merchant_vpases),
            "amount": amount,
            "currency": "INR",
            "txn_type": random.choice(["P2P", "P2M", "M2P"]),
            "device_id": f"device_{random.randint(1, 100)}",
            "client_ip": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 1440))).isoformat(),
        }
        transactions.append(txn)

    output.write_text(json.dumps({"transactions": transactions}, indent=2))
    console.print(f"[green]✓[/green] Generated {count} transactions to {output}")


@dev_app.command("load-test")
def load_test(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key"),
    duration: int = typer.Option(30, "--duration", "-d", help="Duration in seconds"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrent requests"),
    rps: int = typer.Option(100, "--rps", "-r", help="Requests per second"),
):
    """Run load test against server."""
    import random
    from datetime import datetime

    vpases = [f"user{i}@upi" for i in range(100)]
    merchant_vpases = [f"merchant{i}@upi" for i in range(50)]

    async def run_load_test():
        config = UPIShieldClientConfig(base_url=url, api_key=api_key)
        async with UPIShieldClient(config) as client:
            start_time = time.time()
            request_count = 0
            error_count = 0
            latencies = []

            async def make_request():
                nonlocal request_count, error_count
                transaction = UPITransaction(
                    txn_id=f"LOAD{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(1000, 9999)}",
                    payer_vpa=random.choice(vpases),
                    payee_vpa=random.choice(merchant_vpases),
                    amount=round(random.uniform(10, 50000), 2),
                    txn_type=random.choice(["P2P", "P2M"]),
                    device_id=f"device_{random.randint(1, 1000)}",
                    client_ip=f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
                )

                req_start = time.perf_counter()
                try:
                    await client.score(transaction)
                    latencies.append((time.perf_counter() - req_start) * 1000)
                except Exception:
                    error_count += 1
                request_count += 1

            # Run for duration
            while time.time() - start_time < duration:
                tasks = [make_request() for _ in range(concurrency)]
                await asyncio.gather(*tasks)
                await asyncio.sleep(concurrency / rps)

            return request_count, error_count, latencies

    import time

    console.print(f"[blue]Running load test for {duration}s at {rps} RPS...[/blue]")

    with Progress(console=console) as progress:
        task = progress.add_task("Load testing...", total=duration)
        # We can't easily show progress in async, so just run
        request_count, error_count, latencies = asyncio.run(run_load_test())

    if latencies:
        latencies.sort()
        table = Table(title="Load Test Results")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Total Requests", str(request_count))
        table.add_row("Errors", str(error_count))
        table.add_row("Error Rate", f"{error_count / max(request_count, 1) * 100:.2f}%")
        table.add_row("Actual RPS", f"{request_count / duration:.1f}")
        table.add_row("P50 Latency", f"{latencies[len(latencies) // 2]:.2f} ms")
        table.add_row("P95 Latency", f"{latencies[int(len(latencies) * 0.95)]:.2f} ms")
        table.add_row("P99 Latency", f"{latencies[int(len(latencies) * 0.99)]:.2f} ms")
        table.add_row("Max Latency", f"{latencies[-1]:.2f} ms")
        console.print(table)
    else:
        console.print("[red]No successful requests[/red]")


if __name__ == "__main__":
    app()
