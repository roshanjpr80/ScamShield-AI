"""
Model Registry for versioning and A/B testing.

Supports: model versioning, staging/production promotion, rollback, A/B experiments.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import onnx


@dataclass
class ModelMetadata:
    """Complete model metadata for registry."""
    name: str
    version: str
    model_type: str  # xgb, lgbm, transformer, gnn
    framework: str   # onnx, pytorch, sklearn
    description: str
    training_date: str
    training_data_hash: str
    metrics: Dict[str, float]  # accuracy, precision, recall, f1, auc
    latency_ms: Dict[str, float]  # p50, p95, p99
    input_schema: Dict[str, Any]  # feature names, types, shapes
    output_schema: Dict[str, Any]
    tags: List[str]
    author: str
    git_commit: str
    file_hash: str
    file_size_bytes: int
    stage: str = "staging"  # staging, production, archived
    ab_test_weight: float = 0.0  # 0.0-1.0 for A/B testing


class ModelRegistry:
    """
    Local file-based model registry with Git-like versioning.

    Structure:
    models/
    ├── registry.json          # Index of all models
    ├── xgb_fraud/
    │   ├── v1.0.0/
    │   │   ├── model.onnx
    │   │   ├── metadata.json
    │   │   └── model_card.md
    │   └── v1.1.0/
    │       └── ...
    └── transformer_fraud/
        └── ...
    """

    def __init__(self, registry_dir: str = "./models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "registry.json"
        self._index: Dict[str, List[ModelMetadata]] = self._load_index()

    def _load_index(self) -> Dict[str, List[ModelMetadata]]:
        """Load registry index from disk."""
        if not self.index_path.exists():
            return {}
        with open(self.index_path) as f:
            data = json.load(f)
        return {k: [ModelMetadata(**v) for v in vs] for k, vs in data.items()}

    def _save_index(self) -> None:
        """Save registry index to disk."""
        data = {k: [asdict(v) for v in vs] for k, vs in self._index.items()}
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 of model file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _validate_onnx(self, path: Path) -> Dict[str, Any]:
        """Validate ONNX model and extract schema."""
        model = onnx.load(str(path))
        onnx.checker.check_model(model)

        inputs = []
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]
            inputs.append({
                "name": inp.name,
                "type": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
                "shape": shape,
            })

        outputs = []
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim]
            outputs.append({
                "name": out.name,
                "type": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
                "shape": shape,
            })

        return {"inputs": inputs, "outputs": outputs}

    def register(
        self,
        model_path: str,
        name: str,
        version: str,
        model_type: str,
        description: str,
        metrics: Dict[str, float],
        training_data_hash: str,
        author: str = "unknown",
        git_commit: str = "unknown",
        tags: List[str] = None,
        latency_ms: Dict[str, float] = None,
        stage: str = "staging",
    ) -> ModelMetadata:
        """
        Register a new model version.

        Args:
            model_path: Path to ONNX model file
            name: Model name (e.g., "xgb_fraud")
            version: Semantic version (e.g., "1.2.0")
            model_type: One of "xgb", "lgbm", "transformer", "gnn"
            description: Human-readable description
            metrics: Dict with accuracy, precision, recall, f1, auc
            training_data_hash: Hash of training dataset for reproducibility
            author: Author name
            git_commit: Git commit hash
            tags: List of tags
            latency_ms: Optional p50/p95/p99 latency measurements
            stage: "staging" or "production"

        Returns:
            ModelMetadata of registered model
        """
        src_path = Path(model_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Model file not found: {src_path}")

        # Validate ONNX
        schema = self._validate_onnx(src_path)

        # Compute file hash
        file_hash = self._compute_file_hash(src_path)
        file_size = src_path.stat().st_size

        # Check if version already exists
        if name in self._index:
            for m in self._index[name]:
                if m.version == version:
                    raise ValueError(f"Version {version} already exists for {name}")

        # Create model directory
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        dst_path = model_dir / "model.onnx"
        shutil.copy2(src_path, dst_path)

        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            model_type=model_type,
            framework="onnx",
            description=description,
            training_date=datetime.utcnow().isoformat() + "Z",
            training_data_hash=training_data_hash,
            metrics=metrics,
            latency_ms=latency_ms or {},
            input_schema=schema["inputs"],
            output_schema=schema["outputs"],
            tags=tags or [],
            author=author,
            git_commit=git_commit,
            file_hash=file_hash,
            file_size_bytes=file_size,
            stage=stage,
        )

        # Save metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Update index
        if name not in self._index:
            self._index[name] = []
        self._index[name].append(metadata)
        # Sort by version (newest first)
        self._index[name].sort(key=lambda m: m.version, reverse=True)
        self._save_index()

        return metadata

    def get_latest(self, name: str, stage: str = "production") -> Optional[ModelMetadata]:
        """Get latest model version for a given stage."""
        if name not in self._index:
            return None
        for m in self._index[name]:
            if m.stage == stage:
                return m
        return None

    def get_version(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Get specific model version."""
        if name not in self._index:
            return None
        for m in self._index[name]:
            if m.version == version:
                return m
        return None

    def list_versions(self, name: str) -> List[ModelMetadata]:
        """List all versions of a model."""
        return self._index.get(name, [])

    def promote(self, name: str, version: str, target_stage: str = "production") -> ModelMetadata:
        """Promote model version to target stage."""
        metadata = self.get_version(name, version)
        if not metadata:
            raise ValueError(f"Model {name}@{version} not found")

        # Demote current production model
        if target_stage == "production":
            current = self.get_latest(name, "production")
            if current and current.version != version:
                current.stage = "archived"

        metadata.stage = target_stage
        self._save_index()
        return metadata

    def rollback(self, name: str) -> ModelMetadata:
        """Rollback to previous production version."""
        versions = self.list_versions(name)
        production_versions = [v for v in versions if v.stage in ["production", "archived"]]
        production_versions.sort(key=lambda m: m.version, reverse=True)

        if len(production_versions) < 2:
            raise ValueError("No previous version to rollback to")

        # Demote current production
        current = production_versions[0]
        current.stage = "archived"

        # Promote previous
        previous = production_versions[1]
        previous.stage = "production"
        self._save_index()
        return previous

    def set_ab_test_weight(self, name: str, version: str, weight: float) -> ModelMetadata:
        """Set A/B test traffic weight (0.0-1.0)."""
        metadata = self.get_version(name, version)
        if not metadata:
            raise ValueError(f"Model {name}@{version} not found")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        metadata.ab_test_weight = weight
        self._save_index()
        return metadata

    def get_model_path(self, name: str, version: str) -> Path:
        """Get filesystem path to model file."""
        return self.registry_dir / name / version / "model.onnx"

    def export_model_card(self, name: str, version: str) -> str:
        """Generate model card in Google format."""
        metadata = self.get_version(name, version)
        if not metadata:
            raise ValueError(f"Model {name}@{version} not found")

        card = f"""# Model Card: {name} v{version}

## Model Details
- **Name**: {name}
- **Version**: {version}
- **Type**: {metadata.model_type}
- **Framework**: {metadata.framework}
- **Author**: {metadata.author}
- **Date**: {metadata.training_date}
- **Git Commit**: {metadata.git_commit}
- **Tags**: {", ".join(metadata.tags)}

## Description
{metadata.description}

## Intended Use
Real-time UPI fraud detection for Indian payment ecosystem.

## Training Data
- **Data Hash**: {metadata.training_data_hash}
- **Features**: {len(metadata.input_schema.get('inputs', []))} input features

## Metrics
| Metric | Value |
|--------|-------|
"""
        for k, v in metadata.metrics.items():
            card += f"| {k} | {v:.4f} |\n"

        card += "\n## Latency\n| Percentile | Latency (ms) |\n|------------|--------------|\n"
        for k, v in metadata.latency_ms.items():
            card += f"| {k} | {v:.2f} |\n"

        card += f"""
## Ethical Considerations
- Model trained on Indian UPI transaction data
- Fairness audited across: region, transaction amount, user demographics
- No sensitive personal information used as features

## Limitations
- Performance may degrade on novel fraud patterns
- Requires periodic retraining (recommended: monthly)
- Latency varies by hardware (benchmarked on Intel Xeon)

## Model File
- **Path**: {self.get_model_path(name, version)}
- **Hash (SHA256)**: {metadata.file_hash}
- **Size**: {metadata.file_size_bytes / 1024 / 1024:.2f} MB
"""
        return card