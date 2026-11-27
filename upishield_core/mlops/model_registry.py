"""
Model Registry for ScamShield AI.

Provides model versioning, staging, and deployment management
integrated with MLflow Model Registry.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import json

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion, RegisteredModel

from .mlflow_config import MLflowManager, get_mlflow_manager

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model deployment stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelType(Enum):
    """Types of models in the registry."""
    ENSEMBLE = "ensemble"  # Main fraud detection ensemble
    TRANSACTION = "transaction"  # Transaction classifier
    SMS = "sms"  # SMS fraud detector
    DOCUMENT = "document"  # Document forensics
    OCR = "ocr"  # OCR model
    EMBEDDING = "embedding"  # Embedding model


@dataclass
class ModelMetadata:
    """Metadata for registered model."""
    name: str
    model_type: ModelType
    version: int
    stage: ModelStage
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = ""
    artifact_path: str = "model"


@dataclass
class ModelDeployment:
    """Model deployment configuration."""
    model_name: str
    version: int
    stage: ModelStage
    endpoint_name: str
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    traffic_split: Dict[str, float] = field(default_factory=dict)  # For A/B testing
    health_check: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """
    Model Registry for managing fraud detection models.

    Features:
    - Model registration with metadata
    - Version management and promotion
    - Stage transitions (Staging -> Production)
    - Model loading for inference
    - Deployment tracking
    - Model comparison and selection
    """

    def __init__(self, mlflow_manager: Optional[MLflowManager] = None):
        self.mlflow = mlflow_manager or get_mlflow_manager()
        self._client = self.mlflow.client

    def register_model(
        self,
        model_uri: str,
        name: str,
        model_type: ModelType,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        run_id: str = "",
        artifact_path: str = "model",
    ) -> ModelVersion:
        """Register a new model version."""
        # Register in MLflow
        mv = mlflow.register_model(model_uri, name, tags=tags)

        # Add custom tags
        custom_tags = {
            "model_type": model_type.value,
            "project": "scamshield",
            "registered_at": datetime.now().isoformat(),
            **(tags or {}),
        }
        for key, value in custom_tags.items():
            self._client.set_model_version_tag(name, mv.version, key, value)

        # Add description
        if description:
            self._client.update_model_version(name, mv.version, description)

        # Log metrics and params as tags for easy querying
        if metrics:
            for key, value in metrics.items():
                self._client.set_model_version_tag(name, mv.version, f"metric_{key}", str(value))

        if params:
            for key, value in params.items():
                self._client.set_model_version_tag(name, mv.version, f"param_{key}", str(value))

        logger.info(f"Registered model {name} v{mv.version} as {model_type.value}")
        return mv

    def get_model_metadata(self, name: str, version: int) -> ModelMetadata:
        """Get model metadata."""
        mv = self._client.get_model_version(name, version)

        # Extract custom metadata from tags
        tags = dict(mv.tags)
        model_type = ModelType(tags.get("model_type", "ensemble"))
        stage = ModelStage(mv.current_stage)

        # Extract metrics and params from tags
        metrics = {}
        params = {}
        for key, value in tags.items():
            if key.startswith("metric_"):
                metrics[key[7:]] = float(value)
            elif key.startswith("param_"):
                params[key[6:]] = value

        return ModelMetadata(
            name=name,
            model_type=model_type,
            version=version,
            stage=stage,
            description=mv.description or "",
            tags={k: v for k, v in tags.items() if not k.startswith(("metric_", "param_"))},
            metrics=metrics,
            params=params,
            created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000).isoformat(),
            updated_at=datetime.fromtimestamp(mv.last_updated_timestamp / 1000).isoformat(),
            run_id=mv.run_id,
            artifact_path=mv.source,
        )

    def list_versions(self, name: str) -> List[ModelMetadata]:
        """List all versions of a model."""
        versions = self._client.get_latest_versions(name, stages=["None", "Staging", "Production", "Archived"])
        return [self.get_model_metadata(name, v.version) for v in versions]

    def get_latest_version(
        self,
        name: str,
        stage: Optional[ModelStage] = None,
    ) -> Optional[ModelMetadata]:
        """Get latest model version, optionally filtered by stage."""
        stages = [stage.value] if stage else ["Production", "Staging", "None", "Archived"]
        versions = self._client.get_latest_versions(name, stages=stages)
        if not versions:
            return None
        return self.get_model_metadata(name, versions[0].version)

    def promote_model(
        self,
        name: str,
        version: int,
        target_stage: ModelStage,
        archive_existing: bool = True,
    ) -> ModelVersion:
        """Promote model to target stage."""
        self._client.transition_model_version_stage(
            name=name,
            version=version,
            stage=target_stage.value,
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Promoted {name} v{version} to {target_stage.value}")
        return self._client.get_model_version(name, version)

    def demote_model(self, name: str, version: int) -> ModelVersion:
        """Demote model to Archived stage."""
        return self.promote_model(name, version, ModelStage.ARCHIVED)

    def load_model(self, name: str, version: Optional[int] = None, stage: Optional[ModelStage] = None):
        """Load model for inference."""
        if version:
            model_uri = f"models:/{name}/{version}"
        elif stage:
            model_uri = f"models:/{name}/{stage.value}"
        else:
            model_uri = f"models:/{name}/Production"

        return mlflow.pyfunc.load_model(model_uri)

    def load_model_by_type(
        self,
        model_type: ModelType,
        stage: ModelStage = ModelStage.PRODUCTION,
    ):
        """Load model by type."""
        # Map model types to registered names
        type_to_name = {
            ModelType.ENSEMBLE: "scamshield-ensemble",
            ModelType.TRANSACTION: "scamshield-transaction",
            ModelType.SMS: "scamshield-sms",
            ModelType.DOCUMENT: "scamshield-document",
            ModelType.OCR: "scamshield-ocr",
            ModelType.EMBEDDING: "scamshield-embedding",
        }
        name = type_to_name.get(model_type)
        if not name:
            raise ValueError(f"No registered model for type: {model_type}")
        return self.load_model(name, stage=stage)

    def compare_versions(
        self,
        name: str,
        version_a: int,
        version_b: int,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare two model versions."""
        meta_a = self.get_model_metadata(name, version_a)
        meta_b = self.get_model_metadata(name, version_b)

        comparison = {
            "model": name,
            "version_a": version_a,
            "version_b": version_b,
            "stage_a": meta_a.stage.value,
            "stage_b": meta_b.stage.value,
            "metrics_a": meta_a.metrics,
            "metrics_b": meta_b.metrics,
            "params_a": meta_a.params,
            "params_b": meta_b.params,
        }

        if metrics:
            metric_comparison = {}
            for metric in metrics:
                val_a = meta_a.metrics.get(metric)
                val_b = meta_b.metrics.get(metric)
                if val_a is not None and val_b is not None:
                    metric_comparison[metric] = {
                        "version_a": val_a,
                        "version_b": val_b,
                        "diff": val_b - val_a,
                        "improvement": (val_b - val_a) / abs(val_a) if val_a != 0 else float('inf'),
                    }
            comparison["metric_comparison"] = metric_comparison

        return comparison

    def search_models(
        self,
        model_type: Optional[ModelType] = None,
        stage: Optional[ModelStage] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[ModelMetadata]:
        """Search registered models."""
        filter_parts = ["tags.project = 'scamshield'"]
        if model_type:
            filter_parts.append(f"tags.model_type = '{model_type.value}'")
        if stage:
            filter_parts.append(f"current_stage = '{stage.value}'")
        if tags:
            for key, value in tags.items():
                filter_parts.append(f"tags.{key} = '{value}'")

        filter_string = " AND ".join(filter_parts)
        results = self._client.search_model_versions(filter_string)

        return [self.get_model_metadata(r.name, r.version) for r in results]

    def delete_model_version(self, name: str, version: int):
        """Delete a model version."""
        self._client.delete_model_version(name, version)
        logger.info(f"Deleted model {name} v{version}")

    def delete_registered_model(self, name: str):
        """Delete entire registered model."""
        self._client.delete_registered_model(name)
        logger.info(f"Deleted registered model {name}")

    def set_model_alias(self, name: str, version: int, alias: str):
        """Set alias for model version (e.g., 'champion', 'challenger')."""
        self._client.set_registered_model_alias(name, alias, version)
        logger.info(f"Set alias '{alias}' for {name} v{version}")

    def get_model_by_alias(self, name: str, alias: str) -> ModelMetadata:
        """Get model version by alias."""
        mv = self._client.get_model_version_by_alias(name, alias)
        return self.get_model_metadata(name, mv.version)

    def export_model_card(self, name: str, version: int, output_path: str):
        """Export model card as JSON."""
        metadata = self.get_model_metadata(name, version)
        card = {
            "model_name": metadata.name,
            "version": metadata.version,
            "model_type": metadata.model_type.value,
            "stage": metadata.stage.value,
            "description": metadata.description,
            "metrics": metadata.metrics,
            "parameters": metadata.params,
            "tags": metadata.tags,
            "created_at": metadata.created_at,
            "updated_at": metadata.updated_at,
            "run_id": metadata.run_id,
            "artifact_path": metadata.artifact_path,
        }
        with open(output_path, 'w') as f:
            json.dump(card, f, indent=2)
        logger.info(f"Exported model card to {output_path}")

    def get_deployment_history(self, name: str) -> List[Dict[str, Any]]:
        """Get model deployment history from tags."""
        versions = self._client.get_latest_versions(name, stages=["Production", "Staging", "Archived", "None"])
        history = []
        for v in versions:
            tags = dict(v.tags)
            history.append({
                "version": v.version,
                "stage": v.current_stage,
                "deployed_at": tags.get("deployed_at"),
                "deployed_by": tags.get("deployed_by"),
                "deployment_id": tags.get("deployment_id"),
            })
        return sorted(history, key=lambda x: x.get("deployed_at", ""), reverse=True)


# Default registry instance
_default_registry: Optional[ModelRegistry] = None


def get_model_registry(mlflow_manager: Optional[MLflowManager] = None) -> ModelRegistry:
    """Get or create default model registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry(mlflow_manager)
    return _default_registry