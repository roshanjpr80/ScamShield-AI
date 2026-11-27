"""
MLflow Configuration for Experiment Tracking.

Provides centralized MLflow setup for tracking experiments, models,
and metrics across the fraud detection pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, Run

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """MLflow configuration settings."""
    tracking_uri: str = "file:./mlruns"
    registry_uri: Optional[str] = None
    experiment_name: str = "scamshield-fraud-detection"
    artifact_location: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class MLflowManager:
    """
    Manages MLflow experiment tracking for the fraud detection pipeline.

    Features:
    - Experiment creation and management
    - Run logging with parameters, metrics, artifacts
    - Model registration and versioning
    - Artifact storage and retrieval
    - Integration with training pipelines
    """

    def __init__(self, config: Optional[MLflowConfig] = None):
        self.config = config or MLflowConfig()
        self._client: Optional[MlflowClient] = None
        self._experiment: Optional[Experiment] = None
        self._current_run: Optional[Run] = None
        self._setup()

    def _setup(self):
        """Initialize MLflow tracking."""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        if self.config.registry_uri:
            mlflow.set_registry_uri(self.config.registry_uri)

        self._client = MlflowClient(tracking_uri=self.config.tracking_uri)

        # Create or get experiment
        self._experiment = self._get_or_create_experiment()

        logger.info(f"MLflow initialized: {self.config.tracking_uri}")
        logger.info(f"Experiment: {self.config.experiment_name} (ID: {self._experiment.experiment_id})")

    def _get_or_create_experiment(self) -> Experiment:
        """Get existing experiment or create new one."""
        experiment = self._client.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            experiment_id = self._client.create_experiment(
                name=self.config.experiment_name,
                artifact_location=self.config.artifact_location,
                tags=self.config.tags,
            )
            experiment = self._client.get_experiment(experiment_id)
        return experiment

    @property
    def experiment_id(self) -> str:
        """Get current experiment ID."""
        return self._experiment.experiment_id

    @property
    def client(self) -> MlflowClient:
        """Get MLflow client."""
        return self._client

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> Run:
        """Start a new MLflow run."""
        run_tags = {
            "project": "scamshield",
            "timestamp": datetime.now().isoformat(),
            **(tags or {}),
        }

        run = self._client.create_run(
            experiment_id=self.experiment_id,
            tags=run_tags,
        )

        self._current_run = run
        mlflow.start_run(run_id=run.info.run_id, nested=nested)

        if run_name:
            mlflow.set_tag("mlflow.runName", run_name)

        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if self._current_run:
            self._client.set_terminated(self._current_run.info.run_id, status)
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self._current_run.info.run_id} with status: {status}")
            self._current_run = None

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run."""
        if not self._current_run:
            logger.warning("No active run to log parameters")
            return

        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.debug(f"Logged {len(params)} parameters")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics to current run."""
        if not self._current_run:
            logger.warning("No active run to log metrics")
            return

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact to current run."""
        if not self._current_run:
            logger.warning("No active run to log artifact")
            return

        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log all artifacts in directory."""
        if not self._current_run:
            logger.warning("No active run to log artifacts")
            return

        mlflow.log_artifacts(local_dir, artifact_path)
        logger.debug(f"Logged artifacts from: {local_dir}")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        flavor: Optional[str] = None,
        registered_model_name: Optional[str] = None,
        **kwargs,
    ):
        """Log model to current run and optionally register."""
        if not self._current_run:
            logger.warning("No active run to log model")
            return

        # Add trusted types for sklearn models to avoid security warnings
        if flavor == "sklearn" or (flavor is None and hasattr(model, '__class__') and 'sklearn' in str(model.__class__)):
            kwargs.setdefault('skops_trusted_types', ['sklearn.tree._tree.Tree'])

        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_model_name, **kwargs)
        elif flavor == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path, registered_model_name=registered_model_name, **kwargs)
        elif flavor == "tensorflow":
            mlflow.tensorflow.log_model(model, artifact_path, registered_model_name=registered_model_name, **kwargs)
        elif flavor == "onnx":
            mlflow.onnx.log_model(model, artifact_path, registered_model_name=registered_model_name, **kwargs)
        else:
            mlflow.pyfunc.log_model(artifact_path, python_model=model, **kwargs)

        logger.info(f"Logged model to {artifact_path}")

    def load_model(self, model_uri: str) -> Any:
        """Load model from MLflow."""
        return mlflow.pyfunc.load_model(model_uri)

    def search_runs(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> List[Run]:
        """Search runs in experiment."""
        return self._client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["attributes.start_time DESC"],
            max_results=max_results,
        )

    def get_best_run(
        self,
        metric: str,
        ascending: bool = False,
    ) -> Optional[Run]:
        """Get best run by metric."""
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )
        return runs[0] if runs else None

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Register model in MLflow Model Registry."""
        mv = mlflow.register_model(model_uri, name, tags=tags)
        logger.info(f"Registered model: {name} version {mv.version}")
        return mv

    def transition_model_stage(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing: bool = True,
    ):
        """Transition model version to stage (Staging, Production, Archived)."""
        self._client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Transitioned {name} v{version} to {stage}")

    def get_latest_model_version(
        self,
        name: str,
        stages: Optional[List[str]] = None,
    ) -> Optional[int]:
        """Get latest model version for given stages."""
        versions = self._client.get_latest_versions(name, stages=stages or ["Production", "Staging"])
        return versions[0].version if versions else None

    def log_dataset(self, dataset_path: str, name: str, artifact_path: str = "datasets"):
        """Log dataset as artifact."""
        if not self._current_run:
            logger.warning("No active run to log dataset")
            return

        mlflow.log_artifact(dataset_path, f"{artifact_path}/{name}")
        logger.info(f"Logged dataset: {name}")

    def set_tags(self, tags: Dict[str, str]):
        """Set tags on current run."""
        if not self._current_run:
            logger.warning("No active run to set tags")
            return

        for key, value in tags.items():
            mlflow.set_tag(key, value)

    def log_figure(self, figure, artifact_file: str):
        """Log matplotlib figure."""
        if not self._current_run:
            logger.warning("No active run to log figure")
            return

        mlflow.log_figure(figure, artifact_file)


# Default MLflow manager instance
_default_manager: Optional[MLflowManager] = None


def get_mlflow_manager(config: Optional[MLflowConfig] = None) -> MLflowManager:
    """Get or create default MLflow manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = MLflowManager(config)
    return _default_manager


def init_mlflow(
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "scamshield-fraud-detection",
    **kwargs,
) -> MLflowManager:
    """Initialize MLflow with custom configuration."""
    global _default_manager
    config = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        **kwargs,
    )
    _default_manager = MLflowManager(config)
    return _default_manager