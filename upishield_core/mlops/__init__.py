"""
ScamShield AI - MLOps Module.

Provides MLflow experiment tracking, model registry,
training pipeline, drift detection, A/B testing, and statistical significance testing.
"""

from .mlflow_config import (
    MLflowConfig,
    MLflowManager,
    get_mlflow_manager,
    init_mlflow,
)

from .model_registry import (
    ModelRegistry,
    ModelStage,
    ModelType,
    ModelMetadata,
    ModelDeployment,
    get_model_registry,
)

from .training_pipeline import (
    TrainingPipeline,
    TrainingConfig,
    TrainingResult,
    ValidationThresholds,
    ModelAlgorithm,
    create_training_pipeline,
)

from .drift_detection import (
    DriftDetector,
    DriftType,
    DriftSeverity,
    DriftMetric,
    DriftReport,
)

from .statistical_significance import (
    StatisticalSignificance,
    TestResult,
    TestType,
    AlternativeHypothesis,
    PowerAnalysisResult,
    quick_proportion_test,
    quick_t_test,
    quick_mann_whitney,
    quick_bootstrap,
)

from .ab_testing import (
    ABTestingFramework,
    ExperimentConfig,
    ExperimentVariant,
    ExperimentMetric,
    ExperimentResult,
    ExperimentStatus,
    AllocationStrategy,
    MetricType,
    create_fraud_detection_experiment,
    get_ab_framework,
)

__all__ = [
    # MLflow Config
    "MLflowConfig",
    "MLflowManager",
    "get_mlflow_manager",
    "init_mlflow",
    # Model Registry
    "ModelRegistry",
    "ModelStage",
    "ModelType",
    "ModelMetadata",
    "ModelDeployment",
    "get_model_registry",
    # Training Pipeline
    "TrainingPipeline",
    "TrainingConfig",
    "TrainingResult",
    "ValidationThresholds",
    "ModelAlgorithm",
    "create_training_pipeline",
    # Drift Detection
    "DriftDetector",
    "DriftType",
    "DriftSeverity",
    "DriftMetric",
    "DriftReport",
    # Statistical Significance
    "StatisticalSignificance",
    "TestResult",
    "TestType",
    "AlternativeHypothesis",
    "PowerAnalysisResult",
    "quick_proportion_test",
    "quick_t_test",
    "quick_mann_whitney",
    "quick_bootstrap",
    # A/B Testing
    "ABTestingFramework",
    "ExperimentConfig",
    "ExperimentVariant",
    "ExperimentMetric",
    "ExperimentResult",
    "ExperimentStatus",
    "AllocationStrategy",
    "MetricType",
    "create_fraud_detection_experiment",
    "get_ab_framework",
]