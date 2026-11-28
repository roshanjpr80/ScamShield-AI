"""
MLflow Training Pipeline with Feast Feature Store Integration.

Extends the training pipeline to fetch training data from Feast offline store,
and register trained models with feature metadata.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.base import BaseEstimator

from .mlflow_config import MLflowManager, get_mlflow_manager
from .model_registry import ModelRegistry, ModelType, ModelStage, get_model_registry
from upishield_core.features.feast_store import FeastFeatureStore, get_feature_store, ALL_FEATURES

logger = logging.getLogger(__name__)

# Lazy imports
_mlflow = None
_mlflow_sklearn = None

def _get_mlflow():
    global _mlflow, _mlflow_sklearn
    if _mlflow is None:
        import mlflow
        import mlflow.sklearn
        _mlflow = mlflow
        _mlflow_sklearn = mlflow.sklearn
    return _mlflow, _mlflow_sklearn


class ModelAlgorithm(Enum):
    """Supported ML algorithms."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    LOGISTIC_REGRESSION = "logistic_regression"
    ENSEMBLE = "ensemble"


@dataclass
class TrainingConfig:
    """Configuration for training pipeline with Feast integration."""
    algorithm: ModelAlgorithm = ModelAlgorithm.ENSEMBLE
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cross_validation_folds: int = 5
    stratify: bool = True
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_columns: Optional[List[str]] = None
    target_column: str = "label"
    class_weight: Optional[Dict] = None

    # Feast-specific config
    feature_service: str = "model_training"  # Feast feature service to use
    entity_column: str = "transaction_id"  # Entity column in entity_df
    timestamp_column: str = "event_timestamp"  # Timestamp column for point-in-time


@dataclass
class TrainingResult:
    """Result of training run."""
    model: Any
    metrics: Dict[str, float]
    cv_scores: Dict[str, List[float]]
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    run_id: str = ""
    artifact_path: str = ""
    training_time_ms: float = 0.0
    feature_columns: List[str] = field(default_factory=list)
    feature_service: str = ""


@dataclass
class ValidationThresholds:
    """Minimum thresholds for model validation."""
    min_accuracy: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.80
    min_f1: float = 0.80
    min_roc_auc: float = 0.85
    min_pr_auc: float = 0.80
    max_false_positive_rate: float = 0.15


class FeastTrainingPipeline:
    """
    End-to-end training pipeline with Feast Feature Store integration.

    Features:
    - Fetch training data from Feast offline store using feature services
    - Automated data splitting and validation
    - Cross-validation with multiple metrics
    - Hyperparameter logging
    - Model validation against thresholds
    - Automatic model registration with feature metadata
    - Feature importance extraction
    - Training visualization logging
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        mlflow_manager: Optional[MLflowManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        feast_store: Optional[FeastFeatureStore] = None,
        validation_thresholds: Optional[ValidationThresholds] = None,
    ):
        self.config = config or TrainingConfig()
        self.mlflow = mlflow_manager or get_mlflow_manager()
        self.registry = model_registry or get_model_registry(self.mlflow)
        self.feast_store = feast_store or get_feature_store()
        self.thresholds = validation_thresholds or ValidationThresholds()
        self._model: Optional[BaseEstimator] = None

    def fetch_training_data(
        self,
        entity_df: Union[pd.DataFrame, str],
    ) -> pd.DataFrame:
        """
        Fetch training data from Feast offline store.

        Args:
            entity_df: Entity DataFrame with entity_id and timestamp columns,
                       or path to Parquet file

        Returns:
            DataFrame with features joined from Feast
        """
        logger.info("Fetching training data from Feast offline store...")

        # Ensure Feast store is initialized
        if not self.feast_store._initialized:
            self.feast_store.initialize()
            self.feast_store.apply()
            self.feast_store.create_feature_services()

        # Get historical features using feature service
        training_df = self.feast_store.get_historical_features(
            entity_df=entity_df,
            feature_service=self.config.feature_service,
        )

        logger.info(f"Fetched training data: {len(training_df)} rows, {len(training_df.columns)} columns")
        return training_df

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare and split data into train/val/test sets."""
        # Select features
        if self.config.feature_columns:
            X = df[self.config.feature_columns].copy()
        else:
            # Exclude target and non-feature columns
            exclude_cols = [
                self.config.target_column, "id", "timestamp", "transaction_id",
                "event_timestamp", "created_timestamp", "user_id", "merchant_id",
                "device_id", "payer_vpa", "payee_vpa"
            ]
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            X = df[feature_cols].copy()
            self.config.feature_columns = feature_cols

        y = df[self.config.target_column].copy()

        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))

        # Convert categorical columns to numeric using label encoding
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = X[col].fillna("unknown")
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        # Store encoders for later use
        self._label_encoders = label_encoders

        # First split: train+val vs test
        if self.config.stratify:
            stratify_col = y
        else:
            stratify_col = None

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify_col,
        )

        # Second split: train vs val
        val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        if self.config.stratify:
            stratify_col = y_temp
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=stratify_col,
        )

        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, y_train, y_val, X_test, y_test

    def _create_model(self) -> BaseEstimator:
        """Create model instance based on algorithm."""
        algo = self.config.algorithm
        params = self.config.hyperparameters.copy()

        if algo == ModelAlgorithm.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier
            defaults = {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": self.config.class_weight or "balanced",
                "random_state": self.config.random_state,
                "n_jobs": 1,
            }
            defaults.update(params)
            return RandomForestClassifier(**defaults)

        elif algo == ModelAlgorithm.XGBOOST:
            try:
                from xgboost import XGBClassifier
            except ImportError:
                logger.warning("XGBoost not available, falling back to RandomForest")
                return self._create_model(ModelAlgorithm.RANDOM_FOREST)
            defaults = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 1.0,
                "random_state": self.config.random_state,
                "n_jobs": 1,
                "eval_metric": "logloss",
            }
            defaults.update(params)
            return XGBClassifier(**defaults)

        elif algo == ModelAlgorithm.LIGHTGBM:
            try:
                from lightgbm import LGBMClassifier
            except ImportError:
                logger.warning("LightGBM not available, falling back to RandomForest")
                return self._create_model(ModelAlgorithm.RANDOM_FOREST)
            defaults = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "class_weight": self.config.class_weight or "balanced",
                "random_state": self.config.random_state,
                "n_jobs": 1,
            }
            defaults.update(params)
            return LGBMClassifier(**defaults)

        elif algo == ModelAlgorithm.CATBOOST:
            try:
                from catboost import CatBoostClassifier
            except ImportError:
                logger.warning("CatBoost not available, falling back to RandomForest")
                return self._create_model(ModelAlgorithm.RANDOM_FOREST)
            defaults = {
                "iterations": 200,
                "depth": 6,
                "learning_rate": 0.1,
                "random_state": self.config.random_state,
                "verbose": False,
            }
            defaults.update(params)
            return CatBoostClassifier(**defaults)

        elif algo == ModelAlgorithm.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression
            defaults = {
                "C": 1.0,
                "class_weight": self.config.class_weight or "balanced",
                "max_iter": 1000,
                "random_state": self.config.random_state,
                "n_jobs": 1,
            }
            defaults.update(params)
            return LogisticRegression(**defaults)

        elif algo == ModelAlgorithm.ENSEMBLE:
            from sklearn.ensemble import VotingClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            try:
                from xgboost import XGBClassifier
                has_xgb = True
            except ImportError:
                has_xgb = False

            estimators = [
                ("rf", RandomForestClassifier(
                    n_estimators=150, max_depth=12,
                    class_weight="balanced", random_state=self.config.random_state, n_jobs=1
                )),
                ("lr", LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=1000,
                    random_state=self.config.random_state, n_jobs=1
                )),
            ]
            if has_xgb:
                estimators.append(("xgb", XGBClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.1,
                    random_state=self.config.random_state, n_jobs=1, eval_metric="logloss"
                )))

            return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)

        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        run_name: Optional[str] = None,
    ) -> TrainingResult:
        """Train model with validation."""
        start_time = datetime.now()
        self._model = self._create_model()

        # Start MLflow run
        run = self.mlflow.start_run(run_name=run_name or f"train_{self.config.algorithm.value}")

        try:
            # Log parameters
            params = {
                "algorithm": self.config.algorithm.value,
                "test_size": self.config.test_size,
                "validation_size": self.config.validation_size,
                "random_state": self.config.random_state,
                "cv_folds": self.config.cross_validation_folds,
                "n_features": len(self.config.feature_columns) if self.config.feature_columns else X_train.shape[1],
                "n_samples_train": len(X_train),
                "feature_service": self.config.feature_service,
                **self.config.hyperparameters,
            }
            self.mlflow.log_params(params)

            # Log feature columns as artifact
            if self.config.feature_columns:
                fi_path = "/tmp/feature_columns.json"
                with open(fi_path, 'w') as f:
                    json.dump(self.config.feature_columns, f, indent=2)
                self.mlflow.log_artifact(fi_path, "features")

            # Train model
            self._model.fit(X_train, y_train)

            # Validation predictions
            val_metrics = {}
            if X_val is not None and y_val is not None:
                val_metrics = self._evaluate_model(X_val, y_val, prefix="val_")
                self.mlflow.log_metrics(val_metrics)

            # Cross-validation
            cv_scores = self._cross_validate(X_train, y_train)
            cv_metrics = {f"cv_{k}_mean": np.mean(v) for k, v in cv_scores.items()}
            cv_metrics.update({f"cv_{k}_std": np.std(v) for k, v in cv_scores.items()})
            self.mlflow.log_metrics(cv_metrics)

            # Feature importance
            feature_importance = self._extract_feature_importance(X_train.columns)

            # Training time
            training_time = (datetime.now() - start_time).total_seconds() * 1000

            # Log model
            artifact_path = "model"
            self.mlflow.log_model(
                self._model,
                artifact_path,
                flavor="sklearn",
            )

            # Log feature importance as artifact
            if feature_importance:
                fi_path = "/tmp/feature_importance.json"
                with open(fi_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                self.mlflow.log_artifact(fi_path, "feature_importance")

            # Log feature definitions from Feast
            self._log_feast_features()

            # End run
            self.mlflow.end_run()

            result = TrainingResult(
                model=self._model,
                metrics=val_metrics,
                cv_scores=cv_scores,
                feature_importance=feature_importance,
                run_id=run.info.run_id,
                artifact_path=f"runs:/{run.info.run_id}/{artifact_path}",
                training_time_ms=training_time,
                feature_columns=self.config.feature_columns or [],
                feature_service=self.config.feature_service,
            )

            logger.info(f"Training completed in {training_time:.0f}ms")
            logger.info(f"Validation metrics: {val_metrics}")
            return result

        except Exception as e:
            self.mlflow.end_run(status="FAILED")
            raise

    def _log_feast_features(self):
        """Log Feast feature definitions to MLflow."""
        try:
            # Export feature definitions
            export_path = "/tmp/feast_features.json"
            self.feast_store.export_feature_definitions(export_path)
            self.mlflow.log_artifact(export_path, "feast_features")

            # Log feature services
            services = self.feast_store.list_feature_services()
            services_path = "/tmp/feast_services.json"
            with open(services_path, 'w') as f:
                json.dump(services, f, indent=2)
            self.mlflow.log_artifact(services_path, "feast_features")

            logger.info("Logged Feast feature definitions to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log Feast features: {e}")

    def _evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Evaluate model on given data."""
        y_pred = self._model.predict(X)
        y_proba = self._model.predict_proba(X)[:, 1] if hasattr(self._model, "predict_proba") else None

        metrics = {
            f"{prefix}accuracy": accuracy_score(y, y_pred),
            f"{prefix}precision": precision_score(y, y_pred, zero_division=0),
            f"{prefix}recall": recall_score(y, y_pred, zero_division=0),
            f"{prefix}f1": f1_score(y, y_pred, zero_division=0),
        }

        if y_proba is not None:
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y, y_proba)
            metrics[f"{prefix}pr_auc"] = average_precision_score(y, y_proba)

        # Store confusion matrix and classification report
        self._last_cm = confusion_matrix(y, y_pred)
        self._last_cr = classification_report(y, y_pred, output_dict=True)

        return metrics

    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.config.cross_validation_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
        cv_results = {}

        for score in scoring:
            try:
                scores = cross_val_score(self._model, X, y, cv=cv, scoring=score, n_jobs=-1)
                cv_results[score] = scores.tolist()
            except Exception as e:
                logger.warning(f"CV failed for {score}: {e}")
                cv_results[score] = [0.0] * self.config.cross_validation_folds

        return cv_results

    def _extract_feature_importance(self, feature_names: pd.Index) -> Optional[Dict[str, float]]:
        """Extract feature importance from trained model."""
        if self._model is None:
            return None

        importance = None
        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            importance = np.abs(self._model.coef_[0])
        elif hasattr(self._model, "estimators_"):  # VotingClassifier
            importances = []
            for name, est in self._model.named_estimators_.items():
                if hasattr(est, "feature_importances_"):
                    importances.append(est.feature_importances_)
                elif hasattr(est, "coef_"):
                    importances.append(np.abs(est.coef_[0]))
            if importances:
                importance = np.mean(importances, axis=0)

        if importance is not None:
            return dict(zip(feature_names, importance.tolist()))
        return None

    def validate_model(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate model against thresholds."""
        failures = []

        if metrics.get("val_accuracy", 0) < self.thresholds.min_accuracy:
            failures.append(f"Accuracy {metrics.get('val_accuracy', 0):.4f} < {self.thresholds.min_accuracy}")

        if metrics.get("val_precision", 0) < self.thresholds.min_precision:
            failures.append(f"Precision {metrics.get('val_precision', 0):.4f} < {self.thresholds.min_precision}")

        if metrics.get("val_recall", 0) < self.thresholds.min_recall:
            failures.append(f"Recall {metrics.get('val_recall', 0):.4f} < {self.thresholds.min_recall}")

        if metrics.get("val_f1", 0) < self.thresholds.min_f1:
            failures.append(f"F1 {metrics.get('val_f1', 0):.4f} < {self.thresholds.min_f1}")

        if metrics.get("val_roc_auc", 0) < self.thresholds.min_roc_auc:
            failures.append(f"ROC-AUC {metrics.get('val_roc_auc', 0):.4f} < {self.thresholds.min_roc_auc}")

        if metrics.get("val_pr_auc", 0) < self.thresholds.min_pr_auc:
            failures.append(f"PR-AUC {metrics.get('val_pr_auc', 0):.4f} < {self.thresholds.min_pr_auc}")

        return len(failures) == 0, failures

    def register_and_promote(
        self,
        training_result: TrainingResult,
        model_name: str,
        model_type: ModelType,
        description: str = "",
        auto_promote: bool = True,
    ) -> bool:
        """Register model and optionally promote to staging."""
        # Validate
        passed, failures = self.validate_model(training_result.metrics)
        if not passed:
            logger.warning(f"Model validation failed: {failures}")
            return False

        # Register with feature metadata
        mv = self.registry.register_model(
            model_uri=training_result.artifact_path,
            name=model_name,
            model_type=model_type,
            description=description,
            metrics=training_result.metrics,
            params=self.config.hyperparameters,
            run_id=training_result.run_id,
            feature_columns=training_result.feature_columns,
            feature_service=training_result.feature_service,
        )

        # Promote to staging
        if auto_promote:
            self.registry.promote_model(model_name, mv.version, ModelStage.STAGING)

        return True

    def run_full_pipeline(
        self,
        entity_df: Union[pd.DataFrame, str],
        model_name: str,
        model_type: ModelType,
        description: str = "",
        run_name: Optional[str] = None,
    ) -> TrainingResult:
        """Run complete training pipeline: fetch from Feast -> train -> validate -> register."""
        # Fetch training data from Feast
        df = self.fetch_training_data(entity_df)

        # Prepare data
        X_train, X_val, y_train, y_val, X_test, y_test = self.prepare_data(df)

        # Train
        result = self.train(X_train, y_train, X_val, y_val, run_name)

        # Final evaluation on test set
        test_metrics = self._evaluate_model(X_test, y_test, prefix="test_")
        self.mlflow.log_metrics(test_metrics)
        logger.info(f"Test metrics: {test_metrics}")

        # Register
        self.register_and_promote(result, model_name, model_type, description)

        return result


def create_feast_training_pipeline(
    algorithm: ModelAlgorithm = ModelAlgorithm.ENSEMBLE,
    feature_service: str = "model_training",
    **kwargs,
) -> FeastTrainingPipeline:
    """Factory function to create Feast training pipeline."""
    config = TrainingConfig(algorithm=algorithm, feature_service=feature_service, **kwargs)
    return FeastTrainingPipeline(config=config)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Example usage
    print("FeastTrainingPipeline ready for use")
    print("Use create_feast_training_pipeline() to create an instance")