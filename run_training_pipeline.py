"""
Run the complete MLflow training pipeline with Feast Feature Store integration.

This script:
1. Creates sample training data
2. Runs the FeastTrainingPipeline with MLflow tracking
3. Sets up drift detection reference data
4. Registers the model in MLflow Model Registry
"""

import asyncio
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from upishield_core.mlops.training_with_feast import (
    FeastTrainingPipeline,
    TrainingConfig,
    ModelAlgorithm,
    ValidationThresholds,
    create_feast_training_pipeline,
)
from upishield_core.mlops.drift_detection import create_drift_detector, DriftDetector
from upishield_core.mlops.mlflow_config import init_mlflow
from upishield_core.features.materialize import run_materialization_pipeline, create_sample_data
from upishield_core.features.online_store import get_online_store, OnlineStoreConfig, StoreBackend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_training_entity_df(data_dir: str, n_samples: int = 1000) -> pd.DataFrame:
    """
    Create entity DataFrame for Feast historical feature retrieval.
    This simulates the transaction log that would be used for training.
    """
    # Load the transaction data we materialized
    transactions_path = Path(data_dir) / "transactions.parquet"
    if transactions_path.exists():
        df = pd.read_parquet(transactions_path)
    else:
        # Generate synthetic data
        create_sample_data(data_dir)
        df = pd.read_parquet(transactions_path)

    # Sample for training
    df = df.sample(n=min(n_samples, len(df)), random_state=42)

    # Create entity DataFrame with required columns
    # Generate labels based on actual transaction features (high amount, night time, etc.)
    is_high_amount = df["amount"] > df["amount"].quantile(0.95)
    is_night = df["is_night"] == 1
    is_weekend = df["is_weekend"] == 1

    # Fraud more likely for high amount, night, weekend transactions
    fraud_prob = 0.01  # base rate
    fraud_prob += is_high_amount.astype(int) * 0.3
    fraud_prob += is_night.astype(int) * 0.15
    fraud_prob += is_weekend.astype(int) * 0.1
    fraud_prob = np.clip(fraud_prob, 0.005, 0.5)

    labels = np.random.binomial(1, fraud_prob)

    entity_df = pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "event_timestamp": pd.to_datetime(df["txn_hour"].apply(lambda h: datetime.now().replace(hour=h, minute=0, second=0, microsecond=0))),
        "label": labels,
    })

    return entity_df


async def run_full_training():
    """Run the complete training pipeline."""

    print("=" * 70)
    print("SCAMSHIELD AI - MLFLOW TRAINING PIPELINE WITH FEAST")
    print("=" * 70)

    # 1. Initialize MLflow
    print("\n1. Initializing MLflow...")
    mlflow_manager = init_mlflow(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="scamshield-fraud-detection-training",
    )
    print(f"   MLflow tracking URI: sqlite:///mlflow.db")
    print(f"   Experiment: scamshield-fraud-detection-training")

    # 2. Run materialization to populate Feast offline/online stores
    print("\n2. Running feature materialization...")
    materialization_result = run_materialization_pipeline(
        data_dir="./feature_repo/data",
        online_store_backend="memory",
        ttl_seconds=86400 * 90,
    )
    print(f"   Materialized: {materialization_result['materialization_results']}")
    print(f"   Verification: {materialization_result['verification']['verified']}")

    # 3. Create entity DataFrame for training
    print("\n3. Creating training entity DataFrame...")
    entity_df = create_training_entity_df("./feature_repo/data", n_samples=500)
    print(f"   Training samples: {len(entity_df)}")
    print(f"   Fraud rate: {entity_df['label'].mean():.2%}")

    # 4. Create and run training pipeline
    print("\n4. Creating FeastTrainingPipeline...")
    config = TrainingConfig(
        algorithm=ModelAlgorithm.RANDOM_FOREST,  # Use RandomForest to avoid XGBoost memory issues
        test_size=0.2,
        validation_size=0.1,
        random_state=42,
        cross_validation_folds=3,
        stratify=True,
        feature_service="model_training",
        entity_column="transaction_id",
        timestamp_column="event_timestamp",
        hyperparameters={
            "n_estimators": 50,
            "max_depth": 8,
        }
    )

    pipeline = FeastTrainingPipeline(
        config=config,
        mlflow_manager=mlflow_manager,
        validation_thresholds=ValidationThresholds(
            min_accuracy=0.80,
            min_precision=0.75,
            min_recall=0.75,
            min_f1=0.75,
            min_roc_auc=0.80,
            min_pr_auc=0.70,
        )
    )

    # 5. Run full pipeline
    print("\n5. Running full training pipeline...")
    print("   This will: fetch from Feast -> train -> validate -> register")

    try:
        result = pipeline.run_full_pipeline(
            entity_df=entity_df,
            model_name="scamshield-fraud-ensemble",
            model_type="fraud_detection",
            description="Ensemble fraud detection model trained with Feast features",
            run_name=f"ensemble_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        print(f"\n   Training completed!")
        print(f"   Run ID: {result.run_id}")
        print(f"   Model artifact: {result.artifact_path}")
        print(f"   Training time: {result.training_time_ms:.0f}ms")
        print(f"   Features used: {len(result.feature_columns)}")
        print(f"   Feature service: {result.feature_service}")

        print(f"\n   Validation Metrics:")
        for k, v in result.metrics.items():
            print(f"     {k}: {v:.4f}")

        print(f"\n   Cross-Validation Scores:")
        for k, v in result.cv_scores.items():
            print(f"     {k}: mean={np.mean(v):.4f}, std={np.std(v):.4f}")

        if result.feature_importance:
            print(f"\n   Top 10 Feature Importance:")
            sorted_importance = sorted(
                result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for feat, imp in sorted_importance:
                print(f"     {feat}: {imp:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # 6. Set up drift detection reference
    print("\n6. Setting up drift detection reference...")

    # Fetch training data again for drift reference
    training_df = pipeline.fetch_training_data(entity_df)
    X_train, X_val, y_train, y_val, X_test, y_test = pipeline.prepare_data(training_df)

    # Get predictions on reference data
    if pipeline._model:
        ref_predictions = pipeline._model.predict_proba(X_test)[:, 1]
        ref_labels = y_test.values

        drift_detector = create_drift_detector(
            reference_data=X_test,
            reference_predictions=ref_predictions,
            reference_labels=ref_labels,
            feature_columns=pipeline.config.feature_columns,
        )

        # Save drift detector reference
        drift_detector.set_reference(X_test, ref_predictions, ref_labels)
        print(f"   Drift detector reference set: {len(X_test)} samples")
        print(f"   Features monitored: {len(pipeline.config.feature_columns)}")

    # 7. Test drift detection with current data
    print("\n7. Testing drift detection...")
    if pipeline._model:
        # Use validation data as "current" data
        curr_predictions = pipeline._model.predict_proba(X_val)[:, 1]
        curr_labels = y_val.values

        drift_report = drift_detector.detect_data_drift(X_val)
        print(f"   Data drift severity: {drift_report.overall_severity.value}")
        print(f"   Summary: {drift_report.summary}")

        pred_drift_report = drift_detector.detect_prediction_drift(curr_predictions, curr_labels)
        print(f"   Prediction drift severity: {pred_drift_report.overall_severity.value}")
        print(f"   Summary: {pred_drift_report.summary}")

        perf_drift_report = drift_detector.detect_performance_drift(curr_predictions, curr_labels)
        print(f"   Performance drift severity: {perf_drift_report.overall_severity.value}")
        print(f"   Summary: {perf_drift_report.summary}")

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Check MLflow UI: mlflow ui --backend-store-uri file:./mlruns")
    print("  - View experiments, runs, and registered models")
    print("  - Set up scheduled drift detection jobs")
    print("  - Deploy model to production")

    return result


if __name__ == "__main__":
    asyncio.run(run_full_training())