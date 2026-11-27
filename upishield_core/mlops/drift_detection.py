"""
Data and Model Drift Detection for ScamShield AI.

Monitors feature distributions, model predictions, and performance
metrics to detect drift in production.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
from scipy.spatial.distance import jensenshannon
import json

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift detected."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class DriftSeverity(Enum):
    """Severity levels for drift alerts."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftMetric:
    """Individual drift metric result."""
    feature: str
    metric_name: str
    value: float
    threshold: float
    severity: DriftSeverity
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Complete drift detection report."""
    timestamp: str
    drift_type: DriftType
    overall_severity: DriftSeverity
    metrics: List[DriftMetric]
    summary: str
    recommendations: List[str]
    reference_period: Tuple[str, str]
    current_period: Tuple[str, str]


class DriftDetector:
    """
    Comprehensive drift detection for ML pipeline.

    Detects:
    - Feature distribution drift (data drift)
    - Prediction distribution drift
    - Model performance degradation (concept drift)
    - Statistical significance testing
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        reference_predictions: Optional[np.ndarray] = None,
        reference_labels: Optional[np.ndarray] = None,
        feature_columns: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.reference_labels = reference_labels
        self.feature_columns = feature_columns
        self.thresholds = thresholds or {
            "psi_threshold": 0.1,  # Population Stability Index
            "ks_threshold": 0.05,  # Kolmogorov-Smirnov p-value
            "js_threshold": 0.1,   # Jensen-Shannon divergence
            "prediction_drift_threshold": 0.1,
            "performance_drop_threshold": 0.05,
        }

    def set_reference(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ):
        """Set reference dataset for drift comparison."""
        self.reference_data = data.copy()
        self.reference_predictions = predictions
        self.reference_labels = labels
        if self.feature_columns is None:
            self.feature_columns = [c for c in data.columns if data[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        logger.info(f"Reference set: {len(data)} samples, {len(self.feature_columns)} features")

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> DriftReport:
        """Detect data drift using multiple statistical tests."""
        features = feature_columns or self.feature_columns
        if not features:
            features = [c for c in current_data.columns if current_data[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        metrics = []
        now = datetime.now()
        ref_start = now - timedelta(days=30)
        curr_start = now - timedelta(days=1)

        for feature in features:
            if feature not in current_data.columns:
                continue
            if self.reference_data is None or feature not in self.reference_data.columns:
                continue

            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            if len(ref_values) < 30 or len(curr_values) < 30:
                continue

            # Population Stability Index (PSI)
            psi = self._calculate_psi(ref_values, curr_values)
            metrics.append(DriftMetric(
                feature=feature,
                metric_name="psi",
                value=psi,
                threshold=self.thresholds["psi_threshold"],
                severity=self._psi_severity(psi),
                details={"ref_samples": len(ref_values), "curr_samples": len(curr_values)},
            ))

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(ref_values, curr_values)
            metrics.append(DriftMetric(
                feature=feature,
                metric_name="ks_test",
                value=ks_p,
                threshold=self.thresholds["ks_threshold"],
                severity=DriftSeverity.HIGH if ks_p < self.thresholds["ks_threshold"] else DriftSeverity.NONE,
                details={"ks_statistic": ks_stat, "p_value": ks_p},
            ))

            # Jensen-Shannon divergence
            js_div = self._calculate_js_divergence(ref_values, curr_values)
            metrics.append(DriftMetric(
                feature=feature,
                metric_name="js_divergence",
                value=js_div,
                threshold=self.thresholds["js_threshold"],
                severity=self._js_severity(js_div),
                details={},
            ))

            # Mean and std shift
            mean_shift = abs(curr_values.mean() - ref_values.mean()) / (ref_values.std() + 1e-8)
            std_shift = abs(curr_values.std() - ref_values.std()) / (ref_values.std() + 1e-8)

            metrics.append(DriftMetric(
                feature=feature,
                metric_name="mean_shift",
                value=mean_shift,
                threshold=2.0,  # 2 sigma
                severity=DriftSeverity.HIGH if mean_shift > 2 else DriftSeverity.NONE,
                details={"ref_mean": ref_values.mean(), "curr_mean": curr_values.mean()},
            ))

        overall_severity = self._aggregate_severity([m.severity for m in metrics])
        summary = self._generate_data_drift_summary(metrics, features)
        recommendations = self._generate_data_drift_recommendations(metrics)

        return DriftReport(
            timestamp=now.isoformat(),
            drift_type=DriftType.DATA_DRIFT,
            overall_severity=overall_severity,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
            reference_period=(ref_start.isoformat(), now.isoformat()),
            current_period=(curr_start.isoformat(), now.isoformat()),
        )

    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        current_labels: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """Detect prediction distribution drift."""
        if self.reference_predictions is None:
            logger.warning("No reference predictions available")
            return self._empty_report(DriftType.PREDICTION_DRIFT)

        metrics = []
        now = datetime.now()

        # PSI on predictions
        psi = self._calculate_psi(self.reference_predictions, current_predictions)
        metrics.append(DriftMetric(
            feature="predictions",
            metric_name="psi",
            value=psi,
            threshold=self.thresholds["prediction_drift_threshold"],
            severity=self._psi_severity(psi),
            details={},
        ))

        # KS test on predictions
        ks_stat, ks_p = stats.ks_2samp(self.reference_predictions, current_predictions)
        metrics.append(DriftMetric(
            feature="predictions",
            metric_name="ks_test",
            value=ks_p,
            threshold=self.thresholds["ks_threshold"],
            severity=DriftSeverity.HIGH if ks_p < self.thresholds["ks_threshold"] else DriftSeverity.NONE,
            details={"ks_statistic": ks_stat},
        ))

        # Prediction distribution moments
        ref_mean, ref_std = self.reference_predictions.mean(), self.reference_predictions.std()
        curr_mean, curr_std = current_predictions.mean(), current_predictions.std()

        metrics.append(DriftMetric(
            feature="predictions",
            metric_name="mean_shift",
            value=abs(curr_mean - ref_mean) / (ref_std + 1e-8),
            threshold=1.0,
            severity=DriftSeverity.HIGH if abs(curr_mean - ref_mean) / (ref_std + 1e-8) > 1 else DriftSeverity.NONE,
            details={"ref_mean": ref_mean, "curr_mean": curr_mean},
        ))

        # If labels available, check performance drift
        if current_labels is not None and self.reference_labels is not None:
            perf_report = self.detect_performance_drift(current_predictions, current_labels)
            metrics.extend(perf_report.metrics)

        overall_severity = self._aggregate_severity([m.severity for m in metrics])
        summary = f"Prediction drift: PSI={psi:.4f}, KS p-value={ks_p:.4f}"
        recommendations = self._generate_prediction_drift_recommendations(metrics)

        return DriftReport(
            timestamp=now.isoformat(),
            drift_type=DriftType.PREDICTION_DRIFT,
            overall_severity=overall_severity,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
            reference_period=(now - timedelta(days=30)).isoformat(),
            current_period=(now - timedelta(days=1)).isoformat(),
        )

    def detect_performance_drift(
        self,
        current_predictions: np.ndarray,
        current_labels: np.ndarray,
    ) -> DriftReport:
        """Detect model performance degradation."""
        if self.reference_predictions is None or self.reference_labels is None:
            logger.warning("No reference predictions/labels available")
            return self._empty_report(DriftType.PERFORMANCE_DRIFT)

        metrics = []
        now = datetime.now()

        # Reference performance
        ref_pred_binary = (self.reference_predictions > 0.5).astype(int)
        curr_pred_binary = (current_predictions > 0.5).astype(int)

        ref_accuracy = (ref_pred_binary == self.reference_labels).mean()
        curr_accuracy = (curr_pred_binary == current_labels).mean()

        ref_precision = self._safe_precision(self.reference_labels, ref_pred_binary)
        curr_precision = self._safe_precision(current_labels, curr_pred_binary)

        ref_recall = self._safe_recall(self.reference_labels, ref_pred_binary)
        curr_recall = self._safe_recall(current_labels, curr_pred_binary)

        ref_f1 = self._safe_f1(self.reference_labels, ref_pred_binary)
        curr_f1 = self._safe_f1(current_labels, curr_pred_binary)

        # Performance drops
        accuracy_drop = ref_accuracy - curr_accuracy
        precision_drop = ref_precision - curr_precision
        recall_drop = ref_recall - curr_recall
        f1_drop = ref_f1 - curr_f1

        metrics.append(DriftMetric(
            feature="model_performance",
            metric_name="accuracy_drop",
            value=accuracy_drop,
            threshold=self.thresholds["performance_drop_threshold"],
            severity=self._drop_severity(accuracy_drop, self.thresholds["performance_drop_threshold"]),
            details={"reference": ref_accuracy, "current": curr_accuracy},
        ))

        metrics.append(DriftMetric(
            feature="model_performance",
            metric_name="precision_drop",
            value=precision_drop,
            threshold=self.thresholds["performance_drop_threshold"],
            severity=self._drop_severity(precision_drop, self.thresholds["performance_drop_threshold"]),
            details={"reference": ref_precision, "current": curr_precision},
        ))

        metrics.append(DriftMetric(
            feature="model_performance",
            metric_name="recall_drop",
            value=recall_drop,
            threshold=self.thresholds["performance_drop_threshold"],
            severity=self._drop_severity(recall_drop, self.thresholds["performance_drop_threshold"]),
            details={"reference": ref_recall, "current": curr_recall},
        ))

        metrics.append(DriftMetric(
            feature="model_performance",
            metric_name="f1_drop",
            value=f1_drop,
            threshold=self.thresholds["performance_drop_threshold"],
            severity=self._drop_severity(f1_drop, self.thresholds["performance_drop_threshold"]),
            details={"reference": ref_f1, "current": curr_f1},
        ))

        overall_severity = self._aggregate_severity([m.severity for m in metrics])
        summary = f"Performance drift: Accuracy drop={accuracy_drop:.4f}, F1 drop={f1_drop:.4f}"
        recommendations = self._generate_performance_drift_recommendations(metrics)

        return DriftReport(
            timestamp=now.isoformat(),
            drift_type=DriftType.PERFORMANCE_DRIFT,
            overall_severity=overall_severity,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
            reference_period=(now - timedelta(days=30)).isoformat(),
            current_period=(now - timedelta(days=1)).isoformat(),
        )

    def detect_concept_drift(
        self,
        current_data: pd.DataFrame,
        current_predictions: np.ndarray,
        current_labels: np.ndarray,
    ) -> DriftReport:
        """Detect concept drift (relationship between features and target changes)."""
        # Concept drift = performance drift + prediction drift on same data distribution
        # We check if performance drops even when data distribution is similar

        data_drift_report = self.detect_data_drift(current_data)
        perf_drift_report = self.detect_performance_drift(current_predictions, current_labels)

        # If data drift is low but performance drift is high -> concept drift
        data_drift_severity = data_drift_report.overall_severity
        perf_drift_severity = perf_drift_report.overall_severity

        is_concept_drift = (
            data_drift_severity in [DriftSeverity.NONE, DriftSeverity.LOW] and
            perf_drift_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        )

        metrics = []
        metrics.append(DriftMetric(
            feature="concept",
            metric_name="concept_drift_indicator",
            value=1.0 if is_concept_drift else 0.0,
            threshold=0.5,
            severity=DriftSeverity.CRITICAL if is_concept_drift else DriftSeverity.NONE,
            details={
                "data_drift_severity": data_drift_severity.value,
                "performance_drift_severity": perf_drift_severity.value,
            },
        ))

        now = datetime.now()
        return DriftReport(
            timestamp=now.isoformat(),
            drift_type=DriftType.CONCEPT_DRIFT,
            overall_severity=DriftSeverity.CRITICAL if is_concept_drift else DriftSeverity.NONE,
            metrics=metrics,
            summary=f"Concept drift detected: {is_concept_drift}",
            recommendations=["Retrain model with recent data"] if is_concept_drift else ["No action needed"],
            reference_period=(now - timedelta(days=30)).isoformat(),
            current_period=(now - timedelta(days=1)).isoformat(),
        )

    def _calculate_psi(self, ref: np.ndarray, curr: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on reference quantiles
        try:
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(ref, quantiles)
            bin_edges = np.unique(bin_edges)

            if len(bin_edges) < 2:
                return 0.0

            ref_hist, _ = np.histogram(ref, bins=bin_edges)
            curr_hist, _ = np.histogram(curr, bins=bin_edges)

            # Normalize to percentages
            ref_pct = ref_hist / len(ref)
            curr_pct = curr_hist / len(curr)

            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)

            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            return float(psi)
        except Exception:
            return 0.0

    def _calculate_js_divergence(self, ref: np.ndarray, curr: np.ndarray, bins: int = 20) -> float:
        """Calculate Jensen-Shannon divergence."""
        try:
            # Create common bins
            min_val = min(ref.min(), curr.min())
            max_val = max(ref.max(), curr.max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)

            ref_hist, _ = np.histogram(ref, bins=bin_edges, density=True)
            curr_hist, _ = np.histogram(curr, bins=bin_edges, density=True)

            # Normalize
            ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
            curr_hist = curr_hist / (curr_hist.sum() + 1e-10)

            # Add small epsilon
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10

            js = jensenshannon(ref_hist, curr_hist)
            return float(js)
        except Exception:
            return 0.0

    def _psi_severity(self, psi: float) -> DriftSeverity:
        """Determine severity from PSI value."""
        if psi < 0.1:
            return DriftSeverity.NONE
        elif psi < 0.2:
            return DriftSeverity.LOW
        elif psi < 0.5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.HIGH

    def _js_severity(self, js: float) -> DriftSeverity:
        """Determine severity from JS divergence."""
        if js < 0.1:
            return DriftSeverity.NONE
        elif js < 0.2:
            return DriftSeverity.LOW
        elif js < 0.5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.HIGH

    def _drop_severity(self, drop: float, threshold: float) -> DriftSeverity:
        """Determine severity from performance drop."""
        if drop < threshold:
            return DriftSeverity.NONE
        elif drop < threshold * 2:
            return DriftSeverity.LOW
        elif drop < threshold * 5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.HIGH

    def _aggregate_severity(self, severities: List[DriftSeverity]) -> DriftSeverity:
        """Aggregate multiple severities."""
        order = {DriftSeverity.NONE: 0, DriftSeverity.LOW: 1, DriftSeverity.MEDIUM: 2, DriftSeverity.HIGH: 3, DriftSeverity.CRITICAL: 4}
        max_sev = max(severities, key=lambda s: order[s]) if severities else DriftSeverity.NONE
        return max_sev

    def _safe_precision(self, y_true, y_pred) -> float:
        """Safe precision calculation."""
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _safe_recall(self, y_true, y_pred) -> float:
        """Safe recall calculation."""
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _safe_f1(self, y_true, y_pred) -> float:
        """Safe F1 calculation."""
        p = self._safe_precision(y_true, y_pred)
        r = self._safe_recall(y_true, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _empty_report(self, drift_type: DriftType) -> DriftReport:
        """Create empty report."""
        now = datetime.now()
        return DriftReport(
            timestamp=now.isoformat(),
            drift_type=drift_type,
            overall_severity=DriftSeverity.NONE,
            metrics=[],
            summary="No reference data available for comparison",
            recommendations=["Set reference data first"],
            reference_period=("", ""),
            current_period=("", ""),
        )

    def _generate_data_drift_summary(self, metrics: List[DriftMetric], features: List[str]) -> str:
        """Generate summary for data drift."""
        high_drift = [m for m in metrics if m.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
        if not high_drift:
            return f"No significant data drift detected across {len(features)} features"
        return f"Data drift detected in {len(high_drift)} features: {', '.join(m.feature for m in high_drift[:5])}"

    def _generate_data_drift_recommendations(self, metrics: List[DriftMetric]) -> List[str]:
        """Generate recommendations for data drift."""
        high_drift = [m for m in metrics if m.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
        if not high_drift:
            return ["Monitor regularly"]
        return [
            f"Investigate feature drift in: {', '.join(m.feature for m in high_drift[:5])}",
            "Consider retraining model with recent data",
            "Check data pipeline for changes",
        ]

    def _generate_prediction_drift_recommendations(self, metrics: List[DriftMetric]) -> List[str]:
        """Generate recommendations for prediction drift."""
        high = [m for m in metrics if m.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
        if not high:
            return ["Monitor predictions regularly"]
        return [
            "Prediction distribution has shifted significantly",
            "Check for upstream data changes",
            "Consider model recalibration",
        ]

    def _generate_performance_drift_recommendations(self, metrics: List[DriftMetric]) -> List[str]:
        """Generate recommendations for performance drift."""
        high = [m for m in metrics if m.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
        if not high:
            return ["Performance stable"]
        return [
            "Model performance has degraded",
            "Urgent: Retrain model with latest data",
            "Investigate root cause of degradation",
        ]

    def export_report(self, report: DriftReport, path: str):
        """Export drift report to JSON."""
        data = {
            "timestamp": report.timestamp,
            "drift_type": report.drift_type.value,
            "overall_severity": report.overall_severity.value,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "reference_period": report.reference_period,
            "current_period": report.current_period,
            "metrics": [
                {
                    "feature": m.feature,
                    "metric_name": m.metric_name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "severity": m.severity.value,
                    "details": m.details,
                }
                for m in report.metrics
            ],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported drift report to {path}")


def create_drift_detector(
    reference_data: pd.DataFrame,
    reference_predictions: Optional[np.ndarray] = None,
    reference_labels: Optional[np.ndarray] = None,
    **kwargs,
) -> DriftDetector:
    """Factory function to create drift detector."""
    return DriftDetector(
        reference_data=reference_data,
        reference_predictions=reference_predictions,
        reference_labels=reference_labels,
        **kwargs,
    )