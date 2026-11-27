"""
A/B Testing Framework for ScamShield AI.

Provides comprehensive A/B testing infrastructure for model comparison,
including experiment management, traffic allocation, and automated analysis.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading

import numpy as np
import pandas as pd

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

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ARCHIVED = "archived"


class AllocationStrategy(Enum):
    """Traffic allocation strategies."""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    STICKY = "sticky"
    CONTEXTUAL = "contextual"


class MetricType(Enum):
    """Types of metrics to track."""
    CONVERSION_RATE = "conversion_rate"
    MEAN_VALUE = "mean_value"
    LATENCY = "latency"
    REVENUE = "revenue"
    CUSTOM = "custom"


@dataclass
class ExperimentVariant:
    """Definition of an experiment variant."""
    name: str
    model_uri: str
    traffic_allocation: float = 0.5  # Fraction of traffic (0-1)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentMetric:
    """Metric definition for experiment."""
    name: str
    metric_type: MetricType
    higher_is_better: bool = True
    description: str = ""
    transformation: Optional[Callable] = None  # Optional transformation function


@dataclass
class ExperimentConfig:
    """Configuration for A/B experiment."""
    name: str
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    allocation_strategy: AllocationStrategy = AllocationStrategy.RANDOM
    randomization_unit: str = "user_id"  # user_id, session_id, request_id
    minimum_sample_size: int = 1000
    maximum_duration_days: int = 30
    significance_level: float = 0.05
    power: float = 0.8
    minimum_detectable_effect: float = 0.05
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        total_allocation = sum(v.traffic_allocation for v in self.variants)
        if abs(total_allocation - 1.0) > 1e-6:
            raise ValueError(f"Traffic allocations must sum to 1.0, got {total_allocation}")


@dataclass
class ExperimentResult:
    """Results of an A/B experiment."""
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    sample_sizes: Dict[str, int]
    metric_results: Dict[str, Dict[str, Any]]
    statistical_tests: Dict[str, TestResult]
    winner: Optional[str] = None
    confidence: float = 0.0
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingFramework:
    """
    Comprehensive A/B testing framework for model experimentation.

    Features:
    - Experiment lifecycle management
    - Traffic allocation with multiple strategies
    - Real-time metric tracking
    - Automated statistical analysis
    - Sequential testing with early stopping
    - Multiple comparison correction
    - Power analysis and sample size calculation
    - Results export and visualization data
    """

    def __init__(
        self,
        storage_path: str = "./ab_experiments",
        default_alpha: float = 0.05,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.default_alpha = default_alpha
        self.stats = StatisticalSignificance(default_alpha)
        self._experiments: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._load_experiments()

    def _load_experiments(self):
        """Load experiments from storage."""
        for exp_file in self.storage_path.glob("*.json"):
            try:
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                exp_id = data.get("experiment_id")
                if exp_id:
                    self._experiments[exp_id] = data
            except Exception as e:
                logger.warning(f"Failed to load experiment {exp_file}: {e}")

    def _save_experiment(self, experiment_id: str):
        """Save experiment to storage."""
        if experiment_id not in self._experiments:
            return
        filepath = self.storage_path / f"{experiment_id}.json"
        with open(filepath, 'w') as f:
            json.dump(self._experiments[experiment_id], f, indent=2, default=str)

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B experiment."""
        experiment_id = str(uuid.uuid4())[:8]

        experiment = {
            "experiment_id": experiment_id,
            "name": config.name,
            "config": {
                "variants": [
                    {
                        "name": v.name,
                        "model_uri": v.model_uri,
                        "traffic_allocation": v.traffic_allocation,
                        "description": v.description,
                        "metadata": v.metadata,
                    }
                    for v in config.variants
                ],
                "metrics": [
                    {
                        "name": m.name,
                        "metric_type": m.metric_type.value,
                        "higher_is_better": m.higher_is_better,
                        "description": m.description,
                    }
                    for m in config.metrics
                ],
                "allocation_strategy": config.allocation_strategy.value,
                "randomization_unit": config.randomization_unit,
                "minimum_sample_size": config.minimum_sample_size,
                "maximum_duration_days": config.maximum_duration_days,
                "significance_level": config.significance_level,
                "power": config.power,
                "minimum_detectable_effect": config.minimum_detectable_effect,
                "description": config.description,
                "tags": config.tags,
            },
            "status": ExperimentStatus.DRAFT.value,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "ended_at": None,
            "data": {v.name: {"observations": [], "metrics": {}} for v in config.variants},
            "results": None,
        }

        with self._lock:
            self._experiments[experiment_id] = experiment
            self._save_experiment(experiment_id)

        logger.info(f"Created experiment {experiment_id}: {config.name}")
        return experiment_id

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            exp = self._experiments[experiment_id]
            if exp["status"] != ExperimentStatus.DRAFT.value:
                logger.warning(f"Experiment {experiment_id} is not in DRAFT status")
                return False

            exp["status"] = ExperimentStatus.RUNNING.value
            exp["started_at"] = datetime.now().isoformat()
            self._save_experiment(experiment_id)

        logger.info(f"Started experiment {experiment_id}")
        return True

    def stop_experiment(self, experiment_id: str, reason: str = "") -> bool:
        """Stop an experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            exp = self._experiments[experiment_id]
            if exp["status"] != ExperimentStatus.RUNNING.value:
                logger.warning(f"Experiment {experiment_id} is not running")
                return False

            exp["status"] = ExperimentStatus.STOPPED.value
            exp["ended_at"] = datetime.now().isoformat()
            exp["stop_reason"] = reason
            self._save_experiment(experiment_id)

        logger.info(f"Stopped experiment {experiment_id}: {reason}")
        return True

    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            exp = self._experiments[experiment_id]
            if exp["status"] != ExperimentStatus.RUNNING.value:
                return False

            exp["status"] = ExperimentStatus.PAUSED.value
            self._save_experiment(experiment_id)

        logger.info(f"Paused experiment {experiment_id}")
        return True

    def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            exp = self._experiments[experiment_id]
            if exp["status"] != ExperimentStatus.PAUSED.value:
                return False

            exp["status"] = ExperimentStatus.RUNNING.value
            self._save_experiment(experiment_id)

        logger.info(f"Resumed experiment {experiment_id}")
        return True

    def allocate_variant(
        self,
        experiment_id: str,
        unit_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Allocate a variant to a randomization unit.

        Returns variant name or None if experiment not running.
        """
        with self._lock:
            if experiment_id not in self._experiments:
                return None

            exp = self._experiments[experiment_id]
            if exp["status"] != ExperimentStatus.RUNNING.value:
                return None

            variants = exp["config"]["variants"]
            strategy = AllocationStrategy(exp["config"]["allocation_strategy"])

            if strategy == AllocationStrategy.RANDOM:
                return self._random_allocation(variants)

            elif strategy == AllocationStrategy.HASH_BASED:
                return self._hash_allocation(unit_id, variants)

            elif strategy == AllocationStrategy.STICKY:
                return self._sticky_allocation(experiment_id, unit_id, variants)

            elif strategy == AllocationStrategy.CONTEXTUAL:
                return self._contextual_allocation(unit_id, variants, context)

            return variants[0]["name"]  # Fallback

    def _random_allocation(self, variants: List[Dict]) -> str:
        """Random allocation based on traffic weights."""
        r = np.random.random()
        cumulative = 0.0
        for v in variants:
            cumulative += v["traffic_allocation"]
            if r <= cumulative:
                return v["name"]
        return variants[-1]["name"]

    def _hash_allocation(self, unit_id: str, variants: List[Dict]) -> str:
        """Deterministic hash-based allocation."""
        # Use hash of unit_id for consistent allocation
        hash_val = hash(unit_id) % 10000 / 10000.0
        cumulative = 0.0
        for v in variants:
            cumulative += v["traffic_allocation"]
            if hash_val <= cumulative:
                return v["name"]
        return variants[-1]["name"]

    def _sticky_allocation(
        self,
        experiment_id: str,
        unit_id: str,
        variants: List[Dict],
    ) -> str:
        """Sticky allocation - persist assignment."""
        exp = self._experiments[experiment_id]
        assignments = exp.get("assignments", {})

        if unit_id in assignments:
            return assignments[unit_id]

        # New assignment
        variant = self._random_allocation(variants)
        assignments[unit_id] = variant
        exp["assignments"] = assignments
        self._save_experiment(experiment_id)
        return variant

    def _contextual_allocation(
        self,
        unit_id: str,
        variants: List[Dict],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Contextual allocation based on user context."""
        # Simple implementation - can be extended with ML-based allocation
        return self._hash_allocation(unit_id, variants)

    def record_observation(
        self,
        experiment_id: str,
        variant_name: str,
        unit_id: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Record an observation for a variant."""
        with self._lock:
            if experiment_id not in self._experiments:
                return False

            exp = self._experiments[experiment_id]
            if exp["status"] != ExperimentStatus.RUNNING.value:
                return False

            if variant_name not in exp["data"]:
                exp["data"][variant_name] = {"observations": [], "metrics": {}}

            observation = {
                "unit_id": unit_id,
                "metrics": metrics,
                "timestamp": (timestamp or datetime.now()).isoformat(),
            }
            exp["data"][variant_name]["observations"].append(observation)

            # Update running metrics
            self._update_running_metrics(exp, variant_name, metrics)

            # Periodic save
            if len(exp["data"][variant_name]["observations"]) % 100 == 0:
                self._save_experiment(experiment_id)

        return True

    def _update_running_metrics(
        self,
        exp: Dict,
        variant_name: str,
        new_metrics: Dict[str, float],
    ):
        """Update running metric aggregates."""
        variant_data = exp["data"][variant_name]
        metrics = variant_data.setdefault("metrics", {})

        for metric_name, value in new_metrics.items():
            if metric_name not in metrics:
                metrics[metric_name] = {
                    "count": 0,
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "min": float('inf'),
                    "max": float('-inf'),
                }
            m = metrics[metric_name]
            m["count"] += 1
            m["sum"] += value
            m["sum_sq"] += value * value
            m["min"] = min(m["min"], value)
            m["max"] = max(m["max"], value)

    def get_running_metrics(self, experiment_id: str) -> Dict[str, Dict]:
        """Get current running metrics for all variants."""
        with self._lock:
            if experiment_id not in self._experiments:
                return {}

            exp = self._experiments[experiment_id]
            results = {}

            for variant_name, data in exp["data"].items():
                metrics = data.get("metrics", {})
                results[variant_name] = {}
                for metric_name, m in metrics.items():
                    if m["count"] > 0:
                        mean = m["sum"] / m["count"]
                        var = (m["sum_sq"] / m["count"]) - mean**2
                        std = max(0, var)**0.5
                        results[variant_name][metric_name] = {
                            "count": m["count"],
                            "mean": mean,
                            "std": std,
                            "min": m["min"],
                            "max": m["max"],
                        }

            return results

    def analyze_experiment(
        self,
        experiment_id: str,
        primary_metric: str,
        correction_method: str = "none",
    ) -> ExperimentResult:
        """
        Perform statistical analysis of experiment.

        Args:
            experiment_id: Experiment to analyze
            primary_metric: Name of primary metric for decision
            correction_method: Multiple comparison correction (none, bonferroni, benjamini_hochberg, holm)

        Returns:
            ExperimentResult with statistical tests and recommendation
        """
        with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            exp = self._experiments[experiment_id]
            config = exp["config"]
            variants = config["variants"]

            if len(variants) < 2:
                raise ValueError("Need at least 2 variants for analysis")

            # Get observations
            variant_data = {}
            for v in variants:
                obs = exp["data"].get(v["name"], {}).get("observations", [])
                variant_data[v["name"]] = obs

            # Extract metric values
            metric_values = {}
            for v_name, obs in variant_data.items():
                metric_values[v_name] = [
                    o["metrics"].get(primary_metric) for o in obs
                    if primary_metric in o["metrics"]
                ]
                metric_values[v_name] = [v for v in metric_values[v_name] if v is not None]

            # Sample sizes
            sample_sizes = {v_name: len(vals) for v_name, vals in metric_values.items()}

            # Statistical tests (compare each variant to control)
            control_name = variants[0]["name"]
            statistical_tests = {}
            metric_results = {}

            for v_name, vals in metric_values.items():
                if v_name == control_name:
                    continue

                control_vals = np.array(metric_values[control_name])
                variant_vals = np.array(vals)

                if len(control_vals) == 0 or len(variant_vals) == 0:
                    continue

                # Determine test based on metric type
                metric_config = next(
                    (m for m in config["metrics"] if m["name"] == primary_metric),
                    None,
                )
                metric_type = MetricType(metric_config["metric_type"]) if metric_config else MetricType.MEAN_VALUE

                if metric_type == MetricType.CONVERSION_RATE:
                    # Proportion test
                    control_success = int(np.sum(control_vals))
                    control_trials = len(control_vals)
                    variant_success = int(np.sum(variant_vals))
                    variant_trials = len(variant_vals)
                    test = self.stats.proportion_z_test(
                        control_success, control_trials,
                        variant_success, variant_trials,
                    )
                else:
                    # Continuous metric - use Welch's t-test
                    test = self.stats.t_test_independent(
                        control_vals, variant_vals,
                        equal_var=False,
                    )

                statistical_tests[f"{control_name}_vs_{v_name}"] = test

                # Metric summary
                metric_results[v_name] = {
                    "mean": float(np.mean(variant_vals)) if len(variant_vals) > 0 else 0,
                    "std": float(np.std(variant_vals)) if len(variant_vals) > 0 else 0,
                    "count": len(variant_vals),
                    "test": test.to_dict(),
                }

            # Multiple comparison correction
            if correction_method != "none" and statistical_tests:
                p_values = [t.p_value for t in statistical_tests.values()]
                if correction_method == "bonferroni":
                    adjusted, significant = self.stats.bonferroni_correction(p_values)
                elif correction_method == "benjamini_hochberg":
                    adjusted, significant = self.stats.benjamini_hochberg_correction(p_values)
                elif correction_method == "holm":
                    adjusted, significant = self.stats.holm_bonferroni_correction(p_values)
                else:
                    adjusted, significant = p_values, [p < self.default_alpha for p in p_values]

                for (key, test), adj_p, sig in zip(statistical_tests.items(), adjusted, significant):
                    test.metadata["adjusted_p_value"] = adj_p
                    test.metadata["significant_after_correction"] = sig

            # Determine winner
            winner = self._determine_winner(
                control_name, metric_results, statistical_tests, primary_metric, config
            )

            # Overall confidence
            confidence = self._calculate_confidence(statistical_tests)

            # Recommendation
            recommendation = self._generate_recommendation(
                winner, confidence, statistical_tests, config
            )

            result = ExperimentResult(
                experiment_id=experiment_id,
                experiment_name=config["name"],
                status=ExperimentStatus(exp["status"]),
                start_time=datetime.fromisoformat(exp["started_at"]) if exp["started_at"] else datetime.now(),
                end_time=datetime.fromisoformat(exp["ended_at"]) if exp.get("ended_at") else None,
                variants=[ExperimentVariant(**v) for v in variants],
                metrics=[ExperimentMetric(**m) for m in config["metrics"]],
                sample_sizes=sample_sizes,
                metric_results=metric_results,
                statistical_tests=statistical_tests,
                winner=winner,
                confidence=confidence,
                recommendation=recommendation,
                metadata={
                    "primary_metric": primary_metric,
                    "correction_method": correction_method,
                    "config": config,
                },
            )

            # Save results
            exp["results"] = {
                "winner": winner,
                "confidence": confidence,
                "recommendation": recommendation,
                "statistical_tests": {k: v.to_dict() for k, v in statistical_tests.items()},
                "metric_results": metric_results,
                "analyzed_at": datetime.now().isoformat(),
            }
            self._save_experiment(experiment_id)

            return result

    def _determine_winner(
        self,
        control_name: str,
        metric_results: Dict,
        statistical_tests: Dict,
        primary_metric: str,
        config: Dict,
    ) -> Optional[str]:
        """Determine winning variant."""
        metric_config = next(
            (m for m in config["metrics"] if m["name"] == primary_metric),
            None,
        )
        higher_is_better = metric_config.get("higher_is_better", True) if metric_config else True

        # Check if any variant significantly beats control
        best_variant = control_name
        best_mean = metric_results.get(control_name, {}).get("mean", 0)

        for v_name, results in metric_results.items():
            if v_name == control_name:
                continue

            test_key = f"{control_name}_vs_{v_name}"
            test = statistical_tests.get(test_key)

            if test and test.significant:
                mean = results.get("mean", 0)
                if higher_is_better and mean > best_mean:
                    best_variant = v_name
                    best_mean = mean
                elif not higher_is_better and mean < best_mean:
                    best_variant = v_name
                    best_mean = mean

        return best_variant if best_variant != control_name else None

    def _calculate_confidence(self, statistical_tests: Dict) -> float:
        """Calculate overall confidence in results."""
        if not statistical_tests:
            return 0.0

        # Use minimum p-value as confidence indicator
        min_p = min(t.p_value for t in statistical_tests.values())
        return max(0, 1 - min_p)

    def _generate_recommendation(
        self,
        winner: Optional[str],
        confidence: float,
        statistical_tests: Dict,
        config: Dict,
    ) -> str:
        """Generate human-readable recommendation."""
        if winner:
            return f"Implement {winner} - statistically significant improvement with {confidence:.1%} confidence"
        else:
            # Check if any test was close to significance
            min_p = min((t.p_value for t in statistical_tests.values()), default=1.0)
            if min_p < 0.1:
                return f"No clear winner yet. Consider running experiment longer (minimum p-value: {min_p:.4f})"
            return "No statistically significant difference detected. Consider larger sample size or different variants."

    def get_experiment_status(self, experiment_id: str) -> Dict:
        """Get current experiment status and metrics."""
        with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            exp = self._experiments[experiment_id]
            running_metrics = self.get_running_metrics(experiment_id)

            return {
                "experiment_id": experiment_id,
                "name": exp["config"]["name"],
                "status": exp["status"],
                "created_at": exp["created_at"],
                "started_at": exp["started_at"],
                "ended_at": exp.get("ended_at"),
                "variants": exp["config"]["variants"],
                "running_metrics": running_metrics,
                "results": exp.get("results"),
            }

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Dict]:
        """List all experiments."""
        with self._lock:
            experiments = []
            for exp_id, exp in self._experiments.items():
                if status and exp["status"] != status.value:
                    continue
                experiments.append({
                    "experiment_id": exp_id,
                    "name": exp["config"]["name"],
                    "status": exp["status"],
                    "created_at": exp["created_at"],
                    "started_at": exp["started_at"],
                    "variants": [v["name"] for v in exp["config"]["variants"]],
                })
            return sorted(experiments, key=lambda x: x["created_at"], reverse=True)

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                return False

            del self._experiments[experiment_id]
            filepath = self.storage_path / f"{experiment_id}.json"
            if filepath.exists():
                filepath.unlink()

        logger.info(f"Deleted experiment {experiment_id}")
        return True

    def export_results(self, experiment_id: str, output_path: str):
        """Export experiment results to JSON."""
        with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            exp = self._experiments[experiment_id]
            with open(output_path, 'w') as f:
                json.dump(exp, f, indent=2, default=str)

        logger.info(f"Exported experiment {experiment_id} to {output_path}")

    def power_analysis(
        self,
        experiment_id: str,
        primary_metric: str,
    ) -> PowerAnalysisResult:
        """Calculate required sample size for experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            exp = self._experiments[experiment_id]
            config = exp["config"]

            metric_config = next(
                (m for m in config["metrics"] if m["name"] == primary_metric),
                None,
            )
            metric_type = MetricType(metric_config["metric_type"]) if metric_config else MetricType.MEAN_VALUE

            mde = config["minimum_detectable_effect"]
            alpha = config["significance_level"]
            power = config["power"]

            if metric_type == MetricType.CONVERSION_RATE:
                # Assume baseline conversion rate of 0.1 for calculation
                p1 = 0.1
                p2 = p1 * (1 + mde) if mde > 0 else p1 * (1 - abs(mde))
                return self.stats.power_analysis_proportion(p1, p2, alpha, power)
            else:
                return self.stats.power_analysis_t_test(mde, alpha, power)


# Global framework instance
_default_framework: Optional[ABTestingFramework] = None


def get_ab_framework(storage_path: str = "./ab_experiments") -> ABTestingFramework:
    """Get or create default A/B testing framework."""
    global _default_framework
    if _default_framework is None:
        _default_framework = ABTestingFramework(storage_path)
    return _default_framework


def create_fraud_detection_experiment(
    name: str,
    control_model_uri: str,
    treatment_model_uri: str,
    traffic_split: float = 0.5,
    **kwargs,
) -> str:
    """Quick creation of fraud detection A/B experiment."""
    framework = get_ab_framework()

    config = ExperimentConfig(
        name=name,
        variants=[
            ExperimentVariant(
                name="control",
                model_uri=control_model_uri,
                traffic_allocation=1 - traffic_split,
                description="Current production model",
            ),
            ExperimentVariant(
                name="treatment",
                model_uri=treatment_model_uri,
                traffic_allocation=traffic_split,
                description="New candidate model",
            ),
        ],
        metrics=[
            ExperimentMetric(
                name="fraud_detection_rate",
                metric_type=MetricType.CONVERSION_RATE,
                higher_is_better=True,
                description="Fraction of fraud correctly detected",
            ),
            ExperimentMetric(
                name="false_positive_rate",
                metric_type=MetricType.CONVERSION_RATE,
                higher_is_better=False,
                description="Fraction of legitimate transactions flagged as fraud",
            ),
            ExperimentMetric(
                name="inference_latency_ms",
                metric_type=MetricType.LATENCY,
                higher_is_better=False,
                description="Model inference latency in milliseconds",
            ),
        ],
        **kwargs,
    )

    return framework.create_experiment(config)