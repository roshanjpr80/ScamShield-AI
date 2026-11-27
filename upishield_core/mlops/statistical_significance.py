"""
Statistical Significance Testing for ScamShield AI A/B Testing.

Provides rigorous statistical tests for model comparison,
including proportion tests, mean tests, and non-parametric alternatives.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import math

import numpy as np
from scipy import stats
from scipy.stats import norm, t, chi2, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of statistical tests."""
    PROPORTION_Z_TEST = "proportion_z_test"
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


class AlternativeHypothesis(Enum):
    """Alternative hypothesis direction."""
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_type: TestType
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float] = None
    effect_size_ci: Optional[Tuple[float, float]] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    degrees_of_freedom: Optional[float] = None
    sample_size_a: int = 0
    sample_size_b: int = 0
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_type": self.test_type.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "effect_size_ci": self.effect_size_ci,
            "confidence_interval": self.confidence_interval,
            "degrees_of_freedom": self.degrees_of_freedom,
            "sample_size_a": self.sample_size_a,
            "sample_size_b": self.sample_size_b,
            "alternative": self.alternative.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class PowerAnalysisResult:
    """Result of power analysis."""
    power: float
    required_sample_size: int
    effect_size: float
    alpha: float
    test_type: TestType
    alternative: AlternativeHypothesis


class StatisticalSignificance:
    """
    Comprehensive statistical significance testing for A/B experiments.

    Features:
    - Parametric tests (t-test, z-test)
    - Non-parametric tests (Mann-Whitney, Wilcoxon)
    - Proportion tests for conversion rates
    - Effect size calculation (Cohen's d, Cliff's delta, odds ratio)
    - Confidence intervals
    - Power analysis
    - Multiple comparison correction (Bonferroni, Benjamini-Hochberg)
    - Sequential testing support
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

   # Proportion Tests (for conversion rates, click-through rates)
 
    def proportion_z_test(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> TestResult:
        """
        Two-proportion z-test for comparing conversion rates.

        Args:
            successes_a: Number of successes in group A
            trials_a: Total trials in group A
            successes_b: Number of successes in group B
            trials_b: Total trials in group B
            alternative: Alternative hypothesis direction

        Returns:
            TestResult with statistic, p-value, and effect size
        """
        if trials_a <= 0 or trials_b <= 0:
            raise ValueError("Trials must be positive")

        p_a = successes_a / trials_a
        p_b = successes_b / trials_b
        p_pool = (successes_a + successes_b) / (trials_a + trials_b)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1/trials_a + 1/trials_b))

        if se == 0:
            return TestResult(
                test_type=TestType.PROPORTION_Z_TEST,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                sample_size_a=trials_a,
                sample_size_b=trials_b,
                alternative=alternative,
                metadata={"p_a": p_a, "p_b": p_b, "p_pool": p_pool},
            )

        # Z-statistic
        z = (p_a - p_b) / se

        # P-value
        if alternative == AlternativeHypothesis.TWO_SIDED:
            p_value = 2 * (1 - norm.cdf(abs(z)))
        elif alternative == AlternativeHypothesis.GREATER:
            p_value = 1 - norm.cdf(z)
        else:  # LESS
            p_value = norm.cdf(z)

        # Effect size: Cohen's h for proportions
        h = 2 * (math.asin(math.sqrt(p_a)) - math.asin(math.sqrt(p_b)))

        # Confidence interval for difference
        se_diff = math.sqrt(p_a * (1 - p_a) / trials_a + p_b * (1 - p_b) / trials_b)
        z_critical = norm.ppf(1 - self.alpha / 2)
        margin = z_critical * se_diff
        diff = p_a - p_b
        ci = (diff - margin, diff + margin)

        return TestResult(
            test_type=TestType.PROPORTION_Z_TEST,
            statistic=z,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=h,
            confidence_interval=ci,
            sample_size_a=trials_a,
            sample_size_b=trials_b,
            alternative=alternative,
            metadata={
                "p_a": p_a,
                "p_b": p_b,
                "p_pool": p_pool,
                "difference": diff,
                "cohens_h": h,
            },
        )

    def proportion_chi_square_test(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
    ) -> TestResult:
        """Chi-square test for independence (2x2 contingency table)."""
        # Contingency table
        #          Success  Failure  Total
        # Group A   a       b        n1
        # Group B   c       d        n2
        a, b = successes_a, trials_a - successes_a
        c, d = successes_b, trials_b - successes_b

        # Expected frequencies
        n1 = trials_a
        n2 = trials_b
        total = n1 + n2
        e_a = (a + c) * n1 / total
        e_b = (b + d) * n1 / total
        e_c = (a + c) * n2 / total
        e_d = (b + d) * n2 / total

        # Chi-square statistic
        chi2_stat = sum([
            (a - e_a)**2 / e_a if e_a > 0 else 0,
            (b - e_b)**2 / e_b if e_b > 0 else 0,
            (c - e_c)**2 / e_c if e_c > 0 else 0,
            (d - e_d)**2 / e_d if e_d > 0 else 0,
        ])

        p_value = 1 - chi2.cdf(chi2_stat, df=1)

        # Effect size: Phi coefficient
        phi = math.sqrt(chi2_stat / total)

        return TestResult(
            test_type=TestType.CHI_SQUARE,
            statistic=chi2_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=phi,
            degrees_of_freedom=1,
            sample_size_a=trials_a,
            sample_size_b=trials_b,
            metadata={
                "contingency": [[a, b], [c, d]],
                "expected": [[e_a, e_b], [e_c, e_d]],
                "phi_coefficient": phi,
            },
        )

    # ============================================================
    # Mean Comparison Tests (for latency, scores, revenue)
    # ============================================================

    def t_test_independent(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        equal_var: bool = True,
    ) -> TestResult:
        """
        Independent samples t-test (Student's or Welch's).

        Args:
            sample_a: Sample data for group A
            sample_b: Sample data for group B
            alternative: Alternative hypothesis
            equal_var: If True, use Student's t-test; if False, Welch's t-test

        Returns:
            TestResult with statistic, p-value, and Cohen's d
        """
        sample_a = np.asarray(sample_a)
        sample_b = np.asarray(sample_b)

        n_a, n_b = len(sample_a), len(sample_b)
        mean_a, mean_b = np.mean(sample_a), np.mean(sample_b)
        var_a, var_b = np.var(sample_a, ddof=1), np.var(sample_b, ddof=1)

        if equal_var:
            # Pooled variance
            pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
            se = math.sqrt(pooled_var * (1/n_a + 1/n_b))
            df = n_a + n_b - 2
            test_type = TestType.T_TEST_INDEPENDENT
        else:
            # Welch's t-test
            se = math.sqrt(var_a/n_a + var_b/n_b)
            df = (var_a/n_a + var_b/n_b)**2 / (
                (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
            )
            test_type = TestType.WELCH_T_TEST

        if se == 0:
            return TestResult(
                test_type=test_type,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                sample_size_a=n_a,
                sample_size_b=n_b,
                alternative=alternative,
            )

        t_stat = (mean_a - mean_b) / se

        # P-value
        if alternative == AlternativeHypothesis.TWO_SIDED:
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        elif alternative == AlternativeHypothesis.GREATER:
            p_value = 1 - t.cdf(t_stat, df)
        else:
            p_value = t.cdf(t_stat, df)

        # Cohen's d
        if equal_var:
            pooled_sd = math.sqrt(pooled_var)
        else:
            pooled_sd = math.sqrt((var_a + var_b) / 2)
        cohens_d = (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0

        # Confidence interval for difference
        t_critical = t.ppf(1 - self.alpha / 2, df)
        margin = t_critical * se
        diff = mean_a - mean_b
        ci = (diff - margin, diff + margin)

        return TestResult(
            test_type=test_type,
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=cohens_d,
            confidence_interval=ci,
            degrees_of_freedom=df,
            sample_size_a=n_a,
            sample_size_b=n_b,
            alternative=alternative,
            metadata={
                "mean_a": mean_a,
                "mean_b": mean_b,
                "std_a": math.sqrt(var_a),
                "std_b": math.sqrt(var_b),
                "cohens_d": cohens_d,
            },
        )

    def t_test_paired(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> TestResult:
        """Paired samples t-test."""
        sample_a = np.asarray(sample_a)
        sample_b = np.asarray(sample_b)

        if len(sample_a) != len(sample_b):
            raise ValueError("Paired samples must have same length")

        differences = sample_a - sample_b
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        if std_diff == 0:
            return TestResult(
                test_type=TestType.T_TEST_PAIRED,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                sample_size_a=n,
                sample_size_b=n,
                alternative=alternative,
            )

        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se
        df = n - 1

        if alternative == AlternativeHypothesis.TWO_SIDED:
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        elif alternative == AlternativeHypothesis.GREATER:
            p_value = 1 - t.cdf(t_stat, df)
        else:
            p_value = t.cdf(t_stat, df)

        # Cohen's d for paired samples
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        t_critical = t.ppf(1 - self.alpha / 2, df)
        margin = t_critical * se
        ci = (mean_diff - margin, mean_diff + margin)

        return TestResult(
            test_type=TestType.T_TEST_PAIRED,
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=cohens_d,
            confidence_interval=ci,
            degrees_of_freedom=df,
            sample_size_a=n,
            sample_size_b=n,
            alternative=alternative,
            metadata={
                "mean_difference": mean_diff,
                "std_difference": std_diff,
                "cohens_d": cohens_d,
            },
        )

  # Non-Parametric Tests
 
    def mann_whitney_u_test(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> TestResult:
        """Mann-Whitney U test (Wilcoxon rank-sum test)."""
        sample_a = np.asarray(sample_a)
        sample_b = np.asarray(sample_b)

        n_a, n_b = len(sample_a), len(sample_b)

        # Use scipy's implementation
        stat, p_value = mannwhitneyu(
            sample_a, sample_b,
            alternative=alternative.value.replace("-sided", "-sided"),
        )

        # Effect size: Cliff's delta
        cliff_delta = self._cliffs_delta(sample_a, sample_b)

        # Confidence interval for Hodges-Lehmann estimator (median difference)
        ci = self._hodges_lehmann_ci(sample_a, sample_b)

        return TestResult(
            test_type=TestType.MANN_WHITNEY_U,
            statistic=stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=cliff_delta,
            confidence_interval=ci,
            sample_size_a=n_a,
            sample_size_b=n_b,
            alternative=alternative,
            metadata={
                "cliffs_delta": cliff_delta,
                "median_a": np.median(sample_a),
                "median_b": np.median(sample_b),
            },
        )

    def wilcoxon_signed_rank_test(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> TestResult:
        """Wilcoxon signed-rank test for paired samples."""
        sample_a = np.asarray(sample_a)
        sample_b = np.asarray(sample_b)

        if len(sample_a) != len(sample_b):
            raise ValueError("Paired samples must have same length")

        differences = sample_a - sample_b
        # Remove zero differences
        differences = differences[differences != 0]
        n = len(differences)

        if n == 0:
            return TestResult(
                test_type=TestType.WILCOXON_SIGNED_RANK,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                sample_size_a=len(sample_a),
                sample_size_b=len(sample_b),
                alternative=alternative,
            )

        stat, p_value = wilcoxon(
            differences,
            alternative=alternative.value.replace("-sided", "-sided"),
        )

        # Effect size: matched-pairs rank-biserial correlation
        r = 1 - (2 * stat) / (n * (n + 1))

        return TestResult(
            test_type=TestType.WILCOXON_SIGNED_RANK,
            statistic=stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=r,
            sample_size_a=n,
            sample_size_b=n,
            alternative=alternative,
            metadata={
                "rank_biserial": r,
                "median_difference": np.median(sample_a - sample_b),
            },
        )

    def kruskal_wallis_test(
        self,
        *samples: np.ndarray,
    ) -> TestResult:
        """Kruskal-Wallis H test for multiple independent samples."""
        samples = [np.asarray(s) for s in samples]
        n_samples = len(samples)

        if n_samples < 2:
            raise ValueError("Need at least 2 samples")

        stat, p_value = kruskal(*samples)

        # Effect size: epsilon-squared
        n_total = sum(len(s) for s in samples)
        epsilon_sq = (stat - n_samples + 1) / (n_total - n_samples)

        return TestResult(
            test_type=TestType.KRUSKAL_WALLIS,
            statistic=stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=epsilon_sq,
            degrees_of_freedom=n_samples - 1,
            sample_size_a=n_total,
            metadata={
                "epsilon_squared": epsilon_sq,
                "group_sizes": [len(s) for s in samples],
                "group_medians": [np.median(s) for s in samples],
            },
        )

    def friedman_test(
        self,
        *samples: np.ndarray,
    ) -> TestResult:
        """Friedman test for multiple related samples."""
        samples = [np.asarray(s) for s in samples]
        n_samples = len(samples)

        if n_samples < 2:
            raise ValueError("Need at least 2 samples")

        stat, p_value = friedmanchisquare(*samples)

        # Effect size: Kendall's W
        n_blocks = len(samples[0])
        kendalls_w = stat / (n_blocks * (n_samples - 1))

        return TestResult(
            test_type=TestType.FRIEDMAN,
            statistic=stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=kendalls_w,
            degrees_of_freedom=n_samples - 1,
            sample_size_a=n_blocks,
            metadata={
                "kendalls_w": kendalls_w,
                "group_sizes": [len(s) for s in samples],
            },
        )

   # Resampling-Based Tests
 
    def bootstrap_test(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        statistic_func: callable = None,
        n_bootstrap: int = 10000,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        random_state: int = 42,
    ) -> TestResult:
        """
        Bootstrap test for difference in statistics.

        Args:
            sample_a: Sample A
            sample_b: Sample B
            statistic_func: Function to compute statistic (default: mean difference)
            n_bootstrap: Number of bootstrap resamples
            alternative: Alternative hypothesis
            random_state: Random seed
        """
        np.random.seed(random_state)
        sample_a = np.asarray(sample_a)
        sample_b = np.asarray(sample_b)

        if statistic_func is None:
            statistic_func = lambda x, y: np.mean(x) - np.mean(y)

        observed_stat = statistic_func(sample_a, sample_b)

        # Bootstrap resampling
        bootstrap_stats = []
        combined = np.concatenate([sample_a, sample_b])
        n_a, n_b = len(sample_a), len(sample_b)

        for _ in range(n_bootstrap):
            resample_a = np.random.choice(sample_a, size=n_a, replace=True)
            resample_b = np.random.choice(sample_b, size=n_b, replace=True)
            bootstrap_stats.append(statistic_func(resample_a, resample_b))

        bootstrap_stats = np.array(bootstrap_stats)

        # P-value: proportion of bootstrap stats as extreme as observed
        if alternative == AlternativeHypothesis.TWO_SIDED:
            p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
        elif alternative == AlternativeHypothesis.GREATER:
            p_value = np.mean(bootstrap_stats >= observed_stat)
        else:
            p_value = np.mean(bootstrap_stats <= observed_stat)

        # Confidence interval (percentile method)
        alpha_ci = self.alpha
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha_ci / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha_ci / 2))

        return TestResult(
            test_type=TestType.BOOTSTRAP,
            statistic=observed_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=observed_stat,
            confidence_interval=(ci_lower, ci_upper),
            sample_size_a=n_a,
            sample_size_b=n_b,
            alternative=alternative,
            metadata={
                "n_bootstrap": n_bootstrap,
                "bootstrap_mean": np.mean(bootstrap_stats),
                "bootstrap_std": np.std(bootstrap_stats),
            },
        )

    def permutation_test(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        statistic_func: callable = None,
        n_permutations: int = 10000,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        random_state: int = 42,
    ) -> TestResult:
        """
        Permutation test for difference in statistics.

        Exact test under null hypothesis of exchangeability.
        """
        np.random.seed(random_state)
        sample_a = np.asarray(sample_a)
        sample_b = np.asarray(sample_b)

        if statistic_func is None:
            statistic_func = lambda x, y: np.mean(x) - np.mean(y)

        observed_stat = statistic_func(sample_a, sample_b)
        combined = np.concatenate([sample_a, sample_b])
        n_a, n_b = len(sample_a), len(sample_b)

        # Permutation resampling
        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]
            perm_stats.append(statistic_func(perm_a, perm_b))

        perm_stats = np.array(perm_stats)

        if alternative == AlternativeHypothesis.TWO_SIDED:
            p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        elif alternative == AlternativeHypothesis.GREATER:
            p_value = np.mean(perm_stats >= observed_stat)
        else:
            p_value = np.mean(perm_stats <= observed_stat)

        return TestResult(
            test_type=TestType.PERMUTATION,
            statistic=observed_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=observed_stat,
            sample_size_a=n_a,
            sample_size_b=n_b,
            alternative=alternative,
            metadata={
                "n_permutations": n_permutations,
                "perm_mean": np.mean(perm_stats),
                "perm_std": np.std(perm_stats),
            },
        )

   # Effect Size Calculations
 
    def _cliffs_delta(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """Calculate Cliff's delta effect size."""
        n_a, n_b = len(sample_a), len(sample_b)
        if n_a == 0 or n_b == 0:
            return 0.0

        # Count pairwise comparisons
        greater = sum(1 for a in sample_a for b in sample_b if a > b)
        less = sum(1 for a in sample_a for b in sample_b if a < b)
        return (greater - less) / (n_a * n_b)

    def _hodges_lehmann_ci(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Hodges-Lehmann confidence interval for median difference."""
        n_a, n_b = len(sample_a), len(sample_b)
        differences = np.array([a - b for a in sample_a for b in sample_b])
        differences.sort()

        alpha = 1 - confidence
        k = int(np.ceil(n_a * n_b * alpha / 2))

        if k >= len(differences) // 2:
            return (differences[0], differences[-1])

        return (differences[k], differences[-(k + 1)])

    def cohens_d(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """Calculate Cohen's d."""
        sample_a = np.asarray(sample_a)
        sample_b = np.asarray(sample_b)

        mean_a, mean_b = np.mean(sample_a), np.mean(sample_b)
        var_a, var_b = np.var(sample_a, ddof=1), np.var(sample_b, ddof=1)
        n_a, n_b = len(sample_a), len(sample_b)

        pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        if pooled_sd == 0:
            return 0.0
        return (mean_a - mean_b) / pooled_sd

    def hedges_g(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """Calculate Hedges' g (bias-corrected Cohen's d)."""
        d = self.cohens_d(sample_a, sample_b)
        n_a, n_b = len(sample_a), len(sample_b)
        df = n_a + n_b - 2
        correction = 1 - 3 / (4 * df - 1)
        return d * correction

    def odds_ratio(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
    ) -> Tuple[float, Tuple[float, float]]:
        """Calculate odds ratio with confidence interval."""
        a, b = successes_a, trials_a - successes_a
        c, d = successes_b, trials_b - successes_b

        # Add 0.5 correction for zero cells
        if a == 0 or b == 0 or c == 0 or d == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

        or_val = (a * d) / (b * c)
        log_or = math.log(or_val)
        se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)

        z = norm.ppf(1 - self.alpha / 2)
        ci_lower = math.exp(log_or - z * se_log_or)
        ci_upper = math.exp(log_or + z * se_log_or)

        return or_val, (ci_lower, ci_upper)

    # Multiple Comparison Correction
 
    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None,
    ) -> Tuple[List[float], List[bool]]:
        """
        Bonferroni correction for multiple comparisons.

        Returns adjusted p-values and significance decisions.
        """
        alpha = alpha or self.alpha
        n = len(p_values)
        adjusted = [min(p * n, 1.0) for p in p_values]
        significant = [p < alpha for p in adjusted]
        return adjusted, significant

    def benjamini_hochberg_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None,
    ) -> Tuple[List[float], List[bool]]:
        """
        Benjamini-Hochberg FDR correction.

        Returns adjusted p-values (q-values) and significance decisions.
        """
        alpha = alpha or self.alpha
        n = len(p_values)

        # Sort p-values with original indices
        sorted_p = sorted((p, i) for i, p in enumerate(p_values))

        # Compute q-values
        q_values = [0.0] * n
        for rank, (p, idx) in enumerate(sorted_p, 1):
            q_values[idx] = p * n / rank

        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            if q_values[i] > q_values[i + 1]:
                q_values[i] = q_values[i + 1]

        # Cap at 1.0
        q_values = [min(q, 1.0) for q in q_values]

        significant = [q < alpha for q in q_values]
        return q_values, significant

    def holm_bonferroni_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None,
    ) -> Tuple[List[float], List[bool]]:
        """Holm-Bonferroni step-down procedure."""
        alpha = alpha or self.alpha
        n = len(p_values)

        sorted_p = sorted((p, i) for i, p in enumerate(p_values))
        adjusted = [0.0] * n
        significant = [False] * n

        for rank, (p, idx) in enumerate(sorted_p, 1):
            adjusted[idx] = min(p * (n - rank + 1), 1.0)
            significant[idx] = adjusted[idx] < alpha

        return adjusted, significant

  # Power Analysis
  
    def power_analysis_proportion(
        self,
        p1: float,
        p2: float,
        alpha: Optional[float] = None,
        power: float = 0.8,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        ratio: float = 1.0,
    ) -> PowerAnalysisResult:
        """
        Power analysis for two-proportion z-test.

        Returns required sample size per group.
        """
        alpha = alpha or self.alpha

        # Effect size (Cohen's h)
        h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

        # Z values
        if alternative == AlternativeHypothesis.TWO_SIDED:
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)

        # Sample size formula
        n = ((z_alpha + z_beta)**2 * (p1 * (1 - p1) + p2 * (1 - p2) / ratio)) / (p1 - p2)**2
        n = math.ceil(n)

        return PowerAnalysisResult(
            power=power,
            required_sample_size=n,
            effect_size=abs(h),
            alpha=alpha,
            test_type=TestType.PROPORTION_Z_TEST,
            alternative=alternative,
        )

    def power_analysis_t_test(
        self,
        effect_size: float,
        alpha: Optional[float] = None,
        power: float = 0.8,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        ratio: float = 1.0,
    ) -> PowerAnalysisResult:
        """
        Power analysis for two-sample t-test.

        Args:
            effect_size: Cohen's d
            alpha: Significance level
            power: Desired power
            alternative: Alternative hypothesis
            ratio: Sample size ratio (n2/n1)
        """
        alpha = alpha or self.alpha

        if alternative == AlternativeHypothesis.TWO_SIDED:
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)

        # Sample size per group (approximation using normal)
        n1 = ((z_alpha + z_beta)**2 * (1 + 1/ratio)) / effect_size**2
        n1 = math.ceil(n1)

        return PowerAnalysisResult(
            power=power,
            required_sample_size=n1,
            effect_size=effect_size,
            alpha=alpha,
            test_type=TestType.T_TEST_INDEPENDENT,
            alternative=alternative,
        )

    def achieved_power_proportion(
        self,
        p1: float,
        p2: float,
        n1: int,
        n2: int,
        alpha: Optional[float] = None,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> float:
        """Calculate achieved power for given sample sizes."""
        alpha = alpha or self.alpha

        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        effect = abs(p1 - p2)

        if alternative == AlternativeHypothesis.TWO_SIDED:
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)

        z = effect / se - z_alpha
        return norm.cdf(z)

    def achieved_power_t_test(
        self,
        effect_size: float,
        n1: int,
        n2: int,
        alpha: Optional[float] = None,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> float:
        """Calculate achieved power for t-test."""
        alpha = alpha or self.alpha
        df = n1 + n2 - 2

        if alternative == AlternativeHypothesis.TWO_SIDED:
            t_critical = t.ppf(1 - alpha / 2, df)
        else:
            t_critical = t.ppf(1 - alpha, df)

        ncp = effect_size * math.sqrt(n1 * n2 / (n1 + n2))
        return 1 - t.cdf(t_critical, df, ncp) + t.cdf(-t_critical, df, ncp)


# Convenience functions
def quick_proportion_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    alpha: float = 0.05,
) -> TestResult:
    """Quick two-proportion z-test."""
    tester = StatisticalSignificance(alpha)
    return tester.proportion_z_test(successes_a, trials_a, successes_b, trials_b)


def quick_t_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = False,
) -> TestResult:
    """Quick independent t-test (Welch's by default)."""
    tester = StatisticalSignificance(alpha)
    return tester.t_test_independent(sample_a, sample_b, equal_var=equal_var)


def quick_mann_whitney(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Quick Mann-Whitney U test."""
    tester = StatisticalSignificance(alpha)
    return tester.mann_whitney_u_test(sample_a, sample_b)


def quick_bootstrap(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> TestResult:
    """Quick bootstrap test for mean difference."""
    tester = StatisticalSignificance(alpha)
    return tester.bootstrap_test(sample_a, sample_b, n_bootstrap=n_bootstrap)