"""
Feature Computation Engine.

Computes derived features from raw transaction data for real-time scoring.
Optimized for sub-millisecond computation per transaction.
"""

import math
from typing import List

import numpy as np

from upishield_core.features.schemas import (
    UPIFeatures,
)


class FeatureComputer:
    """
    Compute derived features from raw data.

    All computations are pure functions - no I/O, no state.
    Optimized for speed with NumPy vectorization where possible.
    """

    # Merchant category risk mapping (pre-computed from historical data)
    MERCHANT_CATEGORY_RISK = {
        "ecommerce": 0.3,
        "food_delivery": 0.2,
        "ride_hailing": 0.15,
        "bill_payment": 0.1,
        "recharge": 0.05,
        "peer_transfer": 0.4,
        "investment": 0.5,
        "gaming": 0.6,
        "crypto": 0.8,
        "gambling": 0.9,
        "unknown": 0.5,
    }

    # Hour-based risk scores (0-1, higher = riskier)
    HOUR_RISK = np.array(
        [
            0.8,
            0.85,
            0.9,
            0.9,
            0.85,
            0.7,  # 0-5: Night (high risk)
            0.4,
            0.3,
            0.25,
            0.2,
            0.2,
            0.2,  # 6-11: Morning (low risk)
            0.25,
            0.3,
            0.35,
            0.3,
            0.35,
            0.4,  # 12-17: Afternoon (medium)
            0.45,
            0.55,
            0.65,
            0.7,
            0.75,
            0.8,  # 18-23: Evening (rising risk)
        ]
    )

    # Device type risk scores
    DEVICE_TYPE_RISK = {
        "android": 0.3,
        "ios": 0.2,
        "web": 0.4,
        "unknown": 0.5,
    }

    @classmethod
    def compute_all(cls, features: UPIFeatures) -> UPIFeatures:
        """
        Compute all derived features in place.

        This is the main entry point - call after FeatureStore.get_features().
        """
        # Transaction-level derived features
        cls._compute_transaction_derived(features)

        # User behavioral derived features
        if features.user_behavioral:
            cls._compute_user_derived(features)

        # Device/Network derived features
        if features.device_network:
            cls._compute_device_derived(features)

        # Merchant derived features
        if features.merchant:
            cls._compute_merchant_derived(features)

        # Velocity derived features
        if features.velocity:
            cls._compute_velocity_derived(features)

        # Graph derived features
        if features.graph:
            cls._compute_graph_derived(features)

        return features

    @classmethod
    def _compute_transaction_derived(cls, features: UPIFeatures) -> None:
        """Compute transaction-level derived features."""
        # Night transaction (10 PM - 6 AM)
        if features.timestamp:
            hour = features.timestamp.hour
            features.is_night_txn = hour >= 22 or hour < 6
            features.hour_risk_score = float(cls.HOUR_RISK[hour])
        else:
            features.is_night_txn = False
            features.hour_risk_score = 0.5

        # High value transaction (>=50k INR)
        features.is_high_value = features.amount >= 50000
        features.is_very_high_value = features.amount > 200000

        # Round number check (common in fraud)
        features.is_round_amount = features.amount % 1000 == 0
        features.is_suspicious_amount = features.amount in [9999, 19999, 49999, 99999, 199999] or (
            features.amount % 100 == 0 and features.amount > 10000
        )

        # VPA similarity (simple hash-based)
        features.payer_payee_similarity = cls._vpa_similarity(
            features.payer_vpa, features.payee_vpa
        )

    @classmethod
    def _compute_user_derived(cls, features: UPIFeatures) -> None:
        """Compute user behavioral derived features."""
        ub = features.user_behavioral

        # Amount z-score vs 7-day average
        if ub.avg_txn_amount_7d > 0:
            features.amount_zscore_7d = (features.amount - ub.avg_txn_amount_7d) / max(
                ub.avg_txn_amount_7d, 1000
            )
        else:
            features.amount_zscore_7d = 0.0

        # Amount z-score vs 30-day average
        if ub.avg_txn_amount_30d > 0:
            features.amount_zscore_30d = (features.amount - ub.avg_txn_amount_30d) / max(
                ub.avg_txn_amount_30d, 1000
            )
        else:
            features.amount_zscore_30d = 0.0

        # Transaction frequency anomaly
        if ub.txn_count_24h > 0:
            features.txn_frequency_score = min(
                ub.txn_count_1h / max(ub.txn_count_24h / 24.0, 1), 2.0
            )
        else:
            features.txn_frequency_score = 0.0

        # Merchant diversity
        if ub.txn_count_7d > 0:
            features.merchant_diversity_7d = ub.unique_merchants_7d / ub.txn_count_7d
        else:
            features.merchant_diversity_7d = 0.0

        # Use txn_count_7d as proxy since txn_count_30d not in schema
        if ub.txn_count_7d > 0:
            features.merchant_diversity_30d = ub.unique_merchants_30d / ub.txn_count_7d
        else:
            features.merchant_diversity_30d = 0.0

        # New user flag
        features.is_new_user = ub.days_since_first_txn < 30

        # Chargeback/dispute risk
        features.user_fraud_history_score = ub.chargeback_rate_30d * 0.6 + ub.dispute_rate_30d * 0.4

        # Time since last transaction (if we have typical hours)
        if features.timestamp and ub.typical_txn_hours:
            current_hour = features.timestamp.hour
            hours_since_typical = min((current_hour - h) % 24 for h in ub.typical_txn_hours)
            features.hours_since_typical_txn = hours_since_typical
        else:
            features.hours_since_typical_txn = 12  # Default

    @classmethod
    def _compute_device_derived(cls, features: UPIFeatures) -> None:
        """Compute device/network derived features."""
        dn = features.device_network

        # Device trust composite
        device_risk = cls.DEVICE_TYPE_RISK.get(dn.device_type.lower(), 0.5)
        features.device_composite_risk = (
            (1.0 - dn.device_trust_score) * 0.4
            + device_risk * 0.2
            + (1.0 - dn.ip_reputation_score) * 0.2
            + (1.0 if dn.is_vpn_proxy else 0.0) * 0.1
            + (1.0 if dn.is_tor else 0.0) * 0.05
            + (1.0 if dn.rooted_jailbroken else 0.0) * 0.05
        )

        # Network anomaly score
        features.network_anomaly_score = (
            (1.0 if dn.ip_geolocation_mismatch else 0.0) * 0.5
            + (1.0 if dn.is_vpn_proxy else 0.0) * 0.3
            + (1.0 if dn.is_tor else 0.0) * 0.2
        )

        # Device change risk
        features.device_change_risk = (1.0 if dn.sim_change_recent else 0.0) * 0.5 + (
            1.0 if dn.device_change_recent else 0.0
        ) * 0.5

        # ASN risk (could be expanded with ASN reputation DB)
        features.asn_risk_score = cls._get_asn_risk(dn.ip_asn)

    @classmethod
    def _compute_merchant_derived(cls, features: UPIFeatures) -> None:
        """Compute merchant derived features."""
        m = features.merchant

        # Category risk
        features.merchant_category_risk = cls.MERCHANT_CATEGORY_RISK.get(
            m.merchant_category.lower(), 0.5
        )

        # Merchant age risk (newer = riskier)
        if m.merchant_age_days > 0:
            features.merchant_age_risk = 1.0 / (1.0 + math.log1p(m.merchant_age_days / 30.0))
        else:
            features.merchant_age_risk = 1.0  # Brand new = max risk

        # Volume anomaly (if very high volume for age)
        if m.merchant_age_days > 0:
            volume_per_day = m.merchant_txn_volume_30d / max(m.merchant_age_days / 30.0, 1)
            features.merchant_velocity_risk = min(volume_per_day / 10000.0, 1.0)
        else:
            features.merchant_velocity_risk = 1.0

        # Verification status risk
        features.merchant_verification_risk = 0.0 if m.is_verified_merchant else 0.5
        if m.kyc_status == "verified":
            features.merchant_verification_risk = 0.0
        elif m.kyc_status == "pending":
            features.merchant_verification_risk = 0.3
        elif m.kyc_status == "rejected":
            features.merchant_verification_risk = 0.8
        else:
            features.merchant_verification_risk = 0.5

        # Composite merchant risk
        features.merchant_composite_risk = (
            m.merchant_risk_score * 0.3
            + features.merchant_category_risk * 0.15
            + features.merchant_age_risk * 0.15
            + features.merchant_velocity_risk * 0.1
            + m.merchant_chargeback_rate * 0.15
            + m.merchant_dispute_rate * 0.1
            + features.merchant_verification_risk * 0.05
        )

    @classmethod
    def _compute_velocity_derived(cls, features: UPIFeatures) -> None:
        """Compute velocity derived features."""
        v = features.velocity

        # Velocity anomaly score (burst detection)
        if v.vpa_velocity_1h > 0:
            features.velocity_anomaly_score = min(
                v.vpa_velocity_5m / max(v.vpa_velocity_1h / 12.0, 1), 2.0
            )
        else:
            features.velocity_anomaly_score = 0.0

        # Amount velocity anomaly
        if v.amount_velocity_1h > 0:
            features.amount_velocity_anomaly = min(
                v.amount_velocity_5m / max(v.amount_velocity_1h / 12.0, 1), 2.0
            )
        else:
            features.amount_velocity_anomaly = 0.0

        # Merchant velocity anomaly
        if v.unique_merchants_24h > 0:
            features.merchant_velocity_anomaly = min(
                v.unique_merchants_1h / max(v.unique_merchants_24h / 24.0, 1), 2.0
            )
        else:
            features.merchant_velocity_anomaly = 0.0

        # Failed transaction risk
        features.failed_txn_risk = min(v.failed_txn_rate_1h * 2.0, 1.0)

        # Composite velocity risk
        features.velocity_composite_risk = (
            features.velocity_anomaly_score * 0.3
            + features.amount_velocity_anomaly * 0.25
            + features.merchant_velocity_anomaly * 0.25
            + features.failed_txn_risk * 0.2
        )

    @classmethod
    def _compute_graph_derived(cls, features: UPIFeatures) -> None:
        """Compute graph derived features."""
        g = features.graph

        # Distance risk (closer = riskier for fraud rings)
        if g.user_merchant_graph_distance >= 0:
            features.graph_distance_risk = 1.0 / (1.0 + g.user_merchant_graph_distance)
        else:
            features.graph_distance_risk = 0.5  # Unknown

        # Cluster risk (already in g.vpa_cluster_risk)
        features.graph_cluster_risk = g.vpa_cluster_risk

        # Bridge node risk
        features.graph_bridge_risk = 1.0 if g.is_bridge_node else 0.0

        # PageRank anomaly (very high or very low = suspicious)
        if g.pagerank_score > 0:
            features.graph_pagerank_anomaly = abs(g.pagerank_score - 0.01) / 0.01
            features.graph_pagerank_anomaly = min(features.graph_pagerank_anomaly, 1.0)
        else:
            features.graph_pagerank_anomaly = 0.5

        # Composite graph risk
        features.graph_composite_risk = (
            features.graph_distance_risk * 0.3
            + features.graph_cluster_risk * 0.3
            + features.graph_bridge_risk * 0.2
            + features.graph_pagerank_anomaly * 0.2
        )

    @classmethod
    def _vpa_similarity(cls, vpa1: str, vpa2: str) -> float:
        """Compute VPA string similarity (0-1)."""
        # Extract handle part (before @)
        handle1 = vpa1.split("@")[0].lower()
        handle2 = vpa2.split("@")[0].lower()

        # Jaccard similarity on character bigrams
        def bigrams(s: str) -> set:
            return set(s[i : i + 2] for i in range(len(s) - 1))

        bg1 = bigrams(handle1)
        bg2 = bigrams(handle2)

        if not bg1 and not bg2:
            return 1.0
        if not bg1 or not bg2:
            return 0.0

        intersection = len(bg1 & bg2)
        union = len(bg1 | bg2)
        return intersection / union

    @classmethod
    def _get_asn_risk(cls, asn: str) -> float:
        """Get ASN risk score (placeholder for threat intel integration)."""
        # Known risky ASNs (hosting providers, VPNs, etc.)
        risky_asns = {
            "AS16276": 0.8,  # OVH
            "AS14061": 0.7,  # DigitalOcean
            "AS396982": 0.7,  # Google Cloud
            "AS16509": 0.6,  # Amazon AWS
            "AS13335": 0.5,  # Cloudflare
        }
        return risky_asns.get(asn.upper(), 0.2)

    @classmethod
    def compute_risk_vector(cls, features: UPIFeatures) -> np.ndarray:
        """
        Compute composite risk vector for model input.

        Returns 10-dimensional risk vector:
        [txn_risk, user_risk, device_risk, network_risk,
         merchant_risk, velocity_risk, graph_risk,
         behavioral_anomaly, amount_anomaly, composite]
        """
        risk_vec = np.zeros(10, dtype=np.float32)

        # Transaction risk
        risk_vec[0] = (
            (1.0 if features.is_night_txn else 0.0) * 0.3
            + (1.0 if features.is_high_value else 0.0) * 0.3
            + (1.0 if features.is_suspicious_amount else 0.0) * 0.2
            + (1.0 - features.payer_payee_similarity) * 0.2
        )

        # User behavioral risk
        if features.user_behavioral:
            ub = features.user_behavioral
            risk_vec[1] = (
                min(abs(features.amount_zscore_7d) / 3.0, 1.0) * 0.3
                + min(features.txn_frequency_score / 2.0, 1.0) * 0.2
                + (1.0 - features.merchant_diversity_7d) * 0.2
                + (1.0 if features.is_new_user else 0.0) * 0.15
                + features.user_fraud_history_score * 0.15
            )
        else:
            risk_vec[1] = 0.5

        # Device risk
        if features.device_network:
            risk_vec[2] = getattr(features, "device_composite_risk", 0.5)
        else:
            risk_vec[2] = 0.5

        # Network risk
        if features.device_network:
            risk_vec[3] = getattr(features, "network_anomaly_score", 0.5)
        else:
            risk_vec[3] = 0.5

        # Merchant risk
        if features.merchant:
            risk_vec[4] = getattr(features, "merchant_composite_risk", 0.5)
        else:
            risk_vec[4] = 0.5

        # Velocity risk
        if features.velocity:
            risk_vec[5] = getattr(features, "velocity_composite_risk", 0.5)
        else:
            risk_vec[5] = 0.5

        # Graph risk
        if features.graph:
            risk_vec[6] = getattr(features, "graph_composite_risk", 0.5)
        else:
            risk_vec[6] = 0.5

        # Behavioral anomaly (amount z-score based)
        risk_vec[7] = min(abs(getattr(features, "amount_zscore_7d", 0.0)) / 3.0, 1.0)

        # Amount anomaly
        risk_vec[8] = min(getattr(features, "amount_velocity_anomaly", 0.0), 1.0)

        # Composite (weighted average)
        weights = np.array([0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.05, 0.05])
        risk_vec[9] = float(np.dot(risk_vec[:9], weights))

        return risk_vec


class BatchFeatureComputer:
    """Vectorized batch feature computation for training/inference."""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size

    def compute_batch(self, features_list: List[UPIFeatures]) -> List[UPIFeatures]:
        """Compute derived features for batch of transactions."""
        for features in features_list:
            FeatureComputer.compute_all(features)
        return features_list

    def compute_risk_vectors(self, features_list: List[UPIFeatures]) -> np.ndarray:
        """Compute risk vectors for batch."""
        vectors = [FeatureComputer.compute_risk_vector(f) for f in features_list]
        return np.vstack(vectors)

    def compute_feature_arrays(self, features_list: List[UPIFeatures]) -> np.ndarray:
        """Compute model input arrays for batch."""
        arrays = [f.to_array() for f in features_list]
        return np.vstack(arrays)
