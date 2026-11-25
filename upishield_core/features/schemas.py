"""
Pydantic schemas for UPI feature definitions.

All features used in the fraud detection pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class UserBehavioralFeatures:
    """User historical behavior features."""

    user_id: str
    avg_txn_amount_7d: float = 0.0
    avg_txn_amount_30d: float = 0.0
    txn_count_1h: int = 0
    txn_count_24h: int = 0
    txn_count_7d: int = 0
    unique_merchants_7d: int = 0
    unique_merchants_30d: int = 0
    favorite_vpa_similarity: float = 0.0  # 0-1, similarity to usual payees
    typical_txn_hours: List[int] = field(default_factory=list)  # Hours user usually transacts
    typical_txn_amounts: List[float] = field(default_factory=list)  # Usual amount ranges
    days_since_first_txn: int = 0
    chargeback_rate_30d: float = 0.0
    dispute_rate_30d: float = 0.0


@dataclass
class DeviceNetworkFeatures:
    """Device and network context features."""

    device_id: str
    device_trust_score: float = 0.5  # 0-1, from device fingerprinting
    device_type: str = "unknown"  # android, ios, web, unknown
    os_version: str = "unknown"
    app_version: str = "unknown"
    ip_address: str = ""
    ip_reputation_score: float = 0.5  # 0-1, from threat intel
    ip_geolocation_mismatch: bool = False  # IP location != registered location
    ip_asn: str = ""  # ISP/organization
    is_vpn_proxy: bool = False
    is_tor: bool = False
    sim_change_recent: bool = False  # SIM changed in last 7 days
    device_change_recent: bool = False  # New device in last 7 days
    rooted_jailbroken: bool = False
    emulator_detected: bool = False


@dataclass
class MerchantFeatures:
    """Merchant/payee risk features."""

    merchant_vpa: str
    merchant_category: str = "unknown"
    merchant_risk_score: float = 0.5  # 0-1, from historical data
    merchant_age_days: int = 0
    merchant_txn_volume_30d: int = 0
    merchant_unique_users_30d: int = 0
    merchant_chargeback_rate: float = 0.0
    merchant_dispute_rate: float = 0.0
    merchant_refund_rate: float = 0.0
    is_verified_merchant: bool = False
    kyc_status: str = "unknown"  # verified, pending, rejected, unknown


@dataclass
class VelocityFeatures:
    """Real-time velocity/rate features."""

    vpa_velocity_1m: int = 0
    vpa_velocity_5m: int = 0
    vpa_velocity_15m: int = 0
    vpa_velocity_1h: int = 0
    vpa_velocity_24h: int = 0
    amount_velocity_1m: float = 0.0
    amount_velocity_5m: float = 0.0
    amount_velocity_1h: float = 0.0
    amount_velocity_24h: float = 0.0
    unique_merchants_1h: int = 0
    unique_merchants_24h: int = 0
    failed_txn_count_1h: int = 0
    failed_txn_rate_1h: float = 0.0


@dataclass
class GraphFeatures:
    """Graph-based features from user-merchant network."""

    user_merchant_graph_distance: int = -1  # Shortest path, -1 = no connection
    user_merchant_common_neighbors: int = 0
    vpa_cluster_id: int = -1
    vpa_cluster_risk: float = 0.5  # Risk score of VPA's cluster
    vpa_cluster_size: int = 0
    pagerank_score: float = 0.0
    community_id: int = -1
    is_bridge_node: bool = False  # Connects disparate communities


class UPIFeatures(BaseModel):
    """
    Complete feature set for UPI transaction scoring.

    All 50+ features organized by category.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Transaction context
    txn_id: str
    payer_vpa: str
    payee_vpa: str
    amount: float
    currency: str = "INR"
    txn_type: str = "P2M"  # P2P, P2M, M2P
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Feature categories (populated by FeatureStore)
    user_behavioral: Optional[UserBehavioralFeatures] = None
    device_network: Optional[DeviceNetworkFeatures] = None
    merchant: Optional[MerchantFeatures] = None
    velocity: Optional[VelocityFeatures] = None
    graph: Optional[GraphFeatures] = None

    # Computed/derived features (filled by FeatureComputer)
    # Transaction-level
    is_night_txn: bool = False
    is_high_value: bool = False
    is_very_high_value: bool = False
    is_round_amount: bool = False
    is_suspicious_amount: bool = False
    payer_payee_similarity: float = 0.0
    hour_risk_score: float = 0.5
    amount_zscore_7d: float = 0.0
    amount_zscore_30d: float = 0.0
    velocity_anomaly_score: float = 0.0

    # User behavioral derived
    txn_frequency_score: float = 0.0
    merchant_diversity_7d: float = 0.0
    merchant_diversity_30d: float = 0.0
    is_new_user: bool = False
    user_fraud_history_score: float = 0.0
    hours_since_typical_txn: float = 12.0

    # Device/Network derived
    device_composite_risk: float = 0.5
    network_anomaly_score: float = 0.0
    device_change_risk: float = 0.0
    asn_risk_score: float = 0.2

    # Merchant derived
    merchant_category_risk: float = 0.5
    merchant_age_risk: float = 1.0
    merchant_velocity_risk: float = 1.0
    merchant_verification_risk: float = 0.5
    merchant_composite_risk: float = 0.5

    # Velocity derived
    amount_velocity_anomaly: float = 0.0
    merchant_velocity_anomaly: float = 0.0
    failed_txn_risk: float = 0.0
    velocity_composite_risk: float = 0.0

    # Graph derived
    graph_distance_risk: float = 0.5
    graph_cluster_risk: float = 0.5
    graph_bridge_risk: float = 0.0
    graph_pagerank_anomaly: float = 0.5
    graph_composite_risk: float = 0.5

    def to_array(self) -> "np.ndarray":
        """Convert to flat feature array for model input (50 dims)."""
        import numpy as np

        arr = np.zeros(50, dtype=np.float32)

        # Transaction (0-9)
        arr[0] = self.amount / 100000.0  # Normalize
        arr[1] = self.timestamp.hour / 24.0
        arr[2] = self.timestamp.weekday() / 7.0
        arr[3] = 1.0 if self.is_night_txn else 0.0
        arr[4] = 1.0 if self.is_high_value else 0.0
        arr[5] = hash(self.payer_vpa) % 1000 / 1000.0
        arr[6] = hash(self.payee_vpa) % 1000 / 1000.0

        # User behavioral (10-19)
        if self.user_behavioral:
            ub = self.user_behavioral
            arr[10] = ub.avg_txn_amount_7d / 100000.0
            arr[11] = ub.txn_count_1h / 100.0
            arr[12] = ub.txn_count_24h / 500.0
            arr[13] = ub.unique_merchants_7d / 100.0
            arr[14] = ub.favorite_vpa_similarity
            arr[15] = ub.chargeback_rate_30d
            arr[16] = ub.dispute_rate_30d
            arr[17] = ub.days_since_first_txn / 3650.0

        # Device/Network (20-29)
        if self.device_network:
            dn = self.device_network
            arr[20] = dn.device_trust_score
            arr[21] = dn.ip_reputation_score
            arr[22] = 1.0 if dn.ip_geolocation_mismatch else 0.0
            arr[23] = 1.0 if dn.sim_change_recent else 0.0
            arr[24] = 1.0 if dn.device_change_recent else 0.0
            arr[25] = 1.0 if dn.is_vpn_proxy else 0.0
            arr[26] = 1.0 if dn.rooted_jailbroken else 0.0

        # Merchant (30-39)
        if self.merchant:
            m = self.merchant
            arr[30] = m.merchant_risk_score
            arr[31] = hash(m.merchant_category) % 100 / 100.0
            arr[32] = m.merchant_age_days / 3650.0
            arr[33] = m.merchant_chargeback_rate
            arr[34] = m.merchant_dispute_rate
            arr[35] = 1.0 if m.is_verified_merchant else 0.0

        # Velocity (40-45)
        if self.velocity:
            v = self.velocity
            arr[40] = v.vpa_velocity_5m / 100.0
            arr[41] = v.vpa_velocity_1h / 500.0
            arr[42] = v.amount_velocity_1h / 100000.0
            arr[43] = v.unique_merchants_1h / 50.0
            arr[44] = v.failed_txn_rate_1h

        # Graph (46-49)
        if self.graph:
            g = self.graph
            arr[46] = (
                g.user_merchant_graph_distance / 10.0
                if g.user_merchant_graph_distance >= 0
                else 1.0
            )
            arr[47] = g.vpa_cluster_risk
            arr[48] = g.pagerank_score
            arr[49] = 1.0 if g.is_bridge_node else 0.0

        return arr.reshape(1, -1)

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for logging/debugging."""
        arr = self.to_array().flatten()
        feature_names = [
            "amount_norm",
            "hour_norm",
            "weekday_norm",
            "is_night",
            "is_high_value",
            "payer_vpa_hash",
            "payee_vpa_hash",
            "avg_amt_7d",
            "txn_1h",
            "txn_24h",
            "unique_merchants_7d",
            "fav_vpa_sim",
            "chargeback_rate",
            "dispute_rate",
            "days_since_first",
            "device_trust",
            "ip_reputation",
            "geo_mismatch",
            "sim_change",
            "device_change",
            "is_vpn",
            "rooted",
            "merchant_risk",
            "merchant_cat",
            "merchant_age",
            "merchant_chargeback",
            "merchant_dispute",
            "merchant_verified",
            "vel_5m",
            "vel_1h",
            "amt_vel_1h",
            "unique_merch_1h",
            "failed_rate",
            "graph_dist",
            "cluster_risk",
            "pagerank",
            "is_bridge",
        ] + [f"unused_{i}" for i in range(50 - 41)]
        return dict(zip(feature_names, arr.tolist()))
