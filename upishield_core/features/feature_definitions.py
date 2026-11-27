"""
Feature Definitions for ScamShield AI Feast Feature Store.

Centralized feature definitions with metadata, transformations,
and validation rules for all fraud detection features.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class FeatureType(Enum):
    """Feature data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    EMBEDDING = "embedding"


class FeatureSource(Enum):
    """Source of the feature."""
    TRANSACTION = "transaction"
    USER_PROFILE = "user_profile"
    MERCHANT_PROFILE = "merchant_profile"
    DEVICE_FINGERPRINT = "device_fingerprint"
    NETWORK_INTELLIGENCE = "network_intelligence"
    DOCUMENT_ANALYSIS = "document_analysis"
    BEHAVIORAL = "behavioral"
    AGGREGATED = "aggregated"
    DERIVED = "derived"


@dataclass
class FeatureValidationRule:
    """Validation rule for a feature."""
    rule_type: str  # range, enum, null_check, custom
    parameters: Dict[str, Any]
    severity: str = "warning"  # error, warning
    message: str = ""


@dataclass
class FeatureDefinition:
    """Complete feature definition."""
    name: str
    display_name: str
    feature_type: FeatureType
    source: FeatureSource
    description: str
    default_value: Any = None
    validation_rules: List[FeatureValidationRule] = field(default_factory=list)
    transformation: Optional[str] = None  # SQL or Python expression
    tags: Dict[str, str] = field(default_factory=dict)
    owner: str = "ml-team"
    version: str = "1.0"
    enabled: bool = True
    sensitive: bool = False  # PII or sensitive data
    monitoring: bool = True  # Enable drift monitoring


# TRANSACTION FEATURES

TRANSACTION_FEATURES = [
    FeatureDefinition(
        name="txn_amount",
        display_name="Transaction Amount",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.TRANSACTION,
        description="Transaction amount in INR",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 10000000}, "error", "Amount must be positive"),
        ],
        tags={"domain": "payment", "importance": "critical"},
    ),
    FeatureDefinition(
        name="txn_amount_log",
        display_name="Log Transaction Amount",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Natural log of transaction amount + 1",
        transformation="ln(txn_amount + 1)",
        tags={"domain": "payment", "derived": "true"},
    ),
    FeatureDefinition(
        name="txn_amount_zscore",
        display_name="Amount Z-Score",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Z-score of amount vs user's historical transactions",
        transformation="(txn_amount - user_avg_txn_amount) / user_std_txn_amount",
        tags={"domain": "payment", "derived": "true", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="txn_currency",
        display_name="Currency",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.TRANSACTION,
        description="Transaction currency code",
        default_value="INR",
        validation_rules=[
            FeatureValidationRule("enum", {"values": ["INR", "USD", "EUR", "GBP"]}, "warning"),
        ],
        tags={"domain": "payment"},
    ),
    FeatureDefinition(
        name="txn_type",
        display_name="Transaction Type",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.TRANSACTION,
        description="Type of transaction",
        default_value="P2P",
        validation_rules=[
            FeatureValidationRule("enum", {"values": ["P2P", "P2M", "QR", "COLLECT", "REFUND"]}, "error"),
        ],
        tags={"domain": "payment", "importance": "high"},
    ),
    FeatureDefinition(
        name="txn_channel",
        display_name="Transaction Channel",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.TRANSACTION,
        description="Payment channel used",
        default_value="UPI",
        validation_rules=[
            FeatureValidationRule("enum", {"values": ["UPI", "IMPS", "NEFT", "RTGS", "CARD", "WALLET"]}, "error"),
        ],
        tags={"domain": "payment"},
    ),
    FeatureDefinition(
        name="txn_hour",
        display_name="Transaction Hour",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Hour of transaction (0-23)",
        transformation="HOUR(txn_timestamp)",
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 23}, "error"),
        ],
        tags={"domain": "temporal", "importance": "high"},
    ),
    FeatureDefinition(
        name="txn_day_of_week",
        display_name="Day of Week",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Day of week (0=Monday, 6=Sunday)",
        transformation="DAYOFWEEK(txn_timestamp)",
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 6}, "error"),
        ],
        tags={"domain": "temporal"},
    ),
    FeatureDefinition(
        name="txn_is_weekend",
        display_name="Is Weekend",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.DERIVED,
        description="Whether transaction occurred on weekend",
        transformation="txn_day_of_week IN (5,6)",
        tags={"domain": "temporal", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="txn_is_night",
        display_name="Is Night Transaction",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.DERIVED,
        description="Whether transaction occurred at night (10PM-6AM)",
        transformation="txn_hour >= 22 OR txn_hour <= 6",
        tags={"domain": "temporal", "anomaly": "true", "importance": "high"},
    ),
    FeatureDefinition(
        name="sender_vpa",
        display_name="Sender VPA",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.TRANSACTION,
        description="Sender's Virtual Payment Address",
        sensitive=True,
        tags={"domain": "identity"},
    ),
    FeatureDefinition(
        name="receiver_vpa",
        display_name="Receiver VPA",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.TRANSACTION,
        description="Receiver's Virtual Payment Address",
        sensitive=True,
        tags={"domain": "identity"},
    ),
    FeatureDefinition(
        name="vpa_domain_match",
        display_name="VPA Domain Match",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.DERIVED,
        description="Whether sender and receiver VPA domains match",
        transformation="DOMAIN(sender_vpa) = DOMAIN(receiver_vpa)",
        tags={"domain": "identity", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="vpa_is_known_merchant",
        display_name="Known Merchant VPA",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.MERCHANT_PROFILE,
        description="Whether receiver VPA is a known merchant",
        tags={"domain": "merchant", "importance": "high"},
    ),
    FeatureDefinition(
        name="txn_velocity_1h",
        display_name="Transaction Velocity (1hr)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of transactions by user in last hour",
        default_value=0,
        tags={"domain": "behavioral", "velocity": "true"},
    ),
    FeatureDefinition(
        name="txn_velocity_24h",
        display_name="Transaction Velocity (24hr)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of transactions by user in last 24 hours",
        default_value=0,
        tags={"domain": "behavioral", "velocity": "true"},
    ),
]


# USER FEATURES

USER_FEATURES = [
    FeatureDefinition(
        name="user_id",
        display_name="User ID",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.USER_PROFILE,
        description="Unique user identifier",
        sensitive=True,
        tags={"domain": "identity", "key": "true"},
    ),
    FeatureDefinition(
        name="user_age_days",
        display_name="Account Age (Days)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.USER_PROFILE,
        description="Days since account creation",
        default_value=0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 10000}, "warning"),
        ],
        tags={"domain": "identity", "importance": "high"},
    ),
    FeatureDefinition(
        name="user_kyc_level",
        display_name="KYC Level",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.USER_PROFILE,
        description="KYC verification level",
        default_value="none",
        validation_rules=[
            FeatureValidationRule("enum", {"values": ["none", "basic", "full", "enhanced"]}, "error"),
        ],
        tags={"domain": "compliance", "importance": "high"},
    ),
    FeatureDefinition(
        name="user_risk_score",
        display_name="User Risk Score",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Computed user risk score (0-1)",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "error"),
        ],
        transformation="ML_MODEL(user_features)",
        tags={"domain": "risk", "importance": "critical", "ml_generated": "true"},
    ),
    FeatureDefinition(
        name="user_txn_count_7d",
        display_name="Txn Count (7d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Transaction count in last 7 days",
        default_value=0,
        tags={"domain": "behavioral", "window": "7d"},
    ),
    FeatureDefinition(
        name="user_txn_count_30d",
        display_name="Txn Count (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Transaction count in last 30 days",
        default_value=0,
        tags={"domain": "behavioral", "window": "30d"},
    ),
    FeatureDefinition(
        name="user_txn_volume_7d",
        display_name="Txn Volume (7d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Total transaction volume in last 7 days",
        default_value=0.0,
        tags={"domain": "behavioral", "window": "7d"},
    ),
    FeatureDefinition(
        name="user_txn_volume_30d",
        display_name="Txn Volume (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Total transaction volume in last 30 days",
        default_value=0.0,
        tags={"domain": "behavioral", "window": "30d"},
    ),
    FeatureDefinition(
        name="user_avg_txn_amount",
        display_name="Avg Txn Amount",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Average transaction amount",
        default_value=0.0,
        tags={"domain": "behavioral"},
    ),
    FeatureDefinition(
        name="user_max_txn_amount",
        display_name="Max Txn Amount",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Maximum transaction amount",
        default_value=0.0,
        tags={"domain": "behavioral", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="user_unique_merchants_7d",
        display_name="Unique Merchants (7d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of unique merchants in last 7 days",
        default_value=0,
        tags={"domain": "behavioral", "diversity": "true"},
    ),
    FeatureDefinition(
        name="user_unique_vpas_7d",
        display_name="Unique VPAs (7d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of unique VPAs transacted with in last 7 days",
        default_value=0,
        tags={"domain": "behavioral", "diversity": "true"},
    ),
    FeatureDefinition(
        name="user_fraud_count_30d",
        display_name="Fraud Count (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of fraud incidents in last 30 days",
        default_value=0,
        tags={"domain": "risk", "importance": "critical"},
    ),
    FeatureDefinition(
        name="user_chargeback_count_30d",
        display_name="Chargeback Count (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of chargebacks in last 30 days",
        default_value=0,
        tags={"domain": "risk"},
    ),
    FeatureDefinition(
        name="user_device_count_30d",
        display_name="Device Count (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of unique devices used in last 30 days",
        default_value=0,
        tags={"domain": "device", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="user_ip_count_30d",
        display_name="IP Count (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of unique IPs used in last 30 days",
        default_value=0,
        tags={"domain": "network", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="user_preferred_txn_hour",
        display_name="Preferred Txn Hour",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="User's most common transaction hour",
        transformation="MODE(txn_hour) OVER user_history",
        tags={"domain": "behavioral", "pattern": "true"},
    ),
    FeatureDefinition(
        name="user_txn_amount_cv",
        display_name="Txn Amount Coeff. Variation",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Coefficient of variation of transaction amounts",
        transformation="user_std_txn_amount / user_avg_txn_amount",
        tags={"domain": "behavioral", "variability": "true"},
    ),
]


# MERCHANT FEATURES

MERCHANT_FEATURES = [
    FeatureDefinition(
        name="merchant_id",
        display_name="Merchant ID",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.MERCHANT_PROFILE,
        description="Unique merchant identifier",
        tags={"domain": "identity", "key": "true"},
    ),
    FeatureDefinition(
        name="merchant_category",
        display_name="Merchant Category",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.MERCHANT_PROFILE,
        description="Merchant category code (MCC)",
        default_value="5999",
        tags={"domain": "merchant"},
    ),
    FeatureDefinition(
        name="merchant_risk_score",
        display_name="Merchant Risk Score",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Computed merchant risk score (0-1)",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "error"),
        ],
        transformation="ML_MODEL(merchant_features)",
        tags={"domain": "risk", "importance": "critical", "ml_generated": "true"},
    ),
    FeatureDefinition(
        name="merchant_txn_count_7d",
        display_name="Merchant Txn Count (7d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Transaction count for merchant in last 7 days",
        default_value=0,
        tags={"domain": "merchant", "window": "7d"},
    ),
    FeatureDefinition(
        name="merchant_txn_volume_7d",
        display_name="Merchant Volume (7d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Transaction volume for merchant in last 7 days",
        default_value=0.0,
        tags={"domain": "merchant", "window": "7d"},
    ),
    FeatureDefinition(
        name="merchant_fraud_rate_30d",
        display_name="Merchant Fraud Rate (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Fraud rate for merchant in last 30 days",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "warning"),
        ],
        tags={"domain": "risk", "importance": "high"},
    ),
    FeatureDefinition(
        name="merchant_chargeback_rate_30d",
        display_name="Merchant Chargeback Rate (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Chargeback rate for merchant in last 30 days",
        default_value=0.0,
        tags={"domain": "risk"},
    ),
    FeatureDefinition(
        name="merchant_avg_txn_amount",
        display_name="Merchant Avg Amount",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Average transaction amount for merchant",
        default_value=0.0,
        tags={"domain": "merchant"},
    ),
    FeatureDefinition(
        name="merchant_unique_users_7d",
        display_name="Merchant Unique Users (7d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of unique users for merchant in last 7 days",
        default_value=0,
        tags={"domain": "merchant", "diversity": "true"},
    ),
    FeatureDefinition(
        name="merchant_is_new",
        display_name="New Merchant",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.DERIVED,
        description="Merchant onboarded less than 30 days ago",
        transformation="merchant_age_days < 30",
        tags={"domain": "merchant", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="merchant_age_days",
        display_name="Merchant Age (Days)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.MERCHANT_PROFILE,
        description="Days since merchant onboarding",
        default_value=0,
        tags={"domain": "merchant"},
    ),
]


# DEVICE FEATURES

DEVICE_FEATURES = [
    FeatureDefinition(
        name="device_id",
        display_name="Device ID",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.DEVICE_FINGERPRINT,
        description="Unique device fingerprint",
        sensitive=True,
        tags={"domain": "identity", "key": "true"},
    ),
    FeatureDefinition(
        name="device_type",
        display_name="Device Type",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.DEVICE_FINGERPRINT,
        description="Type of device",
        default_value="unknown",
        validation_rules=[
            FeatureValidationRule("enum", {"values": ["mobile", "desktop", "tablet", "unknown"]}, "warning"),
        ],
        tags={"domain": "device"},
    ),
    FeatureDefinition(
        name="device_os",
        display_name="Device OS",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.DEVICE_FINGERPRINT,
        description="Operating system",
        default_value="unknown",
        validation_rules=[
            FeatureValidationRule("enum", {"values": ["android", "ios", "windows", "macos", "linux", "unknown"]}, "warning"),
        ],
        tags={"domain": "device"},
    ),
    FeatureDefinition(
        name="device_os_version",
        display_name="OS Version",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.DEVICE_FINGERPRINT,
        description="Operating system version",
        tags={"domain": "device"},
    ),
    FeatureDefinition(
        name="device_is_rooted",
        display_name="Device Rooted",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.DEVICE_FINGERPRINT,
        description="Whether device is rooted/jailbroken",
        default_value=False,
        tags={"domain": "security", "importance": "high", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="device_is_emulator",
        display_name="Is Emulator",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.DEVICE_FINGERPRINT,
        description="Whether running in emulator",
        default_value=False,
        tags={"domain": "security", "importance": "critical", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="device_has_fraud_history",
        display_name="Device Fraud History",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.AGGREGATED,
        description="Whether device has been associated with fraud",
        default_value=False,
        tags={"domain": "risk", "importance": "critical"},
    ),
    FeatureDefinition(
        name="device_fraud_count_30d",
        display_name="Device Fraud Count (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Fraud incidents on this device in last 30 days",
        default_value=0,
        tags={"domain": "risk"},
    ),
    FeatureDefinition(
        name="device_user_count_30d",
        display_name="Device User Count (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Number of unique users on this device in last 30 days",
        default_value=0,
        tags={"domain": "device", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="device_age_days",
        display_name="Device Age (Days)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Days since device first seen",
        default_value=0,
        tags={"domain": "device"},
    ),
    FeatureDefinition(
        name="device_screen_resolution",
        display_name="Screen Resolution",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.DEVICE_FINGERPRINT,
        description="Device screen resolution",
        tags={"domain": "device"},
    ),
    FeatureDefinition(
        name="device_density",
        display_name="Screen Density",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DEVICE_FINGERPRINT,
        description="Screen pixel density",
        tags={"domain": "device"},
    ),
]


# NETWORK FEATURES

NETWORK_FEATURES = [
    FeatureDefinition(
        name="ip_address",
        display_name="IP Address",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.NETWORK_INTELLIGENCE,
        description="Client IP address",
        sensitive=True,
        tags={"domain": "network", "key": "true"},
    ),
    FeatureDefinition(
        name="ip_country",
        display_name="IP Country",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.NETWORK_INTELLIGENCE,
        description="Country code from IP geolocation",
        default_value="IN",
        tags={"domain": "network", "geolocation": "true"},
    ),
    FeatureDefinition(
        name="ip_region",
        display_name="IP Region",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.NETWORK_INTELLIGENCE,
        description="Region/state from IP geolocation",
        tags={"domain": "network", "geolocation": "true"},
    ),
    FeatureDefinition(
        name="ip_city",
        display_name="IP City",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.NETWORK_INTELLIGENCE,
        description="City from IP geolocation",
        tags={"domain": "network", "geolocation": "true"},
    ),
    FeatureDefinition(
        name="ip_asn",
        display_name="IP ASN",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.NETWORK_INTELLIGENCE,
        description="Autonomous System Number",
        default_value=0,
        tags={"domain": "network"},
    ),
    FeatureDefinition(
        name="ip_isp",
        display_name="IP ISP",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.NETWORK_INTELLIGENCE,
        description="Internet Service Provider",
        tags={"domain": "network"},
    ),
    FeatureDefinition(
        name="ip_is_vpn",
        display_name="IP is VPN/Proxy",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.NETWORK_INTELLIGENCE,
        description="Whether IP is a VPN or proxy",
        default_value=False,
        tags={"domain": "security", "importance": "high", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="ip_is_tor",
        display_name="IP is Tor",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.NETWORK_INTELLIGENCE,
        description="Whether IP is a Tor exit node",
        default_value=False,
        tags={"domain": "security", "importance": "critical", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="ip_is_datacenter",
        display_name="IP is Datacenter",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.NETWORK_INTELLIGENCE,
        description="Whether IP belongs to datacenter/cloud provider",
        default_value=False,
        tags={"domain": "security", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="ip_fraud_count_30d",
        display_name="IP Fraud Count (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Fraud incidents from this IP in last 30 days",
        default_value=0,
        tags={"domain": "risk"},
    ),
    FeatureDefinition(
        name="ip_user_count_30d",
        display_name="IP User Count (30d)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.AGGREGATED,
        description="Unique users from this IP in last 30 days",
        default_value=0,
        tags={"domain": "network", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="ip_distance_km",
        display_name="IP Distance (km)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Distance from user's usual location in km",
        default_value=0.0,
        transformation="HAVERSINE(user_usual_lat, user_usual_lon, ip_lat, ip_lon)",
        tags={"domain": "geolocation", "anomaly": "true", "importance": "high"},
    ),
    FeatureDefinition(
        name="ip_velocity_kmh",
        display_name="IP Velocity (km/h)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DERIVED,
        description="Implied travel velocity between transactions",
        transformation="ip_distance_km / hours_since_last_txn",
        tags={"domain": "geolocation", "anomaly": "true"},
    ),
]


# DOCUMENT FEATURES (from CV pipeline)

DOCUMENT_FEATURES = [
    FeatureDefinition(
        name="doc_type",
        display_name="Document Type",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Type of document analyzed",
        default_value="unknown",
        validation_rules=[
            FeatureValidationRule("enum", {"values": ["aadhaar", "pan", "payment_screenshot", "bank_statement", "salary_slip", "unknown"]}, "warning"),
        ],
        tags={"domain": "document"},
    ),
    FeatureDefinition(
        name="doc_ela_score",
        display_name="ELA Anomaly Score",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Error Level Analysis anomaly score (0-1)",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "error"),
        ],
        tags={"domain": "forensics", "importance": "high"},
    ),
    FeatureDefinition(
        name="doc_noise_score",
        display_name="Noise Inconsistency Score",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Noise pattern inconsistency score (0-1)",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "error"),
        ],
        tags={"domain": "forensics", "importance": "high"},
    ),
    FeatureDefinition(
        name="doc_copymove_score",
        display_name="Copy-Move Forgery Score",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Copy-move forgery detection score (0-1)",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "error"),
        ],
        tags={"domain": "forensics", "importance": "critical"},
    ),
    FeatureDefinition(
        name="doc_color_score",
        display_name="Color Inconsistency Score",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Color consistency analysis score (0-1)",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "error"),
        ],
        tags={"domain": "forensics"},
    ),
    FeatureDefinition(
        name="doc_edge_score",
        display_name="Edge Inconsistency Score",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Edge consistency analysis score (0-1)",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "error"),
        ],
        tags={"domain": "forensics"},
    ),
    FeatureDefinition(
        name="doc_ocr_confidence",
        display_name="OCR Confidence",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="OCR text extraction confidence (0-1)",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "warning"),
        ],
        tags={"domain": "ocr"},
    ),
    FeatureDefinition(
        name="doc_tampering_verdict",
        display_name="Tampering Verdict",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Overall tampering verdict",
        default_value="AUTHENTIC",
        validation_rules=[
            FeatureValidationRule("enum", {"values": ["AUTHENTIC", "SUSPICIOUS", "TAMPERED", "INCONCLUSIVE"]}, "error"),
        ],
        tags={"domain": "forensics", "importance": "critical"},
    ),
    FeatureDefinition(
        name="doc_risk_level",
        display_name="Document Risk Level",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Overall document risk level",
        default_value="LOW",
        validation_rules=[
            FeatureValidationRule("enum", {"values": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}, "error"),
        ],
        tags={"domain": "risk", "importance": "critical"},
    ),
    FeatureDefinition(
        name="doc_page_count",
        display_name="Page Count",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Number of pages in document",
        default_value=1,
        tags={"domain": "document"},
    ),
    FeatureDefinition(
        name="doc_file_size_kb",
        display_name="File Size (KB)",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Document file size in KB",
        default_value=0,
        tags={"domain": "document"},
    ),
    FeatureDefinition(
        name="doc_mime_type",
        display_name="MIME Type",
        feature_type=FeatureType.CATEGORICAL,
        source=FeatureSource.DOCUMENT_ANALYSIS,
        description="Document MIME type",
        tags={"domain": "document"},
    ),
]


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

BEHAVIORAL_FEATURES = [
    FeatureDefinition(
        name="user_session_duration",
        display_name="Session Duration",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.BEHAVIORAL,
        description="Current session duration in seconds",
        default_value=0,
        tags={"domain": "behavioral", "session": "true"},
    ),
    FeatureDefinition(
        name="user_typing_speed",
        display_name="Typing Speed",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.BEHAVIORAL,
        description="Keystrokes per minute",
        default_value=0.0,
        tags={"domain": "behavioral", "biometric": "true"},
    ),
    FeatureDefinition(
        name="user_mouse_velocity",
        display_name="Mouse Velocity",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.BEHAVIORAL,
        description="Average mouse movement velocity",
        default_value=0.0,
        tags={"domain": "behavioral", "biometric": "true"},
    ),
    FeatureDefinition(
        name="user_click_pattern_score",
        display_name="Click Pattern Score",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.BEHAVIORAL,
        description="Click pattern anomaly score (0-1)",
        default_value=0.0,
        validation_rules=[
            FeatureValidationRule("range", {"min": 0, "max": 1}, "warning"),
        ],
        tags={"domain": "behavioral", "anomaly": "true"},
    ),
    FeatureDefinition(
        name="user_navigation_depth",
        display_name="Navigation Depth",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.BEHAVIORAL,
        description="Number of pages visited in session",
        default_value=0,
        tags={"domain": "behavioral"},
    ),
    FeatureDefinition(
        name="user_time_to_txn",
        display_name="Time to Transaction",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.BEHAVIORAL,
        description="Time from session start to transaction (seconds)",
        default_value=0,
        tags={"domain": "behavioral", "intent": "true"},
    ),
    FeatureDefinition(
        name="user_repeat_txn_pattern",
        display_name="Repeat Transaction Pattern",
        feature_type=FeatureType.BOOLEAN,
        source=FeatureSource.DERIVED,
        description="Whether user repeats similar transaction patterns",
        transformation="SIMILARITY(last_txn, current_txn) > 0.9",
        tags={"domain": "behavioral", "pattern": "true", "anomaly": "true"},
    ),
]


# ============================================================
# FEATURE GROUPS / SERVICES
# ============================================================

FEATURE_SERVICES = {
    "fraud_detection": {
        "description": "All features for fraud detection model",
        "features": (
            [f.name for f in TRANSACTION_FEATURES] +
            [f.name for f in USER_FEATURES] +
            [f.name for f in MERCHANT_FEATURES] +
            [f.name for f in DEVICE_FEATURES] +
            [f.name for f in NETWORK_FEATURES] +
            [f.name for f in DOCUMENT_FEATURES]
        ),
    },
    "realtime_scoring": {
        "description": "Minimal features for real-time scoring (<10ms)",
        "features": [
            "txn_amount", "txn_hour", "txn_is_night", "txn_type", "txn_channel",
            "user_risk_score", "user_txn_count_7d", "user_age_days",
            "merchant_risk_score", "merchant_is_new",
            "device_is_rooted", "device_is_emulator", "device_has_fraud_history",
            "ip_is_vpn", "ip_is_tor", "ip_distance_km",
        ],
    },
    "model_training": {
        "description": "Features for offline model training",
        "features": (
            [f.name for f in TRANSACTION_FEATURES] +
            [f.name for f in USER_FEATURES] +
            [f.name for f in MERCHANT_FEATURES] +
            [f.name for f in DEVICE_FEATURES] +
            [f.name for f in NETWORK_FEATURES]
        ),
    },
    "document_verification": {
        "description": "Features for document verification model",
        "features": [f.name for f in DOCUMENT_FEATURES],
    },
    "behavioral_analysis": {
        "description": "Features for behavioral biometrics",
        "features": [f.name for f in BEHAVIORAL_FEATURES],
    },
}


# ALL FEATURES REGISTRY

ALL_FEATURE_DEFINITIONS = (
    TRANSACTION_FEATURES +
    USER_FEATURES +
    MERCHANT_FEATURES +
    DEVICE_FEATURES +
    NETWORK_FEATURES +
    DOCUMENT_FEATURES +
    BEHAVIORAL_FEATURES
)

FEATURE_REGISTRY = {f.name: f for f in ALL_FEATURE_DEFINITIONS}


def get_feature_definition(name: str) -> Optional[FeatureDefinition]:
    """Get feature definition by name."""
    return FEATURE_REGISTRY.get(name)


def get_features_by_source(source: FeatureSource) -> List[FeatureDefinition]:
    """Get all features from a specific source."""
    return [f for f in ALL_FEATURE_DEFINITIONS if f.source == source]


def get_features_by_type(feature_type: FeatureType) -> List[FeatureDefinition]:
    """Get all features of a specific type."""
    return [f for f in ALL_FEATURE_DEFINITIONS if f.feature_type == feature_type]


def get_enabled_features() -> List[FeatureDefinition]:
    """Get all enabled features."""
    return [f for f in ALL_FEATURE_DEFINITIONS if f.enabled]


def export_feature_definitions(output_path: str):
    """Export all feature definitions to JSON."""
    data = []
    for f in ALL_FEATURE_DEFINITIONS:
        data.append({
            "name": f.name,
            "display_name": f.display_name,
            "feature_type": f.feature_type.value,
            "source": f.source.value,
            "description": f.description,
            "default_value": f.default_value,
            "validation_rules": [
                {
                    "rule_type": r.rule_type,
                    "parameters": r.parameters,
                    "severity": r.severity,
                    "message": r.message,
                }
                for r in f.validation_rules
            ],
            "transformation": f.transformation,
            "tags": f.tags,
            "owner": f.owner,
            "version": f.version,
            "enabled": f.enabled,
            "sensitive": f.sensitive,
            "monitoring": f.monitoring,
        })
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_feature_values(feature_values: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate feature values against definitions."""
    violations = []
    for name, value in feature_values.items():
        definition = FEATURE_REGISTRY.get(name)
        if not definition:
            continue

        for rule in definition.validation_rules:
            if rule.rule_type == "range":
                min_val = rule.parameters.get("min")
                max_val = rule.parameters.get("max")
                if min_val is not None and value < min_val:
                    violations.append({
                        "feature": name,
                        "rule": "range",
                        "severity": rule.severity,
                        "message": f"Value {value} below minimum {min_val}",
                    })
                if max_val is not None and value > max_val:
                    violations.append({
                        "feature": name,
                        "rule": "range",
                        "severity": rule.severity,
                        "message": f"Value {value} above maximum {max_val}",
                    })
            elif rule.rule_type == "enum":
                allowed = rule.parameters.get("values", [])
                if value not in allowed:
                    violations.append({
                        "feature": name,
                        "rule": "enum",
                        "severity": rule.severity,
                        "message": f"Value {value} not in allowed values {allowed}",
                    })
            elif rule.rule_type == "null_check":
                if value is None:
                    violations.append({
                        "feature": name,
                        "rule": "null_check",
                        "severity": rule.severity,
                        "message": rule.message or f"Feature {name} cannot be null",
                    })

    return violations