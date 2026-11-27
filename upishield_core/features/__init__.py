"""
ScamShield AI - Feature Engineering Module.

Provides feature definitions, Feast feature store integration,
and online feature serving for real-time fraud detection.
"""

from .feast_store import (
    FeastFeatureStore,
    FeatureDefinition,
    FeatureCategory,
    get_feature_store,
    TRANSACTION_FEATURES,
    USER_FEATURES,
    MERCHANT_FEATURES,
    DEVICE_FEATURES,
    NETWORK_FEATURES,
    DOCUMENT_FEATURES,
    ALL_FEATURES,
)

from .feature_definitions import (
    FeatureDefinition as FDFeatureDefinition,
    FeatureType,
    FeatureSource,
    FeatureValidationRule,
    FEATURE_REGISTRY,
    FEATURE_SERVICES,
    ALL_FEATURE_DEFINITIONS,
    TRANSACTION_FEATURES as FD_TRANSACTION_FEATURES,
    USER_FEATURES as FD_USER_FEATURES,
    MERCHANT_FEATURES as FD_MERCHANT_FEATURES,
    DEVICE_FEATURES as FD_DEVICE_FEATURES,
    NETWORK_FEATURES as FD_NETWORK_FEATURES,
    DOCUMENT_FEATURES as FD_DOCUMENT_FEATURES,
    BEHAVIORAL_FEATURES,
    get_feature_definition,
    get_features_by_source,
    get_features_by_type,
    get_enabled_features,
    export_feature_definitions,
    validate_feature_values,
)

from .online_store import (
    OnlineFeatureStore,
    OnlineStoreConfig,
    StoreBackend,
    FeatureVector,
    FeatureServer,
    get_online_store,
    create_feature_server,
)

# Legacy imports (for backward compatibility)
try:
    from .compute import FeatureComputer
    from .schemas import (
        DeviceNetworkFeatures,
        GraphFeatures,
        MerchantFeatures,
        UserBehavioralFeatures,
        VelocityFeatures,
    )
    from .store import FeatureStore, UPIFeatures
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

__all__ = [
    # Feast Store
    "FeastFeatureStore",
    "FeatureDefinition",
    "FeatureCategory",
    "get_feature_store",
    "TRANSACTION_FEATURES",
    "USER_FEATURES",
    "MERCHANT_FEATURES",
    "DEVICE_FEATURES",
    "NETWORK_FEATURES",
    "DOCUMENT_FEATURES",
    "ALL_FEATURES",
    # Feature Definitions
    "FDFeatureDefinition",
    "FeatureType",
    "FeatureSource",
    "FeatureValidationRule",
    "FEATURE_REGISTRY",
    "FEATURE_SERVICES",
    "ALL_FEATURE_DEFINITIONS",
    "FD_TRANSACTION_FEATURES",
    "FD_USER_FEATURES",
    "FD_MERCHANT_FEATURES",
    "FD_DEVICE_FEATURES",
    "FD_NETWORK_FEATURES",
    "FD_DOCUMENT_FEATURES",
    "BEHAVIORAL_FEATURES",
    "get_feature_definition",
    "get_features_by_source",
    "get_features_by_type",
    "get_enabled_features",
    "export_feature_definitions",
    "validate_feature_values",
    # Online Store
    "OnlineFeatureStore",
    "OnlineStoreConfig",
    "StoreBackend",
    "FeatureVector",
    "FeatureServer",
    "get_online_store",
    "create_feature_server",
]

if LEGACY_AVAILABLE:
    __all__.extend([
        "FeatureStore",
        "UPIFeatures",
        "UserBehavioralFeatures",
        "DeviceNetworkFeatures",
        "MerchantFeatures",
        "VelocityFeatures",
        "GraphFeatures",
        "FeatureComputer",
    ])