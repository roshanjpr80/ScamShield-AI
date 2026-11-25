"""
Decision Engine for UPI Fraud Detection.

Combines:
- ML Ensemble (XGBoost + LightGBM + Transformer + GNN)
- Deterministic Rules (velocity, blocklists, patterns)
- Threat Intelligence (IP reputation, device reputation, etc.)
- Explainability (SHAP values, rule traces)
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np

from upishield_core.api.schemas import FraudDecision, RiskLevel, UPITransaction
from upishield_core.features.compute import FeatureComputer
from upishield_core.features.schemas import UPIFeatures
from upishield_core.models.ensemble import ONNXEnsemble
from upishield_core.models.registry import ModelRegistry


class DecisionSource(str, Enum):
    """Source of decision component."""

    ML_ENSEMBLE = "ml_ensemble"
    DETERMINISTIC_RULES = "deterministic_rules"
    THREAT_INTEL = "threat_intel"
    MANUAL_REVIEW = "manual_review"


@dataclass
class RuleResult:
    """Result of a deterministic rule evaluation."""

    rule_id: str
    rule_name: str
    triggered: bool
    score: float  # 0-1, contribution to fraud score
    severity: str  # low, medium, high, critical
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatIntelResult:
    """Result from threat intelligence lookup."""

    source: str  # ip_reputation, device_reputation, vpa_reputation, etc.
    risk_score: float  # 0-1
    labels: List[str]  # e.g., ["vpn", "tor", "hosting", "malware"]
    confidence: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext:
    """Context for decision making."""

    transaction: UPITransaction
    features: UPIFeatures
    ml_score: float
    ml_confidence: float
    rule_results: List[RuleResult]
    threat_intel_results: List[ThreatIntelResult]
    risk_level: RiskLevel
    final_score: float
    decision: FraudDecision
    explanation: Dict[str, Any]
    latency_ms: float
    model_scores: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RuleEngine:
    """
    Deterministic rule engine for fraud detection.

    Rules are evaluated in priority order. Each rule contributes to the final score.
    """

    # Rule definitions: (rule_id, name, weight, severity, condition_fn)
    RULES = [
        # Velocity rules (high priority)
        (
            "VEL_001",
            "Extreme Velocity (5m > 20)",
            0.3,
            "critical",
            lambda ctx: ctx.features.velocity and ctx.features.velocity.vpa_velocity_5m > 20,
        ),
        (
            "VEL_002",
            "High Velocity (1h > 100)",
            0.2,
            "high",
            lambda ctx: ctx.features.velocity and ctx.features.velocity.vpa_velocity_1h > 100,
        ),
        (
            "VEL_003",
            "Velocity Burst (5m vs 1h ratio)",
            0.15,
            "medium",
            lambda ctx: (
                ctx.features.velocity
                and ctx.features.velocity.vpa_velocity_1h > 0
                and ctx.features.velocity.vpa_velocity_5m
                / (ctx.features.velocity.vpa_velocity_1h / 12.0)
                > 5
            ),
        ),
        # Amount anomaly rules
        (
            "AMT_001",
            "Very High Value (>2L)",
            0.25,
            "high",
            lambda ctx: ctx.features.amount > 200000,
        ),
        (
            "AMT_002",
            "High Value (>50k) + Night",
            0.2,
            "medium",
            lambda ctx: ctx.features.amount > 50000 and ctx.features.is_night_txn,
        ),
        (
            "AMT_003",
            "Amount Z-Score > 5",
            0.2,
            "high",
            lambda ctx: abs(ctx.features.amount_zscore_7d) > 5,
        ),
        (
            "AMT_004",
            "Suspicious Round Amount",
            0.1,
            "low",
            lambda ctx: ctx.features.is_suspicious_amount,
        ),
        # New user rules
        (
            "USR_001",
            "New User (<7 days) + High Value",
            0.2,
            "high",
            lambda ctx: (
                ctx.features.user_behavioral
                and ctx.features.user_behavioral.days_since_first_txn < 7
                and ctx.features.amount > 10000
            ),
        ),
        (
            "USR_002",
            "New User + New Device",
            0.15,
            "medium",
            lambda ctx: (
                ctx.features.user_behavioral
                and ctx.features.user_behavioral.days_since_first_txn < 30
                and ctx.features.device_network
                and ctx.features.device_network.device_change_recent
            ),
        ),
        (
            "USR_003",
            "Chargeback History",
            0.25,
            "high",
            lambda ctx: (
                ctx.features.user_behavioral
                and ctx.features.user_behavioral.chargeback_rate_30d > 0.1
            ),
        ),
        # Device/Network rules
        (
            "DEV_001",
            "Tor Exit Node",
            0.4,
            "critical",
            lambda ctx: ctx.features.device_network and ctx.features.device_network.is_tor,
        ),
        (
            "DEV_002",
            "VPN/Proxy + High Value",
            0.2,
            "medium",
            lambda ctx: (
                ctx.features.device_network
                and ctx.features.device_network.is_vpn_proxy
                and ctx.features.amount > 25000
            ),
        ),
        (
            "DEV_003",
            "Rooted/Jailbroken Device",
            0.15,
            "medium",
            lambda ctx: (
                ctx.features.device_network and ctx.features.device_network.rooted_jailbroken
            ),
        ),
        (
            "DEV_004",
            "Geo Mismatch + New Device",
            0.2,
            "high",
            lambda ctx: (
                ctx.features.device_network
                and ctx.features.device_network.ip_geolocation_mismatch
                and ctx.features.device_network.device_change_recent
            ),
        ),
        # Merchant rules
        (
            "MER_001",
            "Unverified High-Risk Merchant",
            0.2,
            "medium",
            lambda ctx: (
                ctx.features.merchant
                and not ctx.features.merchant.is_verified_merchant
                and ctx.features.merchant.merchant_risk_score > 0.7
            ),
        ),
        (
            "MER_002",
            "New Merchant (<30 days) + High Volume",
            0.15,
            "medium",
            lambda ctx: (
                ctx.features.merchant
                and ctx.features.merchant.merchant_age_days < 30
                and ctx.features.merchant.merchant_txn_volume_30d > 1000
            ),
        ),
        (
            "MER_003",
            "High Chargeback Merchant",
            0.2,
            "high",
            lambda ctx: (
                ctx.features.merchant and ctx.features.merchant.merchant_chargeback_rate > 0.05
            ),
        ),
        # Graph rules
        (
            "GRF_001",
            "Direct Fraud Ring Connection",
            0.35,
            "critical",
            lambda ctx: (
                ctx.features.graph
                and ctx.features.graph.user_merchant_graph_distance == 1
                and ctx.features.graph.vpa_cluster_risk > 0.8
            ),
        ),
        (
            "GRF_002",
            "Bridge Node in Fraud Cluster",
            0.25,
            "high",
            lambda ctx: (
                ctx.features.graph
                and ctx.features.graph.is_bridge_node
                and ctx.features.graph.vpa_cluster_risk > 0.6
            ),
        ),
        (
            "GRF_003",
            "High-Risk Cluster Member",
            0.15,
            "medium",
            lambda ctx: ctx.features.graph and ctx.features.graph.vpa_cluster_risk > 0.8,
        ),
        # Pattern rules
        (
            "PTN_001",
            "Rapid Succession to Same Merchant",
            0.2,
            "medium",
            lambda ctx: (
                ctx.features.velocity
                and ctx.features.velocity.unique_merchants_1h == 1
                and ctx.features.velocity.vpa_velocity_5m > 5
            ),
        ),
        (
            "PTN_002",
            "Amount Structuring Pattern",
            0.15,
            "medium",
            lambda ctx: (
                ctx.features.is_suspicious_amount and ctx.features.velocity.vpa_velocity_1h > 10
            ),
        ),
    ]

    def __init__(self, custom_rules: List[Tuple] = None):
        self.rules = self.RULES
        if custom_rules:
            self.rules.extend(custom_rules)

    def evaluate(self, transaction: UPITransaction, features: UPIFeatures) -> List[RuleResult]:
        """Evaluate all rules against transaction features."""
        results = []

        # Create a context object for rule evaluation
        class RuleContext:
            def __init__(self, txn, feats):
                self.transaction = txn
                self.features = feats

        ctx = RuleContext(transaction, features)

        for rule_id, name, weight, severity, condition in self.rules:
            try:
                triggered = condition(ctx)
            except Exception:
                # Rule evaluation error - log and continue
                triggered = False

            if triggered:
                results.append(
                    RuleResult(
                        rule_id=rule_id,
                        rule_name=name,
                        triggered=True,
                        score=weight,
                        severity=severity,
                        description=f"Rule {rule_id}: {name} triggered",
                        metadata={"weight": weight, "severity": severity},
                    )
                )

        return results

    def compute_rule_score(self, rule_results: List[RuleResult]) -> float:
        """Compute composite score from rule results."""
        if not rule_results:
            return 0.0

        # Weighted sum with diminishing returns
        total_weight = sum(r.score for r in rule_results)
        # Apply sigmoid to bound between 0-1
        return 1.0 / (1.0 + np.exp(-total_weight * 2.0))


class ThreatIntelligenceClient:
    """
    Threat intelligence integration client.

    In production, this would call external APIs:
    - IP reputation (AbuseIPDB, VirusTotal, MaxMind)
    - Device fingerprinting (FingerprintJS, DeviceAtlas)
    - VPA reputation (internal consortium, NPCI)
    - Phone reputation (Truecaller, internal)
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._cache: Dict[str, Tuple[ThreatIntelResult, float]] = {}
        self.cache_ttl = 300  # 5 minutes

    async def lookup_ip(self, ip: str) -> ThreatIntelResult:
        """Look up IP reputation."""
        cache_key = f"ip:{ip}"
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return result

        # Placeholder - integrate with real threat intel APIs
        # Example: AbuseIPDB, VirusTotal, GreyNoise, etc.
        risk_score = 0.0
        labels = []
        confidence = 0.5

        # Known malicious IP ranges (placeholder)
        malicious_prefixes = ["10.0.0.", "192.168."]  # Private - for demo
        for prefix in malicious_prefixes:
            if ip.startswith(prefix):
                risk_score = 0.3
                labels.append("private_range")
                break

        result = ThreatIntelResult(
            source="ip_reputation",
            risk_score=risk_score,
            labels=labels,
            confidence=confidence,
            metadata={"ip": ip},
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def lookup_device(self, device_id: str) -> ThreatIntelResult:
        """Look up device reputation."""
        cache_key = f"device:{device_id}"
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return result

        # Placeholder - integrate with device fingerprinting services
        risk_score = 0.0
        labels = []
        confidence = 0.4

        result = ThreatIntelResult(
            source="device_reputation",
            risk_score=risk_score,
            labels=labels,
            confidence=confidence,
            metadata={"device_id": device_id},
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def lookup_vpa(self, vpa: str) -> ThreatIntelResult:
        """Look up VPA reputation."""
        cache_key = f"vpa:{vpa}"
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return result

        # Placeholder - integrate with NPCI/consortium data
        risk_score = 0.0
        labels = []
        confidence = 0.3

        # Check for known suspicious patterns
        handle = vpa.split("@")[0].lower()
        suspicious_patterns = ["test", "demo", "admin", "root", "fraud", "scam"]
        for pattern in suspicious_patterns:
            if pattern in handle:
                risk_score = 0.4
                labels.append(f"suspicious_handle:{pattern}")
                break

        result = ThreatIntelResult(
            source="vpa_reputation",
            risk_score=risk_score,
            labels=labels,
            confidence=confidence,
            metadata={"vpa": vpa},
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def lookup_all(self, transaction: UPITransaction) -> List[ThreatIntelResult]:
        """Run all threat intel lookups in parallel."""
        tasks = []
        if transaction.client_ip:
            tasks.append(self.lookup_ip(transaction.client_ip))
        if transaction.device_id:
            tasks.append(self.lookup_device(transaction.device_id))
        tasks.append(self.lookup_vpa(transaction.payer_vpa))
        tasks.append(self.lookup_vpa(transaction.payee_vpa))
        return await asyncio.gather(*tasks)


class DecisionEngine:
    """
    Main Decision Engine combining ML + Rules + Threat Intel.

    Flow:
    1. Fetch features from FeatureStore
    2. Compute derived features
    3. Run ML ensemble inference
    4. Evaluate deterministic rules
    5. Query threat intelligence
    6. Combine all signals with weighted ensemble
    7. Apply decision thresholds
    8. Generate explanation
    """

    def __init__(
        self,
        ensemble: ONNXEnsemble,
        registry: ModelRegistry,
        rule_engine: RuleEngine = None,
        threat_intel: ThreatIntelligenceClient = None,
        config: Dict[str, Any] = None,
    ):
        self.ensemble = ensemble
        self.registry = registry
        self.rule_engine = rule_engine or RuleEngine()
        self.threat_intel = threat_intel or ThreatIntelligenceClient()
        self.config = config or self._default_config()

        # Decision thresholds
        self.thresholds = {
            "auto_approve": self.config.get("auto_approve_threshold", 0.15),
            "review": self.config.get("review_threshold", 0.45),
            "auto_decline": self.config.get("auto_decline_threshold", 0.80),
        }

        # Weights for combining signals
        self.weights = {
            "ml_ensemble": self.config.get("ml_weight", 0.55),
            "rules": self.config.get("rules_weight", 0.25),
            "threat_intel": self.config.get("threat_intel_weight", 0.20),
        }

    def _default_config(self) -> Dict[str, Any]:
        return {
            "auto_approve_threshold": 0.15,
            "review_threshold": 0.45,
            "auto_decline_threshold": 0.80,
            "ml_weight": 0.55,
            "rules_weight": 0.25,
            "threat_intel_weight": 0.20,
            "enable_threat_intel": True,
            "enable_rules": True,
            "explainability_enabled": True,
        }

    async def score(self, transaction: UPITransaction) -> DecisionContext:
        """
        Score a transaction and return complete decision context.

        Target latency: <50ms p99
        """
        start_time = time.perf_counter()

        # 1. Get features (from FeatureStore - passed in or fetched)
        # In real usage, features would be fetched from FeatureStore
        # Here we compute them inline for demo
        features = self._build_features(transaction)

        # 2. Compute derived features
        FeatureComputer.compute_all(features)

        # 3. ML Ensemble inference
        ml_start = time.perf_counter()
        ml_score, ml_confidence, model_scores = self.ensemble.predict_with_uncertainty(
            transaction, features
        )
        ml_latency = (time.perf_counter() - ml_start) * 1000

        # 4. Evaluate deterministic rules (parallel with threat intel)
        rule_results = []
        if self.config.get("enable_rules", True):
            rule_results = self.rule_engine.evaluate(transaction, features)

        # 5. Query threat intelligence
        threat_intel_results = []
        if self.config.get("enable_threat_intel", True):
            threat_intel_results = await self.threat_intel.lookup_all(transaction)

        # 6. Combine signals
        rule_score = self.rule_engine.compute_rule_score(rule_results)
        threat_score = self._compute_threat_score(threat_intel_results)

        # Weighted ensemble
        final_score = (
            self.weights["ml_ensemble"] * ml_score
            + self.weights["rules"] * rule_score
            + self.weights["threat_intel"] * threat_score
        )

        # 7. Apply decision thresholds
        decision, risk_level = self._make_decision(final_score, rule_results)

        # 8. Generate explanation
        explanation = self._generate_explanation(
            transaction,
            features,
            ml_score,
            ml_confidence,
            model_scores,
            rule_results,
            threat_intel_results,
            rule_score,
            threat_score,
            final_score,
            decision,
            risk_level,
        )

        total_latency = (time.perf_counter() - start_time) * 1000

        return DecisionContext(
            transaction=transaction,
            features=features,
            ml_score=ml_score,
            ml_confidence=ml_confidence,
            rule_results=rule_results,
            threat_intel_results=threat_intel_results,
            risk_level=risk_level,
            final_score=final_score,
            decision=decision,
            explanation=explanation,
            latency_ms=total_latency,
            model_scores=model_scores,
        )

    def _build_features(self, transaction: UPITransaction) -> UPIFeatures:
        """Build feature object from transaction (mock for demo)."""
        from upishield_core.features.schemas import (
            DeviceNetworkFeatures,
            GraphFeatures,
            MerchantFeatures,
            UserBehavioralFeatures,
            VelocityFeatures,
        )

        # In production, this would fetch from FeatureStore
        # Here we create minimal features with defaults
        return UPIFeatures(
            txn_id=transaction.txn_id,
            payer_vpa=transaction.payer_vpa,
            payee_vpa=transaction.payee_vpa,
            amount=float(transaction.amount),
            currency=transaction.currency,
            txn_type=(
                transaction.txn_type.value
                if hasattr(transaction.txn_type, "value")
                else str(transaction.txn_type)
            ),
            timestamp=transaction.timestamp or datetime.utcnow(),
            user_behavioral=UserBehavioralFeatures(user_id=transaction.payer_vpa),
            device_network=DeviceNetworkFeatures(device_id=transaction.device_id or "unknown"),
            merchant=MerchantFeatures(merchant_vpa=transaction.payee_vpa),
            velocity=VelocityFeatures(),
            graph=GraphFeatures(),
        )

    def _compute_threat_score(self, threat_results: List[ThreatIntelResult]) -> float:
        """Compute composite threat intelligence score."""
        if not threat_results:
            return 0.0

        # Weighted by confidence
        total_weighted = sum(r.risk_score * r.confidence for r in threat_results)
        total_confidence = sum(r.confidence for r in threat_results)

        if total_confidence == 0:
            return 0.0

        return total_weighted / total_confidence

    def _make_decision(
        self, final_score: float, rule_results: List[RuleResult]
    ) -> Tuple[FraudDecision, RiskLevel]:
        """Apply thresholds and make final decision."""

        # Check for critical rules that force decline
        critical_rules = [r for r in rule_results if r.severity == "critical"]
        if critical_rules:
            return FraudDecision.DECLINE, RiskLevel.CRITICAL

        # Apply thresholds
        if final_score >= self.thresholds["auto_decline"]:
            return FraudDecision.DECLINE, RiskLevel.HIGH
        elif final_score >= self.thresholds["review"]:
            return FraudDecision.REVIEW, RiskLevel.MEDIUM
        else:
            return FraudDecision.APPROVE, RiskLevel.LOW

    def _generate_explanation(
        self,
        transaction: UPITransaction,
        features: UPIFeatures,
        ml_score: float,
        ml_confidence: float,
        model_scores: Dict[str, float],
        rule_results: List[RuleResult],
        threat_results: List[ThreatIntelResult],
        rule_score: float,
        threat_score: float,
        final_score: float,
        decision: FraudDecision,
        risk_level: RiskLevel,
    ) -> Dict[str, Any]:
        """Generate human-readable explanation for the decision."""

        # Top contributing factors
        factors = []

        # ML factors
        factors.append(
            {
                "source": "ml_ensemble",
                "factor": "ensemble_score",
                "value": ml_score,
                "confidence": ml_confidence,
                "contribution": self.weights["ml_ensemble"] * ml_score,
                "details": {
                    "model_scores": model_scores,
                    "uncertainty": 1.0 - ml_confidence,
                },
            }
        )

        # Rule factors
        if rule_results:
            top_rules = sorted(rule_results, key=lambda r: r.score, reverse=True)[:5]
            for r in top_rules:
                factors.append(
                    {
                        "source": "deterministic_rules",
                        "factor": r.rule_id,
                        "value": r.score,
                        "contribution": self.weights["rules"] * r.score,
                        "details": {
                            "rule_name": r.rule_name,
                            "severity": r.severity,
                            "description": r.description,
                        },
                    }
                )

        # Threat intel factors
        if threat_results:
            for t in threat_results:
                if t.risk_score > 0:
                    factors.append(
                        {
                            "source": "threat_intel",
                            "factor": t.source,
                            "value": t.risk_score,
                            "contribution": self.weights["threat_intel"] * t.risk_score,
                            "details": {
                                "labels": t.labels,
                                "confidence": t.confidence,
                            },
                        }
                    )

        # Sort by contribution
        factors.sort(key=lambda f: f["contribution"], reverse=True)

        # Feature importance (from ML models)
        feature_importance = self._get_feature_importance(features)

        return {
            "decision": decision.value,
            "risk_level": risk_level.value,
            "final_score": round(final_score, 4),
            "score_breakdown": {
                "ml_ensemble": round(self.weights["ml_ensemble"] * ml_score, 4),
                "rules": round(self.weights["rules"] * rule_score, 4),
                "threat_intel": round(self.weights["threat_intel"] * threat_score, 4),
            },
            "model_scores": {k: round(v, 4) for k, v in model_scores.items()},
            "ml_confidence": round(ml_confidence, 4),
            "top_factors": factors[:10],
            "triggered_rules": [
                {"rule_id": r.rule_id, "name": r.rule_name, "severity": r.severity}
                for r in rule_results
            ],
            "threat_intel_summary": [
                {"source": t.source, "risk_score": t.risk_score, "labels": t.labels}
                for t in threat_results
                if t.risk_score > 0
            ],
            "feature_importance": feature_importance,
            "recommendation": self._get_recommendation(decision, risk_level, factors[:3]),
        }

    def _get_feature_importance(self, features: UPIFeatures) -> Dict[str, float]:
        """Get feature importance from ensemble (SHAP-like)."""
        # In production, compute SHAP values
        # Here return placeholder based on feature values
        importance = {}

        if features.velocity:
            importance["velocity_5m"] = min(features.velocity.vpa_velocity_5m / 20.0, 1.0)
            importance["amount_velocity_1h"] = min(
                features.velocity.amount_velocity_1h / 100000.0, 1.0
            )

        if features.user_behavioral:
            importance["amount_zscore"] = min(abs(features.amount_zscore_7d) / 5.0, 1.0)
            importance["user_age"] = 1.0 if features.is_new_user else 0.0

        if features.device_network:
            importance["device_trust"] = 1.0 - features.device_network.device_trust_score
            importance["network_anomaly"] = getattr(features, "network_anomaly_score", 0.0)

        if features.merchant:
            importance["merchant_risk"] = features.merchant.merchant_risk_score
            importance["merchant_verified"] = 0.0 if features.merchant.is_verified_merchant else 0.5

        return {k: round(v, 4) for k, v in importance.items()}

    def _get_recommendation(
        self, decision: FraudDecision, risk_level: RiskLevel, top_factors: List[Dict]
    ) -> str:
        """Generate human-readable recommendation."""
        if decision == FraudDecision.DECLINE:
            if risk_level == RiskLevel.CRITICAL:
                return "Decline transaction immediately. Critical fraud signals detected."
            return "Decline transaction. High fraud probability."

        elif decision == FraudDecision.REVIEW:
            factor_names = [f.get("factor", "") for f in top_factors[:2]]
            return f"Route to manual review. Key factors: {', '.join(factor_names)}"

        else:
            return "Approve transaction. Low fraud risk."

    # ============================================================
    # Batch Scoring
    # ============================================================

    async def score_batch(self, transactions: List[UPITransaction]) -> List[DecisionContext]:
        """Score multiple transactions in parallel."""
        tasks = [self.score(txn) for txn in transactions]
        return await asyncio.gather(*tasks)

    # ============================================================
    # Health & Monitoring
    # ============================================================

    def get_health(self) -> Dict[str, Any]:
        """Health check for monitoring."""
        return {
            "status": "healthy",
            "ensemble_loaded": self.ensemble is not None,
            "models_loaded": len(self.ensemble.sessions) if self.ensemble else 0,
            "rules_count": len(self.rule_engine.rules),
            "thresholds": self.thresholds,
            "weights": self.weights,
        }


class DecisionEngineSync:
    """Synchronous wrapper for DecisionEngine."""

    def __init__(self, *args, **kwargs):
        self._async_engine = DecisionEngine(*args, **kwargs)

    def score(self, transaction: UPITransaction) -> DecisionContext:
        """Synchronous score - runs async in event loop."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._async_engine.score(transaction))

    def score_batch(self, transactions: List[UPITransaction]) -> List[DecisionContext]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._async_engine.score_batch(transactions))
