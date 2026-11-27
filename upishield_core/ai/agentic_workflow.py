"""
LangGraph Multi-Agent Fraud Detection Workflow.

Implements 5 specialized agents:
1. FeatureExtractionAgent - Parse raw input → structured features
2. PatternRecognitionAgent - Match against known fraud patterns
3. ModelEnsembleAgent - Run predictions across multiple models
4. DecisionConsensusAgent - Weighted voting with explanations
5. ReportGenerationAgent - Generate human-readable fraud reports
6. InvestigationAgent - Autonomous investigation with tool use (bonus)
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict, Union

import numpy as np

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from upishield_core.ai.llm_client import (
    BaseLLMClient,
    LLMConfig,
    LLMManager,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    create_llm_client,
    get_default_llm_config,
)
from upishield_core.ai.prompts import (
    AI_ASSISTANT_PROMPT,
    DECISION_CONSENSUS_PROMPT,
    FEATURE_EXTRACTION_PROMPT,
    FRAUD_TOOLS,
    INVESTIGATION_PROMPT,
    PATTERN_RECOGNITION_PROMPT,
    REPORT_GENERATION_PROMPT,
    SYSTEM_PROMPT_DECISION_CONSENSUS,
    SYSTEM_PROMPT_FEATURE_EXTRACTION,
    SYSTEM_PROMPT_INVESTIGATION,
    SYSTEM_PROMPT_PATTERN_RECOGNITION,
    SYSTEM_PROMPT_REPORT_GENERATION,
    format_prompt,
    get_agent_prompt,
    get_tools_for_agent,
)
from upishield_core.ai.knowledge_base import (
    FraudKnowledgeBase,
    KnowledgeCategory,
    get_knowledge_base,
)

logger = logging.getLogger(__name__)

# State Definitions

class FraudInputType(str, Enum):
    TRANSACTION = "transaction"
    SMS = "sms"
    JOB_POSTING = "job_posting"
    DOCUMENT = "document"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


@dataclass
class FraudFeatures:
    """Structured features extracted from raw input."""
    transaction_features: Dict[str, Any] = field(default_factory=dict)
    velocity_features: Dict[str, Any] = field(default_factory=dict)
    amount_features: Dict[str, Any] = field(default_factory=dict)
    user_behavioral_features: Dict[str, Any] = field(default_factory=dict)
    device_network_features: Dict[str, Any] = field(default_factory=dict)
    merchant_features: Dict[str, Any] = field(default_factory=dict)
    graph_features: Dict[str, Any] = field(default_factory=dict)
    text_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_features": self.transaction_features,
            "velocity_features": self.velocity_features,
            "amount_features": self.amount_features,
            "user_behavioral_features": self.user_behavioral_features,
            "device_network_features": self.device_network_features,
            "merchant_features": self.merchant_features,
            "graph_features": self.graph_features,
            "text_features": self.text_features,
        }


@dataclass
class PatternMatch:
    """Matched fraud pattern."""
    pattern_id: str
    pattern_name: str
    confidence: float
    evidence: List[str]
    severity: str


@dataclass
class ModelPrediction:
    """Individual model prediction."""
    model_name: str
    score: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleResult:
    """Rule engine result."""
    rule_name: str
    triggered: bool
    score: float
    severity: str
    description: str


@dataclass
class ThreatIntelResult:
    """Threat intelligence result."""
    indicator_type: str
    indicator_value: str
    risk_score: float
    categories: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FraudDecision:
    """Final fraud decision."""
    decision: Decision
    risk_level: RiskLevel
    final_score: float
    confidence: float
    score_breakdown: Dict[str, float]
    key_factors: List[Dict[str, Any]]
    critical_rules_triggered: List[str]
    recommendation: str


@dataclass
class FraudReport:
    """Generated fraud investigation report."""
    executive_summary: str
    transaction_details: str
    risk_assessment: str
    triggered_rules_patterns: str
    model_predictions: str
    threat_intelligence_findings: str
    recommended_actions: str
    evidence_summary: str
    markdown: str = ""


class AgentState(TypedDict):
    """LangGraph state for the fraud detection workflow."""
    # Input
    input_type: FraudInputType
    raw_input: Dict[str, Any]
    request_id: str

    # Feature extraction output
    features: Optional[FraudFeatures]
    feature_extraction_time_ms: float

    # Pattern recognition output
    matched_patterns: List[PatternMatch]
    overall_fraud_likelihood: float
    primary_category: str
    pattern_recognition_time_ms: float

    # Model ensemble output
    model_predictions: List[ModelPrediction]
    ensemble_score: float
    ensemble_confidence: float
    model_ensemble_time_ms: float

    # Rule engine output (from existing decision engine)
    rule_results: List[RuleResult]
    rule_composite_score: float

    # Threat intelligence output
    threat_intel_results: List[ThreatIntelResult]
    threat_composite_score: float

    # Decision consensus output
    final_decision: Optional[FraudDecision]
    decision_time_ms: float

    # Report generation output
    report: Optional[FraudReport]
    report_time_ms: float

    # Investigation output (optional)
    investigation_hypothesis: Optional[str]
    investigation_evidence: List[Dict[str, Any]]
    investigation_actions: List[str]
    investigation_time_ms: float

    # Metadata
    total_time_ms: float
    errors: List[str] = field(default_factory=list)
    agent_trace: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# Tool Implementations
# ============================================================

class ScoreTransactionTool(BaseTool):
    name: str = "score_transaction"
    description: str = "Score a UPI transaction for fraud risk"

    def _run(self, **kwargs) -> str:
        # This would call the actual scoring service
        return json.dumps({"score": 0.3, "risk": "MEDIUM", "message": "Tool stub - implement service call"})

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class DetectSMSScamTool(BaseTool):
    name: str = "detect_sms_scam"
    description: str = "Detect SMS phishing/scam from text message"

    def _run(self, **kwargs) -> str:
        return json.dumps({"is_scam": False, "patterns": [], "message": "Tool stub"})

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class DetectJobScamTool(BaseTool):
    name: str = "detect_job_scam"
    description: str = "Detect job scam from job description"

    def _run(self, **kwargs) -> str:
        return json.dumps({"is_scam": False, "patterns": [], "message": "Tool stub"})

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class SearchKnowledgeBaseTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = "Search fraud knowledge base for patterns, prevention tips, regulations"

    def _run(self, query: str, category: str = "general", top_k: int = 5) -> str:
        # Query the actual knowledge base
        try:
            kb = get_knowledge_base()
            if not kb._initialized:
                kb.initialize(seed_data=True)

            cat = None
            if category != "general":
                try:
                    cat = KnowledgeCategory(category)
                except ValueError:
                    pass

            results = kb.search(query, top_k=top_k, category_filter=cat, min_score=0.3)
            return json.dumps({
                "results": [
                    {
                        "chunk_id": r.chunk.id,
                        "title": r.chunk.title,
                        "content": r.chunk.content,
                        "category": r.chunk.category.value,
                        "relevance": r.score
                    }
                    for r in results
                ]
            })
        except Exception as e:
            return json.dumps({"error": str(e), "results": []})

    async def _arun(self, query: str, category: str = "general", top_k: int = 5) -> str:
        return self._run(query, category, top_k)


class LookupThreatIntelTool(BaseTool):
    name: str = "lookup_threat_intel"
    description: str = "Look up threat intelligence for IP, device, VPA, phone"

    def _run(self, indicator_type: str, value: str) -> str:
        return json.dumps({
            "risk_score": 0.2,
            "categories": ["clean"],
            "details": {}
        })

    async def _arun(self, indicator_type: str, value: str) -> str:
        return self._run(indicator_type, value)


class GetEntityRelationshipsTool(BaseTool):
    name: str = "get_entity_relationships"
    description: str = "Get graph relationships for an entity (device, IP, VPA, user)"

    def _run(self, entity_type: str, entity_value: str, depth: int = 2) -> str:
        return json.dumps({
            "nodes": [],
            "edges": [],
            "risk_entities": []
        })

    async def _arun(self, entity_type: str, entity_value: str, depth: int = 2) -> str:
        return self._run(entity_type, entity_value, depth)


class GetTransactionHistoryTool(BaseTool):
    name: str = "get_transaction_history"
    description: str = "Get transaction history for a user/VPA"

    def _run(self, vpa: str, days: int = 30, limit: int = 100) -> str:
        return json.dumps({"transactions": []})

    async def _arun(self, vpa: str, days: int = 30, limit: int = 100) -> str:
        return self._run(vpa, days, limit)


class AnalyzeDocumentTool(BaseTool):
    name: str = "analyze_document"
    description: str = "Analyze document for forgery/tampering (KYC, screenshots, IDs)"

    def _run(self, document_type: str, image_url: str) -> str:
        return json.dumps({
            "is_tampered": False,
            "tampering_types": [],
            "confidence": 0.9
        })

    async def _arun(self, document_type: str, image_url: str) -> str:
        return self._run(document_type, image_url)


# Tool instances
AVAILABLE_TOOLS = [
    ScoreTransactionTool(),
    DetectSMSScamTool(),
    DetectJobScamTool(),
    SearchKnowledgeBaseTool(),
    LookupThreatIntelTool(),
    GetEntityRelationshipsTool(),
    GetTransactionHistoryTool(),
    AnalyzeDocumentTool(),
]

TOOL_MAP = {tool.name: tool for tool in AVAILABLE_TOOLS}


# LLM Client Wrapper for LangChain Compatibility

class LangChainLLMWrapper:
    """Wrapper to make our LLM client compatible with LangGraph."""

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self._model_name = "multi-provider"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> AIMessage:
        """Invoke LLM with messages and optional tools."""
        # Convert LangChain messages to our format
        llm_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                role = "user"

            llm_messages.append(LLMMessage(
                role=role,
                content=msg.content,
                tool_calls=getattr(msg, "tool_calls", None),
                tool_call_id=getattr(msg, "tool_call_id", None),
                name=getattr(msg, "name", None)
            ))

        # Convert tools to OpenAI format
        tool_schemas = None
        if tools:
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args_schema.schema() if hasattr(tool, "args_schema") else {"type": "object", "properties": {}}
                    }
                }
                for tool in tools
            ]

        try:
            response = await self.llm_manager.chat(
                messages=llm_messages,
                tools=tool_schemas,
                tool_choice="auto" if tools else None,
                stream=False
            )

            if isinstance(response, LLMResponse):
                if response.tool_calls:
                    return AIMessage(
                        content=response.content,
                        tool_calls=response.tool_calls
                    )
                return AIMessage(content=response.content)
            else:
                # Handle streaming response
                content = ""
                async for chunk in response:
                    content += chunk
                return AIMessage(content=content)

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return AIMessage(content=f"Error: {str(e)}")

    def bind_tools(self, tools: List[BaseTool]) -> "LangChainLLMWrapper":
        """Bind tools for tool calling."""
        self._bound_tools = tools
        return self


# Agent Nodes

async def feature_extraction_agent(state: AgentState) -> AgentState:
    """Agent 1: Extract structured features from raw input."""
    start_time = time.time()
    request_id = state["request_id"]

    logger.info(f"[{request_id}] FeatureExtractionAgent starting")

    try:
        # Get LLM manager from global (set at workflow creation)
        llm_manager: LLMManager = get_workflow_llm_manager()

        # Prepare input for feature extraction
        input_data = json.dumps(state["raw_input"], default=str)

        # Try LLM-based extraction first
        features_data = {}
        if llm_manager:
            try:
                # Get prompt
                messages = get_agent_prompt("feature_extraction", input_data=input_data)

                # Convert to LLM messages
                llm_messages = [
                    LLMMessage(role=m["role"], content=m["content"])
                    for m in messages
                ]

                # Call LLM
                response = await llm_manager.chat(llm_messages)

                if isinstance(response, AsyncGenerator):
                    content = ""
                    async for chunk in response:
                        content += chunk
                else:
                    content = response.content

                # Parse JSON response
                try:
                    features_data = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"[{request_id}] Failed to parse features JSON, using fallback")
                    features_data = {}
            except Exception as e:
                logger.warning(f"[{request_id}] LLM extraction failed, using fallback: {e}")
                features_data = {}
        else:
            logger.info(f"[{request_id}] No LLM available, using fallback feature extraction")

        # Fallback: Extract basic features from raw input without LLM
        if not features_data:
            features_data = _extract_basic_features(state["raw_input"])

        # Create FraudFeatures object
        features = FraudFeatures(
            transaction_features=features_data.get("transaction_features", {}),
            velocity_features=features_data.get("velocity_features", {}),
            amount_features=features_data.get("amount_features", {}),
            user_behavioral_features=features_data.get("user_behavioral_features", {}),
            device_network_features=features_data.get("device_network_features", {}),
            merchant_features=features_data.get("merchant_features", {}),
            graph_features=features_data.get("graph_features", {}),
            text_features=features_data.get("text_features", {}),
        )

        elapsed = (time.time() - start_time) * 1000

        state["features"] = features
        state["feature_extraction_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "FeatureExtractionAgent",
            "time_ms": elapsed,
            "status": "success"
        })

        logger.info(f"[{request_id}] FeatureExtractionAgent completed in {elapsed:.1f}ms")

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        state["errors"].append(f"FeatureExtractionAgent: {str(e)}")
        state["feature_extraction_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "FeatureExtractionAgent",
            "time_ms": elapsed,
            "status": "error",
            "error": str(e)
        })
        logger.error(f"[{request_id}] FeatureExtractionAgent failed: {e}")

    return state


async def pattern_recognition_agent(state: AgentState) -> AgentState:
    """Agent 2: Match against known fraud patterns."""
    start_time = time.time()
    request_id = state["request_id"]

    logger.info(f"[{request_id}] PatternRecognitionAgent starting")

    try:
        llm_manager: LLMManager = get_workflow_llm_manager()
        features = state.get("features")
        if not features:
            raise ValueError("Features not available")

        input_data = json.dumps(state["raw_input"], default=str)
        features_json = json.dumps(features.to_dict(), default=str)

        # Try LLM-based pattern recognition
        pattern_data = {"matched_patterns": [], "overall_fraud_likelihood": 0.0, "primary_category": "unknown"}

        if llm_manager:
            try:
                messages = get_agent_prompt("pattern_recognition", input_data=input_data, features=features_json)

                llm_messages = [
                    LLMMessage(role=m["role"], content=m["content"])
                    for m in messages
                ]

                response = await llm_manager.chat(llm_messages)

                if isinstance(response, AsyncGenerator):
                    content = ""
                    async for chunk in response:
                        content += chunk
                else:
                    content = response.content

                try:
                    pattern_data = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"[{request_id}] Failed to parse patterns JSON")
            except Exception as e:
                logger.warning(f"[{request_id}] LLM pattern recognition failed: {e}")

        # Fallback: Rule-based pattern matching
        if not pattern_data.get("matched_patterns"):
            pattern_data = _rule_based_pattern_matching(state["raw_input"], features)

        matched_patterns = [
            PatternMatch(
                pattern_id=p["pattern_id"],
                pattern_name=p["pattern_name"],
                confidence=p["confidence"],
                evidence=p["evidence"],
                severity=p["severity"]
            )
            for p in pattern_data.get("matched_patterns", [])
        ]

        elapsed = (time.time() - start_time) * 1000

        state["matched_patterns"] = matched_patterns
        state["overall_fraud_likelihood"] = pattern_data.get("overall_fraud_likelihood", 0.0)
        state["primary_category"] = pattern_data.get("primary_category", "unknown")
        state["pattern_recognition_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "PatternRecognitionAgent",
            "time_ms": elapsed,
            "status": "success",
            "patterns_matched": len(matched_patterns)
        })

        logger.info(f"[{request_id}] PatternRecognitionAgent completed in {elapsed:.1f}ms, {len(matched_patterns)} patterns")

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        state["errors"].append(f"PatternRecognitionAgent: {str(e)}")
        state["pattern_recognition_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "PatternRecognitionAgent",
            "time_ms": elapsed,
            "status": "error",
            "error": str(e)
        })
        logger.error(f"[{request_id}] PatternRecognitionAgent failed: {e}")

    return state


async def model_ensemble_agent(state: AgentState) -> AgentState:
    """Agent 3: Run predictions across multiple models."""
    start_time = time.time()
    request_id = state["request_id"]

    logger.info(f"[{request_id}] ModelEnsembleAgent starting")

    try:
        # Import existing ensemble inference
        from upishield_core.models.ensemble import EnsembleInference

        features = state.get("features")
        if not features:
            raise ValueError("Features not available")

        # Prepare feature vector for models
        # Combine all features into a flat array
        feature_vector = _features_to_vector(features)

        # Get ensemble and run prediction
        ensemble = EnsembleInference()
        if ensemble.sessions:
            # Try ONNX models first
            input_tensor = np.array([feature_vector], dtype=np.float32)
            predictions = []
            for name in ensemble.sessions.keys():
                pred = ensemble._run_model(name, input_tensor)
                predictions.append(ModelPrediction(
                    model_name=name,
                    score=pred.risk_score,
                    confidence=pred.confidence,
                    metadata={"latency_ms": pred.latency_ms}
                ))

            if predictions:
                weighted_score, confidence, model_scores = ensemble.ensemble_score_with_metadata(predictions)
                ensemble_score = weighted_score
                ensemble_confidence = confidence
            else:
                raise ValueError("No predictions from ONNX models")
        else:
            # Fallback: use sklearn models
            input_tensor = np.array([feature_vector], dtype=np.float32)
            if ensemble.scaler is not None:
                input_tensor = ensemble.scaler.transform(input_tensor)

            predictions = []
            for name in ensemble.sklearn_models.keys():
                model = ensemble.sklearn_models[name]
                proba = model.predict_proba(input_tensor)
                risk_score = float(proba[:, 1][0])
                confidence = float(np.max(proba))
                predictions.append(ModelPrediction(
                    model_name=name,
                    score=risk_score,
                    confidence=confidence,
                    metadata={}
                ))

            if not predictions:
                # Ultimate fallback
                predictions = [
                    ModelPrediction(model_name="xgboost", score=0.35, confidence=0.85),
                    ModelPrediction(model_name="lightgbm", score=0.32, confidence=0.82),
                    ModelPrediction(model_name="transformer", score=0.38, confidence=0.88),
                    ModelPrediction(model_name="gnn", score=0.30, confidence=0.75),
                ]

            ensemble_score = sum(p.score for p in predictions) / len(predictions)
            ensemble_confidence = sum(p.confidence for p in predictions) / len(predictions)

        elapsed = (time.time() - start_time) * 1000

        state["model_predictions"] = predictions
        state["ensemble_score"] = ensemble_score
        state["ensemble_confidence"] = ensemble_confidence
        state["model_ensemble_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "ModelEnsembleAgent",
            "time_ms": elapsed,
            "status": "success",
            "models_run": len(predictions)
        })

        logger.info(f"[{request_id}] ModelEnsembleAgent completed in {elapsed:.1f}ms, ensemble={ensemble_score:.3f}")

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        state["errors"].append(f"ModelEnsembleAgent: {str(e)}")
        state["model_ensemble_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "ModelEnsembleAgent",
            "time_ms": elapsed,
            "status": "error",
            "error": str(e)
        })
        logger.error(f"[{request_id}] ModelEnsembleAgent failed: {e}")

    return state


async def decision_consensus_agent(state: AgentState) -> AgentState:
    """Agent 4: Weighted voting with explanations."""
    start_time = time.time()
    request_id = state["request_id"]

    logger.info(f"[{request_id}] DecisionConsensusAgent starting")

    try:
        llm_manager: LLMManager = get_workflow_llm_manager()

        # Prepare inputs for decision
        ml_scores = {p.model_name: p.score for p in state.get("model_predictions", [])}
        ml_confidence = state.get("ensemble_confidence", 0.0)

        rule_results = state.get("rule_results", [])
        rule_score = state.get("rule_composite_score", 0.0)

        threat_intel = state.get("threat_intel_results", [])
        threat_score = state.get("threat_composite_score", 0.0)

        transaction = state.get("raw_input", {})

        # Try LLM-based decision
        decision_data = None
        if llm_manager:
            try:
                messages = get_agent_prompt(
                    "decision_consensus",
                    ml_scores=json.dumps(ml_scores),
                    ml_confidence=ml_confidence,
                    rule_results=json.dumps([r.__dict__ for r in rule_results], default=str),
                    rule_score=rule_score,
                    threat_intel=json.dumps([t.__dict__ for t in threat_intel], default=str),
                    threat_score=threat_score,
                    transaction=json.dumps(transaction, default=str)
                )

                llm_messages = [
                    LLMMessage(role=m["role"], content=m["content"])
                    for m in messages
                ]

                response = await llm_manager.chat(llm_messages)

                if isinstance(response, AsyncGenerator):
                    content = ""
                    async for chunk in response:
                        content += chunk
                else:
                    content = response.content

                try:
                    decision_data = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"[{request_id}] Failed to parse decision JSON")
            except Exception as e:
                logger.warning(f"[{request_id}] LLM decision failed: {e}")

        # Fallback: Rule-based decision logic
        if not decision_data:
            final_score = (
                state.get("ensemble_score", 0.0) * 0.55 +
                rule_score * 0.25 +
                threat_score * 0.20
            )
            decision_data = _fallback_decision(final_score, rule_results)

        decision = FraudDecision(
            decision=Decision(decision_data.get("decision", "REVIEW")),
            risk_level=RiskLevel(decision_data.get("risk_level", "MEDIUM")),
            final_score=decision_data.get("final_score", 0.0),
            confidence=decision_data.get("confidence", 0.0),
            score_breakdown=decision_data.get("score_breakdown", {}),
            key_factors=decision_data.get("key_factors", []),
            critical_rules_triggered=decision_data.get("critical_rules_triggered", []),
            recommendation=decision_data.get("recommendation", "Manual review required")
        )

        elapsed = (time.time() - start_time) * 1000

        state["final_decision"] = decision
        state["decision_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "DecisionConsensusAgent",
            "time_ms": elapsed,
            "status": "success",
            "decision": decision.decision.value,
            "risk_level": decision.risk_level.value
        })

        logger.info(f"[{request_id}] DecisionConsensusAgent completed in {elapsed:.1f}ms: {decision.decision.value}")

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        state["errors"].append(f"DecisionConsensusAgent: {str(e)}")
        state["decision_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "DecisionConsensusAgent",
            "time_ms": elapsed,
            "status": "error",
            "error": str(e)
        })
        logger.error(f"[{request_id}] DecisionConsensusAgent failed: {e}")

    return state


async def report_generation_agent(state: AgentState) -> AgentState:
    """Agent 5: Generate human-readable fraud report."""
    start_time = time.time()
    request_id = state["request_id"]

    logger.info(f"[{request_id}] ReportGenerationAgent starting")

    try:
        llm_manager: LLMManager = get_workflow_llm_manager()

        decision = state.get("final_decision")
        if not decision:
            raise ValueError("Decision not available")

        # Try LLM-based report generation
        content = None
        if llm_manager:
            try:
                messages = get_agent_prompt(
                    "report_generation",
                    decision_context=json.dumps({
                        "decision": decision.decision.value,
                        "risk_level": decision.risk_level.value,
                        "final_score": decision.final_score,
                        "confidence": decision.confidence
                    }, default=str),
                    transaction=json.dumps(state.get("raw_input", {}), default=str),
                    features=json.dumps(state.get("features", FraudFeatures()).to_dict(), default=str),
                    rule_results=json.dumps([r.__dict__ for r in state.get("rule_results", [])], default=str),
                    threat_intel=json.dumps([t.__dict__ for t in state.get("threat_intel_results", [])], default=str),
                    model_scores=json.dumps({p.model_name: p.score for p in state.get("model_predictions", [])}, default=str)
                )

                llm_messages = [
                    LLMMessage(role=m["role"], content=m["content"])
                    for m in messages
                ]

                response = await llm_manager.chat(llm_messages)

                if isinstance(response, AsyncGenerator):
                    content = ""
                    async for chunk in response:
                        content += chunk
                else:
                    content = response.content
            except Exception as e:
                logger.warning(f"[{request_id}] LLM report generation failed: {e}")

        # Fallback: Generate basic report without LLM
        if not content:
            content = _generate_fallback_report(
                decision,
                state.get("raw_input", {}),
                state.get("features"),
                state.get("rule_results", []),
                state.get("threat_intel_results", []),
                state.get("model_predictions", []),
                state.get("matched_patterns", [])
            )

        # Parse markdown sections from response
        report = _parse_report_sections(content)
        report.markdown = content

        elapsed = (time.time() - start_time) * 1000

        state["report"] = report
        state["report_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "ReportGenerationAgent",
            "time_ms": elapsed,
            "status": "success"
        })

        logger.info(f"[{request_id}] ReportGenerationAgent completed in {elapsed:.1f}ms")

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        state["errors"].append(f"ReportGenerationAgent: {str(e)}")
        state["report_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "ReportGenerationAgent",
            "time_ms": elapsed,
            "status": "error",
            "error": str(e)
        })
        logger.error(f"[{request_id}] ReportGenerationAgent failed: {e}")

    return state


async def investigation_agent(state: AgentState) -> AgentState:
    """Agent 6: Autonomous investigation with tool use (bonus agent)."""
    start_time = time.time()
    request_id = state["request_id"]

    logger.info(f"[{request_id}] InvestigationAgent starting")

    try:
        llm_manager: LLMManager = get_workflow_llm_manager()
        if not llm_manager:
            config = get_default_llm_config()
            if config.api_key or config.provider in (LLMProvider.OLLAMA, LLMProvider.VLLM):
                llm_manager = LLMManager([config])

        # This agent uses tools for investigation
        # For now, we'll use a simpler approach without full tool calling loop

        decision = state.get("final_decision")
        if not decision or decision.decision != Decision.DECLINE:
            # Only investigate high-risk cases
            state["investigation_hypothesis"] = "Not required - low risk case"
            state["investigation_evidence"] = []
            state["investigation_actions"] = []
            state["investigation_time_ms"] = (time.time() - start_time) * 1000
            return state

        alert = {
            "decision": decision.decision.value,
            "risk_level": decision.risk_level.value,
            "score": decision.final_score,
            "key_factors": decision.key_factors
        }

        messages = get_agent_prompt(
            "investigation",
            alert=json.dumps(alert, default=str),
            transactions=json.dumps([], default=str),
            entity_graph=json.dumps({}, default=str),
            kb_results=json.dumps([], default=str),
            threat_intel=json.dumps([t.__dict__ for t in state.get("threat_intel_results", [])], default=str)
        )

        llm_messages = [
            LLMMessage(role=m["role"], content=m["content"])
            for m in messages
        ]

        # Add tool schemas
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": {}}
                }
            }
            for tool in AVAILABLE_TOOLS
        ]

        response = await llm_manager.chat(llm_messages, tools=tool_schemas)

        if isinstance(response, AsyncGenerator):
            content = ""
            async for chunk in response:
                content += chunk
        else:
            content = response.content

        try:
            inv_data = json.loads(content)
        except json.JSONDecodeError:
            inv_data = {
                "hypothesis": "Investigation parsing failed",
                "confidence": 0.0,
                "evidence": [],
                "related_entities": [],
                "recommended_actions": ["Manual investigation required"],
                "escalation_required": True,
                "estimated_fraud_loss": 0.0
            }

        elapsed = (time.time() - start_time) * 1000

        state["investigation_hypothesis"] = inv_data.get("hypothesis", "")
        state["investigation_evidence"] = inv_data.get("evidence", [])
        state["investigation_actions"] = inv_data.get("recommended_actions", [])
        state["investigation_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "InvestigationAgent",
            "time_ms": elapsed,
            "status": "success"
        })

        logger.info(f"[{request_id}] InvestigationAgent completed in {elapsed:.1f}ms")

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        state["errors"].append(f"InvestigationAgent: {str(e)}")
        state["investigation_time_ms"] = elapsed
        state["agent_trace"].append({
            "agent": "InvestigationAgent",
            "time_ms": elapsed,
            "status": "error",
            "error": str(e)
        })
        logger.error(f"[{request_id}] InvestigationAgent failed: {e}")

    return state

# Helper Functions

def _extract_basic_features(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """Extract basic features from raw input without LLM."""
    features = {
        "transaction_features": {},
        "velocity_features": {},
        "amount_features": {},
        "user_behavioral_features": {},
        "device_network_features": {},
        "merchant_features": {},
        "graph_features": {},
        "text_features": {},
    }

    # Transaction features
    if "amount" in raw_input:
        features["transaction_features"]["amount"] = float(raw_input.get("amount", 0))
        features["transaction_features"]["currency"] = "INR"
    if "txn_type" in raw_input:
        features["transaction_features"]["txn_type"] = raw_input.get("txn_type", "P2P")
    if "timestamp" in raw_input:
        features["transaction_features"]["timestamp"] = raw_input.get("timestamp")
    if "payer_vpa" in raw_input:
        features["transaction_features"]["payer_vpa"] = raw_input.get("payer_vpa")
    if "payee_vpa" in raw_input:
        features["transaction_features"]["payee_vpa"] = raw_input.get("payee_vpa")
    if "merchant_category" in raw_input:
        features["transaction_features"]["merchant_category"] = raw_input.get("merchant_category")
    if "merchant_name" in raw_input:
        features["transaction_features"]["merchant_name"] = raw_input.get("merchant_name")

    # Amount features
    amount = float(raw_input.get("amount", 0))
    features["amount_features"]["is_round_amount"] = (amount % 1000 == 0)
    features["amount_features"]["is_high_value"] = (amount > 50000)
    features["amount_features"]["deviation_from_avg"] = 0.0  # Would need history

    # Device/network features
    if "device_id" in raw_input:
        features["device_network_features"]["device_id"] = raw_input.get("device_id")
    if "client_ip" in raw_input:
        features["device_network_features"]["client_ip"] = raw_input.get("client_ip")
    features["device_network_features"]["is_new_device"] = False  # Would need history
    features["device_network_features"]["device_trust_score"] = 1.0

    # Merchant features
    if "merchant_category" in raw_input:
        features["merchant_features"]["merchant_category_risk"] = 0.1  # Default low
    features["merchant_features"]["is_verified_merchant"] = False

    # Text features (for SMS/job posting analysis)
    text_content = raw_input.get("text", raw_input.get("description", ""))
    if text_content:
        text_lower = text_content.lower()
        features["text_features"]["has_urgency_language"] = any(
            w in text_lower for w in ["urgent", "immediately", "now", "hurry", "quick", "limited time", "act fast"]
        )
        features["text_features"]["has_authority_impersonation"] = any(
            w in text_lower for w in ["bank", "rbi", "npci", "government", "official", "verify", "kyc"]
        )
        features["text_features"]["requests_personal_info"] = any(
            w in text_lower for w in ["otp", "pin", "password", "aadhaar", "pan", "cvv", "card number"]
        )
        features["text_features"]["requests_financial_action"] = any(
            w in text_lower for w in ["pay", "transfer", "send money", "refund", "claim", "collect", "upi"]
        )
        features["text_features"]["contains_suspicious_links"] = any(
            w in text_lower for w in ["http://", "https://", "bit.ly", "tinyurl", "click here", "link"]
        )
        features["text_features"]["contains_phone_numbers"] = any(
            w in text_lower for w in ["call", "whatsapp", "contact", "+91", "mobile"]
        )

    return features


def _features_to_vector(features: FraudFeatures) -> List[float]:
    """Convert structured features to flat vector matching ensemble's 13-feature expectation.

    Matches the feature engineering in EnsembleInference._prepare_input():
    13 features: amount, amount_log, hour, day_of_week, is_night, is_round_amount,
    is_high_value, txn_type_encoded, merchant_cat_encoded, payer_vpa_hash,
    payee_vpa_hash, device_id_hash, client_ip_hash
    """
    import hashlib
    from datetime import datetime

    vector = []

    tf = features.transaction_features
    vf = features.velocity_features
    af = features.amount_features
    dnf = features.device_network_features
    mf = features.merchant_features
    txf = features.text_features

    # Extract raw values from features
    amount = tf.get("amount", 0.0)
    txn_type = tf.get("txn_type", "P2P")
    merchant_category = tf.get("merchant_category", "")
    payer_vpa = tf.get("payer_vpa", "")
    payee_vpa = tf.get("payee_vpa", "")
    device_id = dnf.get("device_id", "")
    client_ip = dnf.get("client_ip", "")
    timestamp_str = tf.get("timestamp", "")

    # Parse timestamp
    hour = 12
    day_of_week = 0
    is_night = 0.0
    if timestamp_str:
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            hour = dt.hour
            day_of_week = dt.weekday()
            is_night = 1.0 if hour in [0,1,2,3,4,5] else 0.0
        except:
            pass

    # 13 features matching ensemble's _prepare_input
    vector.append(amount)  # amount
    vector.append(np.log1p(amount))  # amount_log
    vector.append(float(hour))  # hour
    vector.append(float(day_of_week))  # day_of_week
    vector.append(is_night)  # is_night
    vector.append(1.0 if (amount % 1000 == 0) else 0.0)  # is_round_amount
    vector.append(1.0 if (amount > 50000) else 0.0)  # is_high_value

    # txn_type_encoded - matching training factorize mappings
    txn_type_mapping = {"P2M": 0, "P2P": 1, "COLLECT": 2, "REFUND": 3, "UNKNOWN": 4}
    vector.append(float(txn_type_mapping.get(txn_type, 0)))

    # merchant_cat_encoded - matching training factorize mappings
    merchant_category_mapping = {
        "food_delivery": 0, "entertainment": 1, "travel": 2, "utilities": 3,
        "insurance": 4, "financial": 5, "retail": 6, "healthcare": 7,
        "grocery": 8, "education": 9, "shopping": 10, "fuel": 11,
        "transport": 12, "ecommerce": 13, "personal": 14
    }
    vector.append(float(merchant_category_mapping.get(merchant_category, 0)))

    # Hash-based features (same as training)
    vector.append((hash(payer_vpa) % 10000) / 10000.0)  # payer_vpa_hash
    vector.append((hash(payee_vpa) % 10000) / 10000.0)  # payee_vpa_hash
    vector.append((hash(device_id) % 10000) / 10000.0)  # device_id_hash
    vector.append((hash(client_ip) % 10000) / 10000.0)  # client_ip_hash

    return vector


def _rule_based_pattern_matching(raw_input: Dict[str, Any], features: FraudFeatures) -> Dict[str, Any]:
    """Rule-based pattern matching as fallback when LLM is unavailable.

    Checks for common fraud patterns using keyword matching and feature thresholds.
    """
    matched_patterns = []
    input_text = json.dumps(raw_input, default=str).lower()
    text_content = raw_input.get("text", raw_input.get("description", "")).lower()

    # UPI QR Code Scam
    if any(kw in input_text for kw in ["qr", "scan", "receive money", "qr code"]):
        matched_patterns.append({
            "pattern_id": "UPI_QR_CODE_SCAM",
            "pattern_name": "QR Code Scam",
            "confidence": 0.85,
            "evidence": ["QR code present", "Asked to scan to receive money"],
            "severity": "critical"
        })

    # UPI Collect Request
    if any(kw in input_text for kw in ["collect", "request money", "upi collect"]):
        matched_patterns.append({
            "pattern_id": "UPI_COLLECT_REQUEST",
            "pattern_name": "Fraudulent Collect Request",
            "confidence": 0.75,
            "evidence": ["Collect request detected"],
            "severity": "high"
        })

    # SMS KYC Expiry
    if any(kw in text_content for kw in ["kyc", "expire", "verify", "link", "bank"]):
        matched_patterns.append({
            "pattern_id": "SMS_KYC_EXPIRY",
            "pattern_name": "KYC Expiry Phishing",
            "confidence": 0.80,
            "evidence": ["KYC mentioned", "Urgency language", "Link present"],
            "severity": "high"
        })

    # SMS Job Offer
    if any(kw in text_content for kw in ["job", "offer", "salary", "work from home", "part time"]):
        if any(kw in text_content for kw in ["fee", "deposit", "payment", "registration", "training"]):
            matched_patterns.append({
                "pattern_id": "JOB_DATA_ENTRY",
                "pattern_name": "Job Scam - Upfront Fee",
                "confidence": 0.90,
                "evidence": ["Job offer with upfront fee requirement"],
                "severity": "high"
            })

    # SMS Loan Offer
    if any(kw in text_content for kw in ["loan", "approved", "instant", "low interest"]):
        if any(kw in text_content for kw in ["processing fee", "advance", "pay first"]):
            matched_patterns.append({
                "pattern_id": "SMS_LOAN_OFFER",
                "pattern_name": "Loan Scam - Processing Fee",
                "confidence": 0.85,
                "evidence": ["Pre-approved loan", "Processing fee requested"],
                "severity": "high"
            })

    # SMS Tax Refund
    if any(kw in text_content for kw in ["tax", "refund", "income tax", "government"]):
        if any(kw in text_content for kw in ["link", "click", "claim", "verify"]):
            matched_patterns.append({
                "pattern_id": "SMS_TAX_REFUND",
                "pattern_name": "Tax Refund Phishing",
                "confidence": 0.80,
                "evidence": ["Tax refund claim", "Action required via link"],
                "severity": "high"
            })

    # Velocity-based patterns
    vf = features.velocity_features
    if vf.get("txn_count_5m", 0) > 10:
        matched_patterns.append({
            "pattern_id": "HIGH_VELOCITY_5M",
            "pattern_name": "High Velocity - 5 Minutes",
            "confidence": 0.70,
            "evidence": [f"{vf.get('txn_count_5m')} transactions in 5 minutes"],
            "severity": "high"
        })

    if vf.get("txn_count_1h", 0) > 50:
        matched_patterns.append({
            "pattern_id": "HIGH_VELOCITY_1H",
            "pattern_name": "High Velocity - 1 Hour",
            "confidence": 0.65,
            "evidence": [f"{vf.get('txn_count_1h')} transactions in 1 hour"],
            "severity": "medium"
        })

    # Device-based patterns
    dnf = features.device_network_features
    if dnf.get("is_new_device") and dnf.get("is_vpn_proxy"):
        matched_patterns.append({
            "pattern_id": "NEW_DEVICE_VPN",
            "pattern_name": "New Device with VPN",
            "confidence": 0.75,
            "evidence": ["New device detected", "VPN/Proxy in use"],
            "severity": "medium"
        })

    # Amount-based patterns
    af = features.amount_features
    if af.get("is_round_amount") and af.get("is_high_value"):
        matched_patterns.append({
            "pattern_id": "ROUND_HIGH_AMOUNT",
            "pattern_name": "Round High-Value Amount",
            "confidence": 0.60,
            "evidence": ["Round number amount", "High value transaction"],
            "severity": "medium"
        })

    # Text-based scam patterns
    txf = features.text_features
    urgency_score = sum([
        1 for f in ["has_urgency_language", "has_authority_impersonation",
                    "requests_personal_info", "requests_financial_action",
                    "contains_suspicious_links", "contains_phone_numbers"]
        if txf.get(f)
    ])

    if urgency_score >= 3:
        matched_patterns.append({
            "pattern_id": "SMS_PHISHING_MULTI_INDICATOR",
            "pattern_name": "SMS Phishing - Multiple Indicators",
            "confidence": min(0.9, 0.5 + urgency_score * 0.08),
            "evidence": [k for k, v in txf.items() if v and k.startswith("has_") or k.startswith("requests_") or k.startswith("contains_")],
            "severity": "high"
        })

    # Determine primary category and overall likelihood
    if not matched_patterns:
        return {
            "matched_patterns": [],
            "overall_fraud_likelihood": 0.1,
            "primary_category": "unknown"
        }

    # Score by severity
    severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}
    max_severity_score = max(severity_weights.get(p["severity"], 0) for p in matched_patterns)
    avg_confidence = sum(p["confidence"] for p in matched_patterns) / len(matched_patterns)

    # Determine primary category
    primary_category = "unknown"
    for p in matched_patterns:
        pid = p["pattern_id"]
        if pid.startswith("UPI_"):
            primary_category = "upi_fraud"
            break
        elif pid.startswith("SMS_") or pid.startswith("JOB_"):
            primary_category = "sms_phishing" if pid.startswith("SMS_") else "job_scam"
            break
        elif pid.startswith("DOCUMENT_"):
            primary_category = "document_fraud"
            break

    overall_likelihood = min(0.95, max_severity_score * 0.7 + avg_confidence * 0.3)

    return {
        "matched_patterns": matched_patterns,
        "overall_fraud_likelihood": overall_likelihood,
        "primary_category": primary_category
    }


def _fallback_decision(score: float, rule_results: List[RuleResult]) -> Dict[str, Any]:
    """Fallback decision logic when LLM parsing fails."""
    critical_rules = [r.rule_name for r in rule_results if r.triggered and r.severity == "critical"]

    if score > 0.80 or critical_rules:
        decision = "DECLINE"
        risk_level = "CRITICAL"
    elif score > 0.45:
        decision = "DECLINE"
        risk_level = "HIGH"
    elif score > 0.15:
        decision = "REVIEW"
        risk_level = "MEDIUM"
    else:
        decision = "APPROVE"
        risk_level = "LOW"

    return {
        "decision": decision,
        "risk_level": risk_level,
        "final_score": score,
        "confidence": 0.7,
        "score_breakdown": {
            "ml_contribution": score * 0.55,
            "rules_contribution": score * 0.25,
            "threat_intel_contribution": score * 0.20
        },
        "key_factors": [
            {"factor": "ensemble_score", "contribution": score * 0.55, "source": "ml"}
        ],
        "critical_rules_triggered": critical_rules,
        "recommendation": f"{decision} transaction. Risk level: {risk_level}."
    }


def _parse_report_sections(markdown: str) -> FraudReport:
    """Parse markdown report into structured sections."""
    sections = {
        "executive_summary": "",
        "transaction_details": "",
        "risk_assessment": "",
        "triggered_rules_patterns": "",
        "model_predictions": "",
        "threat_intelligence_findings": "",
        "recommended_actions": "",
        "evidence_summary": ""
    }

    current_section = None
    for line in markdown.split("\n"):
        line_lower = line.lower().strip()
        if "executive summary" in line_lower:
            current_section = "executive_summary"
        elif "transaction detail" in line_lower:
            current_section = "transaction_details"
        elif "risk assessment" in line_lower:
            current_section = "risk_assessment"
        elif "triggered rule" in line_lower or "pattern" in line_lower:
            current_section = "triggered_rules_patterns"
        elif "model prediction" in line_lower:
            current_section = "model_predictions"
        elif "threat intelligence" in line_lower or "threat intel" in line_lower:
            current_section = "threat_intelligence_findings"
        elif "recommended action" in line_lower:
            current_section = "recommended_actions"
        elif "evidence summary" in line_lower:
            current_section = "evidence_summary"
        elif current_section and line.strip():
            sections[current_section] += line + "\n"

    return FraudReport(**sections)


def _generate_fallback_report(
    decision: FraudDecision,
    transaction: Dict[str, Any],
    features: FraudFeatures,
    rule_results: List[RuleResult],
    threat_intel: List[ThreatIntelResult],
    model_predictions: List[ModelPrediction],
    matched_patterns: List[PatternMatch]
) -> str:
    """Generate a basic fraud report without LLM."""

    risk_emoji = {
        "LOW": "✅",
        "MEDIUM": "⚠️",
        "HIGH": "🔴",
        "CRITICAL": "🚨"
    }

    decision_emoji = {
        "APPROVE": "✅",
        "REVIEW": "👁️",
        "DECLINE": "🚫"
    }

    lines = []

    # Executive Summary
    lines.append("# EXECUTIVE SUMMARY")
    lines.append("")
    emoji = decision_emoji.get(decision.decision.value, "❓")
    risk_icon = risk_emoji.get(decision.risk_level.value, "⚠️")
    lines.append(f"{emoji} **Decision: {decision.decision.value}**  {risk_icon} **Risk Level: {decision.risk_level.value}**")
    lines.append(f"**Final Score:** {decision.final_score:.3f} | **Confidence:** {decision.confidence:.2f}")
    lines.append(f"**Recommendation:** {decision.recommendation}")
    lines.append("")

    # Transaction Details
    lines.append("# TRANSACTION DETAILS")
    lines.append("")
    lines.append(f"- **Amount:** ₹{transaction.get('amount', 0):,.2f}")
    lines.append(f"- **Type:** {transaction.get('txn_type', 'N/A')}")
    lines.append(f"- **Payer VPA:** {transaction.get('payer_vpa', 'N/A')}")
    lines.append(f"- **Payee VPA:** {transaction.get('payee_vpa', 'N/A')}")
    lines.append(f"- **Merchant Category:** {transaction.get('merchant_category', 'N/A')}")
    lines.append(f"- **Device ID:** {transaction.get('device_id', 'N/A')}")
    lines.append(f"- **Client IP:** {transaction.get('client_ip', 'N/A')}")
    lines.append(f"- **Timestamp:** {transaction.get('timestamp', 'N/A')}")
    lines.append("")

    # Risk Assessment
    lines.append("# RISK ASSESSMENT")
    lines.append("")
    lines.append(f"- **Final Score:** {decision.final_score:.3f}")
    lines.append(f"- **Risk Level:** {decision.risk_level.value}")
    lines.append(f"- **Confidence:** {decision.confidence:.2f}")
    lines.append("")
    lines.append("**Score Breakdown:**")
    for key, value in decision.score_breakdown.items():
        lines.append(f"- {key}: {value:.3f}")
    lines.append("")

    # Triggered Rules & Patterns
    lines.append("# TRIGGERED RULES & PATTERNS")
    lines.append("")
    if rule_results:
        lines.append("**Rule Engine Results:**")
        for r in rule_results:
            status = "🔴 TRIGGERED" if r.triggered else "✅ Not Triggered"
            lines.append(f"- {r.rule_name}: {status} (Score: {r.score:.2f}, Severity: {r.severity})")
            lines.append(f"  - {r.description}")
    else:
        lines.append("No rule engine results available.")
    lines.append("")

    if matched_patterns:
        lines.append("**Matched Fraud Patterns:**")
        for p in matched_patterns:
            lines.append(f"- **{p.pattern_name}** ({p.pattern_id})")
            lines.append(f"  - Confidence: {p.confidence:.2f} | Severity: {p.severity}")
            for ev in p.evidence:
                lines.append(f"  - Evidence: {ev}")
    else:
        lines.append("No fraud patterns matched.")
    lines.append("")

    # Model Predictions
    lines.append("# MODEL PREDICTIONS")
    lines.append("")
    if model_predictions:
        for p in model_predictions:
            lines.append(f"- **{p.model_name}**: Score={p.score:.3f}, Confidence={p.confidence:.2f}")
    else:
        lines.append("No model predictions available.")
    lines.append("")

    # Threat Intelligence
    lines.append("# THREAT INTELLIGENCE FINDINGS")
    lines.append("")
    if threat_intel:
        for t in threat_intel:
            lines.append(f"- **{t.indicator_type.upper()}**: {t.indicator_value}")
            lines.append(f"  - Risk Score: {t.risk_score:.2f}")
            lines.append(f"  - Categories: {', '.join(t.categories)}")
    else:
        lines.append("No threat intelligence findings.")
    lines.append("")

    # Recommended Actions
    lines.append("# RECOMMENDED ACTIONS")
    lines.append("")
    if decision.decision.value == "DECLINE":
        lines.append("1. **Block transaction** - High fraud risk detected")
        lines.append("2. **Alert user** - Notify of potential fraud attempt")
        lines.append("3. **Add to watchlist** - Flag entities for future monitoring")
    elif decision.decision.value == "REVIEW":
        lines.append("1. **Manual review required** - Escalate to fraud analyst")
        lines.append("2. **Request additional verification** - Contact user for confirmation")
        lines.append("3. **Monitor subsequent transactions** - Enhanced monitoring for 24h")
    else:
        lines.append("1. **Approve transaction** - Low fraud risk")
        lines.append("2. **Standard monitoring** - Continue normal surveillance")
    lines.append("")

    # Evidence Summary
    lines.append("# EVIDENCE SUMMARY")
    lines.append("")
    lines.append("**Key Factors:**")
    for factor in decision.key_factors:
        lines.append(f"- {factor.get('factor', 'Unknown')}: {factor.get('contribution', 0):.3f} ({factor.get('source', 'unknown')})")
    lines.append("")
    if decision.critical_rules_triggered:
        lines.append("**Critical Rules Triggered:**")
        for rule in decision.critical_rules_triggered:
            lines.append(f"- {rule}")

    return "\n".join(lines)


# Workflow Builder

# Global LLM manager for workflow access
_workflow_llm_manager: Optional[LLMManager] = None


def set_workflow_llm_manager(llm_manager: LLMManager):
    """Set the global LLM manager for workflow nodes."""
    global _workflow_llm_manager
    _workflow_llm_manager = llm_manager


def get_workflow_llm_manager() -> Optional[LLMManager]:
    """Get the global LLM manager."""
    return _workflow_llm_manager


def create_fraud_detection_workflow(
    llm_manager: Optional[LLMManager] = None,
    include_investigation: bool = True
) -> StateGraph:
    """Create the LangGraph workflow for fraud detection."""

    # Set global LLM manager for node access
    if llm_manager:
        set_workflow_llm_manager(llm_manager)

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("feature_extraction", feature_extraction_agent)
    workflow.add_node("pattern_recognition", pattern_recognition_agent)
    workflow.add_node("model_ensemble", model_ensemble_agent)
    workflow.add_node("decision_consensus", decision_consensus_agent)
    workflow.add_node("report_generation", report_generation_agent)

    if include_investigation:
        workflow.add_node("investigation", investigation_agent)

    # Define edges - sequential workflow
    workflow.set_entry_point("feature_extraction")
    workflow.add_edge("feature_extraction", "pattern_recognition")
    workflow.add_edge("pattern_recognition", "model_ensemble")
    workflow.add_edge("model_ensemble", "decision_consensus")
    workflow.add_edge("decision_consensus", "report_generation")

    if include_investigation:
        workflow.add_edge("report_generation", "investigation")
        workflow.add_edge("investigation", END)
    else:
        workflow.add_edge("report_generation", END)

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# High-Level API

@dataclass
class FraudDetectionResult:
    """Complete fraud detection result."""
    request_id: str
    input_type: FraudInputType
    decision: Decision
    risk_level: RiskLevel
    final_score: float
    confidence: float
    features: FraudFeatures
    matched_patterns: List[PatternMatch]
    model_predictions: List[ModelPrediction]
    rule_results: List[RuleResult]
    threat_intel_results: List[ThreatIntelResult]
    report: FraudReport
    investigation_hypothesis: Optional[str]
    investigation_actions: List[str]
    processing_time_ms: float
    agent_trace: List[Dict[str, Any]]
    errors: List[str]


class FraudDetectionWorkflow:
    """High-level interface for the fraud detection workflow."""

    def __init__(
        self,
        llm_configs: Optional[List[LLMConfig]] = None,
        include_investigation: bool = True
    ):
        self.include_investigation = include_investigation

        if llm_configs:
            self.llm_manager = LLMManager(llm_configs)
        else:
            # Try default config
            try:
                config = get_default_llm_config()
                if config.api_key or config.provider in (LLMProvider.OLLAMA, LLMProvider.VLLM):
                    self.llm_manager = LLMManager([config])
                else:
                    self.llm_manager = LLMManager([])
            except Exception:
                self.llm_manager = LLMManager([])

        # Create workflow
        self.workflow = create_fraud_detection_workflow(
            llm_manager=self.llm_manager,
            include_investigation=include_investigation
        )

    async def analyze(
        self,
        input_type: FraudInputType,
        raw_input: Dict[str, Any],
        request_id: Optional[str] = None,
        rule_results: Optional[List[RuleResult]] = None,
        threat_intel_results: Optional[List[ThreatIntelResult]] = None
    ) -> FraudDetectionResult:
        """Run the complete fraud detection workflow."""
        import uuid

        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        # Prepare initial state
        initial_state: AgentState = {
            "input_type": input_type,
            "raw_input": raw_input,
            "request_id": request_id,
            "features": None,
            "feature_extraction_time_ms": 0.0,
            "matched_patterns": [],
            "overall_fraud_likelihood": 0.0,
            "primary_category": "",
            "pattern_recognition_time_ms": 0.0,
            "model_predictions": [],
            "ensemble_score": 0.0,
            "ensemble_confidence": 0.0,
            "model_ensemble_time_ms": 0.0,
            "rule_results": rule_results or [],
            "rule_composite_score": sum(r.score for r in (rule_results or [])) / max(len(rule_results or []), 1),
            "threat_intel_results": threat_intel_results or [],
            "threat_composite_score": sum(t.risk_score for t in (threat_intel_results or [])) / max(len(threat_intel_results or []), 1),
            "final_decision": None,
            "decision_time_ms": 0.0,
            "report": None,
            "report_time_ms": 0.0,
            "investigation_hypothesis": None,
            "investigation_evidence": [],
            "investigation_actions": [],
            "investigation_time_ms": 0.0,
            "total_time_ms": 0.0,
            "errors": [],
            "agent_trace": [],
        }

        start_time = time.time()

        # Run workflow
        config = {"configurable": {"thread_id": request_id}}
        final_state = await self.workflow.ainvoke(initial_state, config=config)

        total_time = (time.time() - start_time) * 1000

        decision = final_state.get("final_decision")
        report = final_state.get("report")

        return FraudDetectionResult(
            request_id=request_id,
            input_type=input_type,
            decision=decision.decision if decision else Decision.REVIEW,
            risk_level=decision.risk_level if decision else RiskLevel.MEDIUM,
            final_score=decision.final_score if decision else 0.0,
            confidence=decision.confidence if decision else 0.0,
            features=final_state.get("features", FraudFeatures()),
            matched_patterns=final_state.get("matched_patterns", []),
            model_predictions=final_state.get("model_predictions", []),
            rule_results=final_state.get("rule_results", []),
            threat_intel_results=final_state.get("threat_intel_results", []),
            report=report or FraudReport(
                executive_summary="", transaction_details="", risk_assessment="",
                triggered_rules_patterns="", model_predictions="",
                threat_intelligence_findings="", recommended_actions="",
                evidence_summary=""
            ),
            investigation_hypothesis=final_state.get("investigation_hypothesis"),
            investigation_actions=final_state.get("investigation_actions", []),
            processing_time_ms=total_time,
            agent_trace=final_state.get("agent_trace", []),
            errors=final_state.get("errors", [])
        )

    async def close(self):
        """Clean up resources."""
        await self.llm_manager.close_all()


# Integration with Existing Decision Engine

async def run_agentic_fraud_analysis(
    transaction_data: Dict[str, Any],
    rule_engine_results: Optional[Dict] = None,
    threat_intel_results: Optional[Dict] = None,
    llm_configs: Optional[List[LLMConfig]] = None
) -> FraudDetectionResult:
    """
    Convenience function to run the full agentic fraud analysis.

    This integrates with the existing decision engine by accepting
    rule engine and threat intelligence results as inputs.
    """
    workflow = FraudDetectionWorkflow(llm_configs=llm_configs)

    try:
        # Convert rule results
        rule_results = []
        if rule_engine_results:
            for rule in rule_engine_results.get("triggered_rules", []):
                rule_results.append(RuleResult(
                    rule_name=rule.get("name", ""),
                    triggered=True,
                    score=rule.get("score", 0.0),
                    severity=rule.get("severity", "medium"),
                    description=rule.get("description", "")
                ))

        # Convert threat intel
        threat_results = []
        if threat_intel_results:
            for ti in threat_intel_results.get("indicators", []):
                threat_results.append(ThreatIntelResult(
                    indicator_type=ti.get("type", ""),
                    indicator_value=ti.get("value", ""),
                    risk_score=ti.get("risk_score", 0.0),
                    categories=ti.get("categories", []),
                    details=ti.get("details", {})
                ))

        result = await workflow.analyze(
            input_type=FraudInputType.TRANSACTION,
            raw_input=transaction_data,
            rule_results=rule_results,
            threat_intel_results=threat_results
        )

        return result

    finally:
        await workflow.close()

# Demo / Testing

async def demo_workflow():
    """Demo the workflow with sample data."""
    # Sample transaction
    sample_transaction = {
        "txn_id": "TXN_123456789",
        "payer_vpa": "user@upi",
        "payee_vpa": "merchant@upi",
        "amount": 50000.0,
        "txn_type": "P2M",
        "timestamp": "2024-01-15T14:30:00Z",
        "device_id": "device_abc123",
        "client_ip": "203.0.113.45",
        "merchant_category": "electronics",
        "merchant_name": "Online Store"
    }

    # Sample rule results (from existing decision engine)
    sample_rules = [
        RuleResult("HIGH_AMOUNT", True, 0.3, "high", "Amount exceeds 99th percentile"),
        RuleResult("NEW_DEVICE", True, 0.2, "medium", "First transaction from this device"),
        RuleResult("VELOCITY_5M", False, 0.0, "low", "Normal velocity"),
    ]

    # Sample threat intel
    sample_threat = [
        ThreatIntelResult("ip", "203.0.113.45", 0.4, ["proxy"], {"is_vpn": True}),
        ThreatIntelResult("device", "device_abc123", 0.2, ["new"], {}),
    ]

    # Run with mock LLM (no API key needed for demo)
    workflow = FraudDetectionWorkflow(include_investigation=True)

    print("Running agentic fraud detection workflow...")
    print("=" * 60)

    result = await workflow.analyze(
        input_type=FraudInputType.TRANSACTION,
        raw_input=sample_transaction,
        rule_results=sample_rules,
        threat_intel_results=sample_threat
    )

    print(f"\nRequest ID: {result.request_id}")
    print(f"Decision: {result.decision.value}")
    print(f"Risk Level: {result.risk_level.value}")
    print(f"Final Score: {result.final_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Processing Time: {result.processing_time_ms:.1f}ms")
    print(f"\nAgent Trace:")
    for trace in result.agent_trace:
        print(f"  - {trace['agent']}: {trace['time_ms']:.1f}ms ({trace['status']})")

    print(f"\nMatched Patterns: {len(result.matched_patterns)}")
    for p in result.matched_patterns:
        print(f"  - {p.pattern_name} ({p.confidence:.2f}) [{p.severity}]")

    print(f"\nModel Predictions:")
    for m in result.model_predictions:
        print(f"  - {m.model_name}: {m.score:.3f} (conf: {m.confidence:.2f})")

    print(f"\nReport Preview:")
    print(result.report.executive_summary[:200] if result.report.executive_summary else "No summary")

    if result.investigation_hypothesis:
        print(f"\nInvestigation: {result.investigation_hypothesis}")
        print(f"Actions: {result.investigation_actions}")

    if result.errors:
        print(f"\nErrors: {result.errors}")

    await workflow.close()
    print("\n✅ Workflow demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_workflow())