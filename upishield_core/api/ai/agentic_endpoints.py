"""
Agentic AI API Endpoints for Fraud Detection.

Provides REST endpoints for the LangGraph multi-agent workflow.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from upishield_core.ai import (
    Decision,
    FraudDetectionResult,
    FraudDetectionWorkflow,
    FraudFeatures,
    FraudInputType,
    FraudReport,
    FraudInputType,
    LLMConfig,
    LLMManager,
    LLMProvider,
    ModelPrediction,
    PatternMatch,
    RiskLevel,
    RuleResult,
    ThreatIntelResult,
    run_agentic_fraud_analysis,
)
from upishield_core.api.ai.assistant import AIAssistant
from upishield_core.api.schemas import ScoreRequest, ScoreResponse

router = APIRouter(prefix="/api/v1/agentic", tags=["Agentic AI"])


# Pydantic Models for API

class AgenticAnalysisRequest(BaseModel):
    """Request for agentic fraud analysis."""
    input_type: FraudInputType = Field(..., description="Type of input to analyze")
    raw_input: Dict[str, Any] = Field(..., description="Raw input data")
    rule_results: Optional[List[Dict[str, Any]]] = Field(None, description="Rule engine results")
    threat_intel_results: Optional[List[Dict[str, Any]]] = Field(None, description="Threat intelligence results")
    request_id: Optional[str] = Field(None, description="Optional request ID")
    include_investigation: bool = Field(True, description="Include investigation agent")


class AgenticAnalysisResponse(BaseModel):
    """Response from agentic fraud analysis."""
    request_id: str
    input_type: str
    decision: str
    risk_level: str
    final_score: float
    confidence: float
    features: Optional[Dict[str, Any]] = None
    matched_patterns: List[Dict[str, Any]] = []
    model_predictions: List[Dict[str, Any]] = []
    rule_results: List[Dict[str, Any]] = []
    threat_intel_results: List[Dict[str, Any]] = []
    report: Optional[Dict[str, Any]] = None
    investigation_hypothesis: Optional[str] = None
    investigation_actions: List[str] = []
    processing_time_ms: float
    agent_trace: List[Dict[str, Any]] = []
    errors: List[str] = []


class WorkflowStatusResponse(BaseModel):
    """Workflow status and health."""
    status: str
    version: str
    agents: List[str]
    llm_providers: List[str]
    last_updated: str


# Global workflow instance
_workflow: Optional[FraudDetectionWorkflow] = None


def get_workflow() -> FraudDetectionWorkflow:
    """Get or create the workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = FraudDetectionWorkflow()
    return _workflow


# Endpoints

@router.post("/analyze", response_model=AgenticAnalysisResponse)
async def analyze_fraud(
    request: AgenticAnalysisRequest,
    workflow: FraudDetectionWorkflow = Depends(get_workflow),
):
    """
    Run the full agentic fraud detection workflow.

    This endpoint executes the 5-agent LangGraph workflow:
    1. FeatureExtractionAgent - Parse raw input → structured features
    2. PatternRecognitionAgent - Match against known fraud patterns
    3. ModelEnsembleAgent - Run predictions across multiple models
    4. DecisionConsensusAgent - Weighted voting with explanations
    5. ReportGenerationAgent - Generate human-readable fraud reports
    6. InvestigationAgent (optional) - Autonomous investigation with tool use
    """
    try:
        # Convert rule results
        rule_results = []
        if request.rule_results:
            for rule in request.rule_results:
                rule_results.append(RuleResult(
                    rule_name=rule.get("name", ""),
                    triggered=rule.get("triggered", False),
                    score=rule.get("score", 0.0),
                    severity=rule.get("severity", "medium"),
                    description=rule.get("description", "")
                ))

        # Convert threat intel
        threat_results = []
        if request.threat_intel_results:
            for ti in request.threat_intel_results:
                threat_results.append(ThreatIntelResult(
                    indicator_type=ti.get("type", ""),
                    indicator_value=ti.get("value", ""),
                    risk_score=ti.get("risk_score", 0.0),
                    categories=ti.get("categories", []),
                    details=ti.get("details", {})
                ))

        # Run workflow
        result = await workflow.analyze(
            input_type=request.input_type,
            raw_input=request.raw_input,
            request_id=request.request_id,
            rule_results=rule_results,
            threat_intel_results=threat_results
        )

        return AgenticAnalysisResponse(
            request_id=result.request_id,
            input_type=result.input_type.value,
            decision=result.decision.value,
            risk_level=result.risk_level.value,
            final_score=result.final_score,
            confidence=result.confidence,
            features=result.features.to_dict() if result.features else None,
            matched_patterns=[
                {
                    "pattern_id": p.pattern_id,
                    "pattern_name": p.pattern_name,
                    "confidence": p.confidence,
                    "evidence": p.evidence,
                    "severity": p.severity
                }
                for p in result.matched_patterns
            ],
            model_predictions=[
                {
                    "model_name": m.model_name,
                    "score": m.score,
                    "confidence": m.confidence,
                    "metadata": m.metadata
                }
                for m in result.model_predictions
            ],
            rule_results=[
                {
                    "rule_name": r.rule_name,
                    "triggered": r.triggered,
                    "score": r.score,
                    "severity": r.severity,
                    "description": r.description
                }
                for r in result.rule_results
            ],
            threat_intel_results=[
                {
                    "indicator_type": t.indicator_type,
                    "indicator_value": t.indicator_value,
                    "risk_score": t.risk_score,
                    "categories": t.categories,
                    "details": t.details
                }
                for t in result.threat_intel_results
            ],
            report=result.report.__dict__ if result.report else None,
            investigation_hypothesis=result.investigation_hypothesis,
            investigation_actions=result.investigation_actions,
            processing_time_ms=result.processing_time_ms,
            agent_trace=result.agent_trace,
            errors=result.errors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.post("/analyze/transaction", response_model=AgenticAnalysisResponse)
async def analyze_transaction(
    transaction_data: Dict[str, Any] = Body(...),
    rule_results: Optional[List[Dict[str, Any]]] = Body(None),
    threat_intel_results: Optional[List[Dict[str, Any]]] = Body(None),
    request_id: Optional[str] = Query(None),
    workflow: FraudDetectionWorkflow = Depends(get_workflow),
):
    """
    Analyze a UPI transaction with the agentic workflow.

    Convenience endpoint for transaction analysis with pre-filled input_type.
    """
    request = AgenticAnalysisRequest(
        input_type=FraudInputType.TRANSACTION,
        raw_input=transaction_data,
        rule_results=rule_results,
        threat_intel_results=threat_intel_results,
        request_id=request_id,
    )
    return await analyze_fraud(request, workflow)


@router.post("/analyze/sms", response_model=AgenticAnalysisResponse)
async def analyze_sms(
    text: str = Body(..., embed=True),
    sender: Optional[str] = Body(None, embed=True),
    request_id: Optional[str] = Query(None),
    workflow: FraudDetectionWorkflow = Depends(get_workflow),
):
    """
    Analyze an SMS for phishing/scam with the agentic workflow.
    """
    request = AgenticAnalysisRequest(
        input_type=FraudInputType.SMS,
        raw_input={"text": text, "sender": sender},
        request_id=request_id,
    )
    return await analyze_fraud(request, workflow)


@router.post("/analyze/job", response_model=AgenticAnalysisResponse)
async def analyze_job_posting(
    description: str = Body(..., embed=True),
    source_url: Optional[str] = Body(None, embed=True),
    request_id: Optional[str] = Query(None),
    workflow: FraudDetectionWorkflow = Depends(get_workflow),
):
    """
    Analyze a job posting for scam indicators with the agentic workflow.
    """
    request = AgenticAnalysisRequest(
        input_type=FraudInputType.JOB_POSTING,
        raw_input={"description": description, "source_url": source_url},
        request_id=request_id,
    )
    return await analyze_fraud(request, workflow)


@router.get("/workflow/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow: FraudDetectionWorkflow = Depends(get_workflow),
):
    """Get workflow status and configuration."""
    return WorkflowStatusResponse(
        status="ready" if workflow.llm_manager.clients else "no_llm_configured",
        version="1.0.0",
        agents=[
            "FeatureExtractionAgent",
            "PatternRecognitionAgent",
            "ModelEnsembleAgent",
            "DecisionConsensusAgent",
            "ReportGenerationAgent",
            "InvestigationAgent"
        ],
        llm_providers=[c.config.provider.value for c in workflow.llm_manager.clients],
        last_updated=datetime.utcnow().isoformat()
    )


@router.post("/workflow/configure")
async def configure_workflow(
    providers: List[Dict[str, Any]] = Body(...),
    workflow: FraudDetectionWorkflow = Depends(get_workflow),
):
    """
    Configure LLM providers for the workflow.

    Example request:
    {
        "providers": [
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-...",
                "temperature": 0.1,
                "max_tokens": 4000
            }
        ]
    }
    """
    try:
        configs = []
        for p in providers:
            configs.append(LLMConfig(
                provider=LLMProvider(p["provider"]),
                model=p["model"],
                api_key=p.get("api_key"),
                base_url=p.get("base_url"),
                temperature=p.get("temperature", 0.1),
                max_tokens=p.get("max_tokens", 4000),
                api_version=p.get("api_version"),
                deployment_name=p.get("deployment_name"),
            ))

        # Recreate workflow with new config
        global _workflow
        await workflow.close()
        _workflow = FraudDetectionWorkflow(llm_configs=configs)

        return {"status": "configured", "providers": [p["provider"] for p in providers]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Configuration failed: {str(e)}")


# Integration with Existing Scoring Endpoint

async def enhance_score_with_agentic(
    score_request: ScoreRequest,
    agentic_result: FraudDetectionResult
) -> ScoreResponse:
    """
    Enhance existing score response with agentic analysis results.
    """
    from upishield_core.api.schemas import (
        ExplanationResponse,
        FeatureImportance,
        RuleTriggered,
    )

    # Build enhanced explanation
    explanation = ExplanationResponse(
        summary=f"Agentic analysis: {agentic_result.decision.value} with {agentic_result.risk_level.value} risk",
        key_factors=agentic_result.final_decision.key_factors if agentic_result.final_decision else [],
        rule_details=[
            {
                "rule": r.rule_name,
                "triggered": r.triggered,
                "score": r.score,
                "severity": r.severity,
                "description": r.description
            }
            for r in agentic_result.rule_results
        ],
        model_details={
            "ensemble_score": agentic_result.ensemble_score if hasattr(agentic_result, 'ensemble_score') else 0,
            "model_predictions": [
                {"model": m.model_name, "score": m.score, "confidence": m.confidence}
                for m in agentic_result.model_predictions
            ]
        },
        confidence=agentic_result.confidence,
    )

    return ScoreResponse(
        txn_id=score_request.transaction.txn_id,
        decision=agentic_result.decision.value,
        risk_score=agentic_result.final_score,
        risk_level=agentic_result.risk_level.value,
        explanation=explanation,
        ml_confidence=agentic_result.confidence,
    )


# Streaming Endpoint (for real-time updates)

@router.post("/analyze/stream")
async def analyze_fraud_stream(
    request: AgenticAnalysisRequest,
    workflow: FraudDetectionWorkflow = Depends(get_workflow),
):
    """
    Stream agentic analysis results as they become available.

    Returns Server-Sent Events (SSE) with agent progress updates.
    """
    from fastapi.responses import StreamingResponse

    async def generate():
        import json

        # Send initial status
        yield f"data: {json.dumps({'type': 'start', 'request_id': request.request_id or 'new'})}\n\n"

        try:
            # Convert rule results
            rule_results = []
            if request.rule_results:
                for rule in request.rule_results:
                    rule_results.append(RuleResult(
                        rule_name=rule.get("name", ""),
                        triggered=rule.get("triggered", False),
                        score=rule.get("score", 0.0),
                        severity=rule.get("severity", "medium"),
                        description=rule.get("description", "")
                    ))

            threat_results = []
            if request.threat_intel_results:
                for ti in request.threat_intel_results:
                    threat_results.append(ThreatIntelResult(
                        indicator_type=ti.get("type", ""),
                        indicator_value=ti.get("value", ""),
                        risk_score=ti.get("risk_score", 0.0),
                        categories=ti.get("categories", []),
                        details=ti.get("details", {})
                    ))

            # Set global LLM manager for agent nodes to use
            from upishield_core.ai.agentic_workflow import set_workflow_llm_manager
            set_workflow_llm_manager(workflow.llm_manager)

            # Prepare initial state (without LLMManager - agents use global)
            initial_state: Dict[str, Any] = {
                "input_type": request.input_type,
                "raw_input": request.raw_input,
                "request_id": request.request_id or str(uuid.uuid4())[:8],
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
                "rule_results": rule_results,
                "rule_composite_score": sum(r.score for r in rule_results) / max(len(rule_results), 1),
                "threat_intel_results": threat_results,
                "threat_composite_score": sum(t.risk_score for t in threat_results) / max(len(threat_results), 1),
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

            # Execute workflow with streaming updates
            config = {"configurable": {"thread_id": initial_state["request_id"]}}

            async for event in workflow.workflow.astream(initial_state, config=config):
                for node_name, node_state in event.items():
                    # Send progress update for each node completion
                    trace = node_state.get("agent_trace", [])
                    if trace:
                        latest = trace[-1]
                        yield f"data: {json.dumps({'type': 'agent_complete', 'agent': latest['agent'], 'status': latest['status'], 'time_ms': latest['time_ms']})}\n\n"

            # Get final state
            final_state = await workflow.workflow.ainvoke(initial_state, config=config)

            # Send final result
            report = final_state.get("report")
            report_dict = report.__dict__ if report else None

            features = final_state.get("features", FraudFeatures())
            features_dict = features.to_dict() if hasattr(features, 'to_dict') else {}

            result = FraudDetectionResult(
                request_id=final_state["request_id"],
                input_type=final_state["input_type"],
                decision=final_state["final_decision"].decision if final_state.get("final_decision") else Decision.REVIEW,
                risk_level=final_state["final_decision"].risk_level if final_state.get("final_decision") else RiskLevel.MEDIUM,
                final_score=final_state["final_decision"].final_score if final_state.get("final_decision") else 0.0,
                confidence=final_state["final_decision"].confidence if final_state.get("final_decision") else 0.0,
                features=features_dict,
                matched_patterns=final_state.get("matched_patterns", []),
                model_predictions=final_state.get("model_predictions", []),
                rule_results=final_state.get("rule_results", []),
                threat_intel_results=final_state.get("threat_intel_results", []),
                report=report_dict,
                investigation_hypothesis=final_state.get("investigation_hypothesis"),
                investigation_actions=final_state.get("investigation_actions", []),
                processing_time_ms=final_state.get("total_time_ms", 0.0),
                agent_trace=final_state.get("agent_trace", []),
                errors=final_state.get("errors", [])
            )

            yield f"data: {json.dumps({'type': 'complete', 'result': result.__dict__}, default=str)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Cleanup on Shutdown

async def close_workflow():
    """Close the workflow on application shutdown."""
    global _workflow
    if _workflow:
        await _workflow.close()
        _workflow = None