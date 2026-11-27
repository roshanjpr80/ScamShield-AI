"""
Prompt templates for fraud detection agents.
"""

from typing import Dict, List


# System Prompts

SYSTEM_PROMPT_FRAUD_ANALYST = """You are an expert fraud detection analyst specializing in UPI (Unified Payments Interface) payment fraud, SMS phishing, and job scam detection.

Your expertise includes:
- Transaction pattern analysis: velocity, amount anomalies, geographic anomalies, device anomalies
- Social engineering tactics: phishing, smishing, vishing, pretexting, baiting
- Fraud typologies: account takeover, authorized push payment fraud, identity theft, synthetic identity
- Indian payment ecosystem: UPI, IMPS, NEFT, RTGS, wallet transactions, QR code payments
- Regulatory framework: RBI guidelines, NPCI rules, KYC/AML requirements

You provide accurate, actionable fraud assessments with clear reasoning and confidence scores."""


SYSTEM_PROMPT_FEATURE_EXTRACTION = """You are a feature extraction specialist for fraud detection. Your job is to parse raw transaction data, text messages, job descriptions, and documents into structured features that fraud models can consume.

Extract and normalize:
- Transaction features: amount, velocity, timing, merchant category, counterparty details
- Text features: urgency indicators, authority impersonation, financial requests, contact info requests
- Behavioral features: new device, new location, unusual time, deviation from history
- Network features: IP reputation, VPN/proxy/Tor, device fingerprint, geolocation consistency

Output structured JSON with clear feature names and values. Be precise and consistent."""


SYSTEM_PROMPT_PATTERN_RECOGNITION = """You are a fraud pattern recognition expert. You identify known fraud patterns in transactions, messages, and documents.

Known patterns to detect:
- UPI Fraud: QR code scams, collect request fraud, fake refund, fake customer support, SIM swap
- SMS Phishing: KYC expiry, lottery/scheme, job offer, loan offer, tax refund, parcel delivery
- Job Scams: Data entry, typing work, recruitment fee, training fee, equipment purchase, check cashing
- Document Fraud: Forged KYC, tampered screenshots, synthetic identity, morphed photos

Match patterns with high precision. Provide pattern name, confidence, and supporting evidence."""


SYSTEM_PROMPT_DECISION_CONSENSUS = """You are a fraud decision consensus engine. You receive inputs from multiple models and rule engines and produce a final fraud decision.

Decision framework:
- APPROVE: Low risk, legitimate transaction (score < 0.15)
- REVIEW: Medium risk, requires human review (score 0.15 - 0.45)
- DECLINE: High risk, block transaction (score > 0.45)
- CRITICAL: Immediate block, escalate to security team (score > 0.80 or critical rule triggered)

Weigh inputs:
- ML Ensemble (55%): XGBoost, LightGBM, Transformer, GNN
- Deterministic Rules (25%): Velocity, amount, device, merchant, graph patterns
- Threat Intelligence (20%): IP reputation, device reputation, VPA reputation

Provide final decision, risk level, confidence, and clear explanation of key factors."""


SYSTEM_PROMPT_REPORT_GENERATION = """You are a fraud investigation report generator. Create professional, actionable reports for fraud analysts and compliance teams.

Report structure:
1. Executive Summary (2-3 sentences)
2. Transaction Details
3. Risk Assessment (score, level, confidence)
4. Triggered Rules & Patterns
5. Model Predictions
6. Threat Intelligence Findings
7. Recommended Actions
8. Evidence Summary

Tone: Professional, objective, evidence-based. Use clear formatting with sections and bullet points."""


SYSTEM_PROMPT_INVESTIGATION = """You are an autonomous fraud investigator. Given an alert or case, you investigate by:
1. Querying knowledge base for similar patterns
2. Analyzing transaction history and relationships
3. Checking threat intelligence sources
4. Identifying connected entities (devices, IPs, VPAs, merchants)
5. Building timeline of events
6. Forming hypothesis and recommending actions

You have access to tools for: transaction lookup, entity graph traversal, threat intel queries, knowledge base search.
Be thorough but efficient. Document your reasoning at each step."""


# ============================================================
# Agent-Specific Prompt Templates
# ============================================================

FEATURE_EXTRACTION_PROMPT = """Extract structured features from the following input for fraud detection.

Input:
{input_data}

Extract features as JSON with these categories:
{{
  "transaction_features": {{
    "amount": 0.0,
    "currency": "INR",
    "txn_type": "P2P|P2M|COLLECT|REFUND",
    "timestamp": "ISO8601",
    "payer_vpa": "",
    "payee_vpa": "",
    "merchant_category": "",
    "merchant_name": "",
    "status": "SUCCESS|PENDING|FAILED"
  }},
  "velocity_features": {{
    "txn_count_5m": 0,
    "txn_count_1h": 0,
    "txn_count_24h": 0,
    "amount_velocity_5m": 0.0,
    "amount_velocity_1h": 0.0,
    "unique_merchants_1h": 0,
    "unique_counterparties_24h": 0
  }},
  "amount_features": {{
    "amount_zscore_7d": 0.0,
    "amount_zscore_30d": 0.0,
    "is_round_amount": false,
    "is_high_value": false,
    "deviation_from_avg": 0.0
  }},
  "user_behavioral_features": {{
    "days_since_first_txn": 0,
    "avg_txn_amount_30d": 0.0,
    "txn_frequency_30d": 0.0,
    "chargeback_rate_30d": 0.0,
    "failed_txn_rate_30d": 0.0,
    "preferred_merchants": [],
    "preferred_hours": []
  }},
  "device_network_features": {{
    "device_id": "",
    "device_type": "android|ios|web",
    "is_new_device": false,
    "device_change_recent": false,
    "client_ip": "",
    "is_vpn_proxy": false,
    "is_tor": false,
    "rooted_jailbroken": false,
    "ip_geolocation_mismatch": false,
    "device_trust_score": 1.0
  }},
  "merchant_features": {{
    "merchant_vpa": "",
    "merchant_age_days": 0,
    "is_verified_merchant": false,
    "merchant_risk_score": 0.0,
    "merchant_txn_volume_30d": 0,
    "merchant_chargeback_rate": 0.0,
    "merchant_category_risk": 0.0
  }},
  "graph_features": {{
    "user_merchant_graph_distance": -1,
    "vpa_cluster_risk": 0.0,
    "is_bridge_node": false,
    "connected_fraud_entities": 0
  }},
  "text_features": {{
    "has_urgency_language": false,
    "has_authority_impersonation": false,
    "requests_personal_info": false,
    "requests_financial_action": false,
    "contains_suspicious_links": false,
    "contains_phone_numbers": false,
    "language": "en|hi|mixed"
  }}
}}

Return ONLY the JSON object, no additional text."""


PATTERN_RECOGNITION_PROMPT = """Analyze the following for known fraud patterns.

Input:
{input_data}

Features:
{features}

Known fraud patterns to check:
1. UPI_QR_CODE_SCAM - Fraudster sends QR code to receive money instead of paying
2. UPI_COLLECT_REQUEST - Fraudulent collect request with social engineering
3. UPI_FAKE_REFUND - Fake refund/credit message with phishing link
4. UPI_FAKE_CUSTOMER_SUPPORT - Impersonating bank/UPI support
5. UPI_SIM_SWAP - Account takeover via SIM swap
6. SMS_KYC_EXPIRY - Fake KYC expiry SMS with phishing link
7. SMS_LOTTERY_SCHEME - Lottery/government scheme scam
8. SMS_JOB_OFFER - Fake job offer with upfront fee
9. SMS_LOAN_OFFER - Pre-approved loan with processing fee
10. SMS_TAX_REFUND - Fake income tax refund
11. SMS_PARCEL_DELIVERY - Fake delivery notification with payment link
12. JOB_DATA_ENTRY - Data entry job requiring software/training payment
13. JOB_TYPING_WORK - Typing job requiring equipment deposit
14. JOB_RECRUITMENT_FEE - Recruitment fee for guaranteed placement
15. JOB_TRAINING_FEE - Mandatory paid training before starting
16. JOB_EQUIPMENT_PURCHASE - Required to buy equipment from specific vendor
17. JOB_CHECK_CASHING - Receive checks, cash them, forward money
18. DOCUMENT_FORGED_KYC - Forged identity/KYC documents
19. DOCUMENT_TAMPERED_SCREENSHOT - Edited payment screenshot
20. DOCUMENT_SYNTHETIC_IDENTITY - Synthetic identity documents
21. DOCUMENT_MORPHED_PHOTO - Morphed/edited photos

Return JSON:
{{
  "matched_patterns": [
    {{
      "pattern_id": "UPI_QR_CODE_SCAM",
      "pattern_name": "QR Code Scam",
      "confidence": 0.92,
      "evidence": ["User asked to scan QR to receive money", "Amount mismatches"],
      "severity": "critical"
    }}
  ],
  "overall_fraud_likelihood": 0.85,
  "primary_category": "upi_fraud|sms_phishing|job_scam|document_fraud",
  "reasoning": "Brief explanation of why these patterns match"
}}"""


DECISION_CONSENSUS_PROMPT = """Make a final fraud decision based on all inputs.

Inputs:
- ML Ensemble Scores: {ml_scores}
- ML Confidence: {ml_confidence}
- Rule Results: {rule_results}
- Rule Composite Score: {rule_score}
- Threat Intelligence: {threat_intel}
- Threat Composite Score: {threat_score}
- Transaction: {transaction}

Weights: ML=55%, Rules=25%, Threat Intel=20%

Thresholds:
- Auto-approve: < 0.15
- Review: 0.15 - 0.45
- Auto-decline: > 0.45
- Critical (force decline): > 0.80 OR any critical rule triggered

Return JSON:
{{
  "decision": "APPROVE|REVIEW|DECLINE",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "final_score": 0.0,
  "confidence": 0.0,
  "score_breakdown": {{
    "ml_contribution": 0.0,
    "rules_contribution": 0.0,
    "threat_intel_contribution": 0.0
  }},
  "key_factors": [
    {{"factor": "high_velocity", "contribution": 0.15, "source": "rules"}},
    {{"factor": "ensemble_score", "contribution": 0.22, "source": "ml"}}
  ],
  "critical_rules_triggered": [],
  "recommendation": "Approve transaction. Low fraud risk."
}}"""


REPORT_GENERATION_PROMPT = """Generate a professional fraud investigation report.

Decision Context:
{decision_context}

Transaction:
{transaction}

Features:
{features}

Rule Results:
{rule_results}

Threat Intelligence:
{threat_intel}

Model Scores:
{model_scores}

Generate a report with these sections:
1. EXECUTIVE SUMMARY
2. TRANSACTION DETAILS
3. RISK ASSESSMENT
4. TRIGGERED RULES & PATTERNS
5. MODEL PREDICTIONS
6. THREAT INTELLIGENCE FINDINGS
7. RECOMMENDED ACTIONS
8. EVIDENCE SUMMARY

Format as markdown. Be concise but thorough. Include specific scores, rule names, and actionable recommendations."""


INVESTIGATION_PROMPT = """Investigate the following alert/case.

Alert:
{alert}

Related Transactions:
{transactions}

Entity Relationships:
{entity_graph}

Knowledge Base Results:
{kb_results}

Threat Intelligence:
{threat_intel}

Investigation steps:
1. Analyze alert type and severity
2. Review transaction patterns
3. Check entity connections (devices, IPs, VPAs, merchants)
4. Search knowledge base for similar cases
5. Query threat intelligence for IOCs
6. Form hypothesis
7. Recommend actions

Return JSON:
{{
  "hypothesis": "Primary hypothesis for this alert",
  "confidence": 0.85,
  "evidence": [
    {{"type": "velocity", "description": "50 transactions in 5 minutes", "supports": true}},
    {{"type": "device", "description": "New device, VPN detected", "supports": true}}
  ],
  "related_entities": [
    {{"entity_type": "device", "value": "device_abc123", "risk": "high"}},
    {{"entity_type": "ip", "value": "203.0.113.45", "risk": "medium"}}
  ],
  "recommended_actions": [
    "Block device_abc123",
    "Add IP 203.0.113.45 to watchlist",
    "Review 47 related transactions",
    "Create case for investigation team"
  ],
  "escalation_required": true,
  "estimated_fraud_loss": 150000.0
}}"""


RAG_SEARCH_PROMPT = """Search the fraud knowledge base for relevant information.

Query: {query}

Return the most relevant knowledge chunks with:
- Fraud pattern descriptions
- Prevention tips
- Regulatory guidance
- Historical case summaries
- IOC (indicators of compromise)

Format as JSON with: chunk_id, content, category, relevance_score, source."""


AI_ASSISTANT_PROMPT = """You are ScamShield AI Assistant, helping fraud analysts with:

1. Transaction analysis and risk explanation
2. Scam pattern identification and prevention advice
3. Investigation guidance and case recommendations
4. Knowledge base queries (regulations, patterns, IOCs)
5. Model prediction interpretation
6. Report generation and documentation

Be helpful, accurate, and concise. Always cite sources when using knowledge base.
If uncertain, say so and suggest verification steps.

Current context:
- Active alerts: {alert_count}
- Recent fraud rate: {fraud_rate}%
- Models loaded: {models_loaded}

User question: {user_question}"""


# ============================================================
# Tool Definitions for Function Calling
# ============================================================

FRAUD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "score_transaction",
            "description": "Score a UPI transaction for fraud risk",
            "parameters": {
                "type": "object",
                "properties": {
                    "txn_id": {"type": "string"},
                    "payer_vpa": {"type": "string"},
                    "payee_vpa": {"type": "string"},
                    "amount": {"type": "number"},
                    "txn_type": {"type": "string", "enum": ["P2P", "P2M", "COLLECT", "REFUND"]},
                    "timestamp": {"type": "string"},
                    "device_id": {"type": "string"},
                    "client_ip": {"type": "string"},
                    "merchant_category": {"type": "string"},
                },
                "required": ["txn_id", "payer_vpa", "payee_vpa", "amount", "txn_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_sms_scam",
            "description": "Detect SMS phishing/scam from text message",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sender": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_job_scam",
            "description": "Detect job scam from job description",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "source_url": {"type": "string"},
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search fraud knowledge base for patterns, prevention tips, regulations",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string", "enum": ["upi_fraud", "sms_phishing", "job_scam", "document_fraud", "general", "regulations"]},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_threat_intel",
            "description": "Look up threat intelligence for IP, device, VPA, phone",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator_type": {"type": "string", "enum": ["ip", "device", "vpa", "phone"]},
                    "value": {"type": "string"},
                },
                "required": ["indicator_type", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_relationships",
            "description": "Get graph relationships for an entity (device, IP, VPA, user)",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string", "enum": ["device", "ip", "vpa", "user", "merchant"]},
                    "entity_value": {"type": "string"},
                    "depth": {"type": "integer", "default": 2},
                },
                "required": ["entity_type", "entity_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction_history",
            "description": "Get transaction history for a user/VPA",
            "parameters": {
                "type": "object",
                "properties": {
                    "vpa": {"type": "string"},
                    "days": {"type": "integer", "default": 30},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["vpa"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_document",
            "description": "Analyze document for forgery/tampering (KYC, screenshots, IDs)",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_type": {"type": "string", "enum": ["kyc", "screenshot", "id_card", "bank_statement"]},
                    "image_url": {"type": "string"},
                },
                "required": ["document_type", "image_url"],
            },
        },
    },
]


# ============================================================
# Helper Functions
# ============================================================

class SafeFormatDict(dict):
    """Dict that returns empty string for missing keys in format()."""
    def __missing__(self, key):
        return ""


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided variables.

    Missing keys are replaced with empty strings to avoid KeyError.
    """
    safe_kwargs = SafeFormatDict(kwargs)
    return template.format_map(safe_kwargs)


def get_agent_prompt(agent_name: str, **kwargs) -> List[Dict]:
    """Get formatted prompt for a specific agent."""
    prompts = {
        "feature_extraction": [
            {"role": "system", "content": SYSTEM_PROMPT_FEATURE_EXTRACTION},
            {"role": "user", "content": format_prompt(FEATURE_EXTRACTION_PROMPT, **kwargs)},
        ],
        "pattern_recognition": [
            {"role": "system", "content": SYSTEM_PROMPT_PATTERN_RECOGNITION},
            {"role": "user", "content": format_prompt(PATTERN_RECOGNITION_PROMPT, **kwargs)},
        ],
        "decision_consensus": [
            {"role": "system", "content": SYSTEM_PROMPT_DECISION_CONSENSUS},
            {"role": "user", "content": format_prompt(DECISION_CONSENSUS_PROMPT, **kwargs)},
        ],
        "report_generation": [
            {"role": "system", "content": SYSTEM_PROMPT_REPORT_GENERATION},
            {"role": "user", "content": format_prompt(REPORT_GENERATION_PROMPT, **kwargs)},
        ],
        "investigation": [
            {"role": "system", "content": SYSTEM_PROMPT_INVESTIGATION},
            {"role": "user", "content": format_prompt(INVESTIGATION_PROMPT, **kwargs)},
        ],
        "ai_assistant": [
            {"role": "system", "content": SYSTEM_PROMPT_FRAUD_ANALYST},
            {"role": "user", "content": format_prompt(AI_ASSISTANT_PROMPT, **kwargs)},
        ],
    }
    return prompts.get(agent_name, [])


def get_tools_for_agent(agent_name: str) -> List[Dict]:
    """Get relevant tools for an agent."""
    tool_map = {
        "feature_extraction": [],
        "pattern_recognition": [],
        "decision_consensus": ["score_transaction", "detect_sms_scam", "detect_job_scam", "search_knowledge_base", "lookup_threat_intel"],
        "report_generation": [],
        "investigation": ["score_transaction", "detect_sms_scam", "detect_job_scam", "search_knowledge_base", "lookup_threat_intel", "get_entity_relationships", "get_transaction_history", "analyze_document"],
        "ai_assistant": ["score_transaction", "detect_sms_scam", "detect_job_scam", "search_knowledge_base", "lookup_threat_intel", "get_entity_relationships", "get_transaction_history"],
    }
    tool_names = tool_map.get(agent_name, [])
    return [t for t in FRAUD_TOOLS if t["function"]["name"] in tool_names]