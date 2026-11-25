"""
AI Assistant for fraud detection, education, and investigation support.

Provides intelligent responses for:
- Transaction analysis and fraud patterns
- Scam detection education
- Investigation assistance
- Knowledge base Q&A
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ConversationMemory:
    """In-memory conversation store with persistence."""

    def __init__(self, max_sessions: int = 50):
        self.sessions: Dict[str, List[Dict]] = {}
        self.max_sessions = max_sessions

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add message to session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history for a session."""
        return self.sessions.get(session_id, [])[-limit:]

    def cleanup_old_sessions(self, keep_order: List[str] = None) -> None:
        """Remove oldest sessions if exceeding max_sessions."""
        if len(self.sessions) <= self.max_sessions:
            return

        # Sort by last message timestamp
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: max(m.get("timestamp", "") for m in x[1]) if x[1] else "",
        )

        # Remove oldest sessions
        while len(self.sessions) > self.max_sessions:
            oldest_id = sorted_sessions.pop(0)[0]
            del self.sessions[oldest_id]


class IntentClassifier:
    """Classify user intent from message content."""

    INTENT_PATTERNS = {
        "transaction_analysis": [
            "score this transaction", "check transaction", "analyze txn",
            "is this fraud", "fraud check", "transaction analysis",
        ],
        "knowledge_question": [
            "what is", "what are", "define", "explain", "how does",
            "meaning of", "tell me about", "describe", "types of",
            "examples of", "how to identify", "how to detect",
        ],
        "investigation": [
            "investigate", "case", "evidence", "timeline", "pattern",
            "related transactions", "similar cases", "analysis",
        ],
        "rule_builder": [
            "create rule", "build rule", "alert rule", "condition",
            "rule for", "if transaction", "when txn",
        ],
        "model_insights": [
            "model performance", "model metrics", "accuracy",
            "precision", "recall", "f1 score", "confusion matrix",
        ],
        "compliance": [
            "compliance", "audit", "report", "regulation", "rbi",
            "npci", "pci dss", "gdpr", "sla",
        ],
    }

    @classmethod
    def classify(cls, text: str) -> str:
        """Classify intent from text."""
        lower = text.lower().strip()

        for intent, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in lower:
                    return intent

        return "general"


class ResponseGenerator:
    """Generate context-aware responses for different intents."""

    KNOWLEDGE_RESPONSES = {
        "upi_scam": {
            "what is upi scam": {
                "response": "UPI scam refers to unauthorized transactions made through India's Unified Payments Interface. Fraudsters exploit the instant, real-time nature of UPI payments. Key types include:\n\n1. **Collect Request Scam (Most Common)**: Fraudster sends a 'Request Money' instead of 'Pay'. You think you're receiving money, but actually authorize money OUT of your account.\n\n2. **QR Code Scams**: Fraudster sends a QR code saying 'Scan to receive ₹5000'. You scan and approve, but money goes TO them instead.\n\n3. **Fake Support Scams**: Caller pretends to be from Google Pay/PhonePe support, asks for UPI PIN or remote access app.\n\n4. **KYC Phishing**: Fake SMS says 'Your UPI/KYC expired, verify now' with a phishing link.\n\n**Red Flags**: Never approve collect requests you didn't initiate, never share PIN/OTP, verify sender identity, be wary of 'cashback' or 'refund' offers.",
                "confidence": 0.95,
                "sources": ["upi_fraud", "protection_tips"],
            },
            "what are upi scams": {
                "response": "Major UPI scam types to watch for:\n\n• **Collect Request Scam**: You receive a request to pay (pull) rather than pay (push). Fraudster tricks you into authorizing money transfer.\n\n• **QR Code Payment Scam**: Fraudster sends QR code to receive money - opposite of legitimate QR payment.\n\n• **Fake UPI Apps**: Fake 'UPI PIN Manager' apps steal credentials when you enter your PIN.\n\n• **Super QR Scams**: QR codes on websites/ads that look legitimate but contain malicious links.\n\n• **Social Engineering**: 'I'll pay you' messages that turn into 'collect' requests after you enter PIN.\n\n**Safe Practice**: Always verify the direction of payment before approving UPI transactions.",
                "confidence": 0.92,
                "sources": ["upi_fraud", "qr_code_scams", "protection_tips"],
            },
        },
        "job_scam": {
            "what is job scam": {
                "response": "A job scam is a fraudulent employment opportunity where scammers exploit job seekers. Common tactics:\n\n1. **Upfront Fees**: Ask for money for 'registration', 'training', 'equipment', or 'background check'. Legitimate employers never pay you.\n\n2. **Fake Offer Letters**: Use logos of real companies with fake email domains (@gmail.com instead of @company.com).\n\n3. **Money Mule Recruitment**: Hire you to 'reship packages' - actually handling stolen goods purchased with stolen cards.\n\n4. **Phishing for Documents**: Request PAN, Aadhaar, passport copies, bank details for 'verification'.\n\n5. **Work-from-Home Scams**: Promise unrealistic pay for simple tasks like data entry with 'processing fees'.\n\n**Protect Yourself**: Never pay money to get a job. Verify company independently. Check email domains. Report fake listings to the platform.",
                "confidence": 0.94,
                "sources": ["job_scam", "prevention"],
            },
        },
        "sms_phishing": {
            "what is sms phishing": {
                "response": "SMS phishing (smishing) is fraudulent text messages designed to steal credentials, money, or install malware. Key tactics:\n\n1. **Bank Alert Spoofing**: 'Your account debited ₹15,000. If not you, click: link'. Official banks never ask for sensitive info via SMS links.\n\n2. **Delivery Scams**: 'Package held at customs. Pay ₹2,500: link'. Legitimate couriers don't demand payment via SMS.\n\n3. **Prize/Lottery**: 'You won ₹10 Lakhs! Claim: link'. You can't win a lottery you didn't enter.\n\n4. **Authority Impersonation**: Fake IRS/Police notices demanding payment. Government agencies don't use SMS for such communications.\n\n5. **UPI PIN Requests**: 'Your UPI PIN has been compromised, verify now'. Never share PINs.\n\n**Identify Smishing**: Unknown sender, urgent language, links/shortened URLs, requests for OTP/PIN/password, poor grammar.",
                "confidence": 0.93,
                "sources": ["smishing", "identification", "protection"],
            },
        },
        "transaction_analysis": {
            "analyze transaction": {
                "response": "For transaction analysis, I need specific details to provide an accurate fraud risk assessment. Please provide:\n\n1. **Transaction details**: Amount, date/time, parties involved (VPA/phone numbers)\n2. **Transaction type**: P2P, P2M, COLLECT, REFUND, etc.\n3. **Context**: Is this a new/payee? Any recent changes?\n4. **Red flags observed**: Unusual amount, new device, odd timing, etc.\n\nKey indicators of suspicious transactions to watch for:\n• Amount significantly larger than usual\n• New/unfamiliar payee\n• Transaction during unusual hours\n• Multiple rapid transactions\n• Request from unknown contact\n• Pressure tactics ('urgent', 'limited time')\n• Request for personal/UPI PIN/OTP\n\nBased on this info, I can analyze the transaction for risk patterns common in UPI fraud.",
                "confidence": 0.88,
                "sources": ["transaction_fraud_patterns", "fraud_indicators"],
            },
            "check if fraud": {
                "response": "To determine if a transaction is fraudulent, I analyze these key indicators:\n\n**High-Risk Patterns:**\n• **Velocity anomalies**: Multiple transactions in short time\n• **Amount anomalies**: Spending beyond normal patterns\n• **Direction changes**: Unexpected collect requests\n• **Timing**: Transactions at unusual hours\n• **New payees**: First-time recipients of large amounts\n\n**Warning Signs:**\n• Requests from unknown numbers/QR codes\n• \"Too good to be true\" offers (cashback, prizes)\n• Pressure to act quickly\n• Requests for OTP/verification codes\n• Link clicking before approving\n\nWould you like to share transaction details for a specific analysis? I can check for these fraud patterns.",
                "confidence": 0.85,
                "sources": ["fraud_detection", "pattern_analysis"],
            },
        },
        "fraud_prevention": {
            "how to protect from upi scams": {
                "response": "Protect yourself from UPI scams with these essential practices:\n\n1. **Never approve Collect Requests** you didn't initiate yourself.\n2. **Verify payment direction** before approving - check if sending or receiving.\n3. **Never share UPI PIN, OTP, or passwords** with anyone.\n4. **Enable device PIN/biometrics** for additional protection.\n5. **Use official apps only** - download from Google Play/App Store.\n6. **Check app permissions** - legitimate UPI apps have reasonable permissions.\n7. **Monitor transactions** regularly in your UPI app history.\n8. **Report suspicious transactions** immediately to your bank.\n9. **Be wary of 'too good' offers** - cashback, prizes, 'free gifts'.\n10. **Verify caller identity** if someone claims to be from your bank/app.\n\nAlways contact your bank through official channels (saved numbers, app help) for verification.",
                "confidence": 0.96,
                "sources": ["protection_tips", "best_practices"],
            },
            "how to avoid scams": {
                "response": "General scam prevention strategies:\n\n**Verify Before Acting:**\n• Call companies directly on official numbers (not caller ID)\n• Check emails for official domain names\n• Research job offers through company websites\n\n**Never Share:**\n• Personal identification (Aadhaar, PAN)\n• Financial details\n• UPI PIN, OTP, passwords\n• One-time codes sent to you\n\n**Red Flags to Watch:**\n• Urgent/emergency pressure\n• Requests for upfront payment\n• Unusually high rewards\n• Too-good-to-be-true offers\n• Poor grammar/spelling\n• Unauthenticated contacts\n\n**Safe Practices:**\n• Use official apps/websites only\n• Enable two-factor authentication\n• Keep software updated\n• Use secure networks\n• Trust your instincts - if it feels wrong, verify independently\n\nWhen in doubt, contact the official support channel of the organization.",
                "confidence": 0.92,
                "sources": ["scam_prevention", "security_best_practices"],
            },
        },
        "fraud_detection": {
            "what are common fraud types": {
                "response": "Common fraud types to watch for:\n\n**Payment/Transaction Fraud:**\n• UPI Collect Request Scams (most common)\n• QR Code Payment Scams\n• Fake Refund/Cashback Scams\n• Account Takeover using shared credentials\n\n**Digital/Online Scams:**\n• SMS Phishing (Smishing)\n• Email Phishing (Spear-phishing, Whaling)\n• Vishing (Voice Phishing)\n• Fake Online Shopping/Job portals\n\n**Identity & Document Scams:**\n• Aadhaar/PAN cloning\n• Fake employment offers\n• Impersonation scams\n• Lottery/Prize scams\n\n**Investment & Financial Scams:**\n• Cryptocurrency investment frauds\n• Fake trading platforms\n• Ponzi schemes\n• Unregistered investment offers\n\nEach has distinctive patterns - would you like details on any specific type?",
                "confidence": 0.94,
                "sources": ["fraud_categories", "scam_types"],
            },
            "types of scams": {
                "response": "Major scam categories include:\n\n**1. UPI Payment Scams:**\n- Collect Request Scam: You unknowingly authorize money transfer\n- QR Code Scam: Fraudulent QR codes for reverse payments\n- Refund Scam: Fake cashback/refund offers\n\n**2. SMS/WhatsApp Scams:**\n- Bank alert spoofing\n- Delivery/prize notifications\n- Support impersonation\n- UPI PIN requests\n\n**3. Job/Work-at-Home Scams:**\n- Upfront payment demands\n• Fake offer letters\n• Document verification scams\n• Package forwarding (money mule)\n\n**4. Investment Scams:**\n- Cryptocurrency frauds\n- Fake trading platforms\n• Ponzi schemes\n• Unregistered investment offers\n\n**5. Phishing Attacks:**\n- Email phishing\n- Phone/SMS phishing (smishing)\n• Social media impersonation\n\nEach scam type exploits human psychology - urgency, greed, fear, trust.",
                "confidence": 0.91,
                "sources": ["scam_classification", "fraud_prevention"],
            },
        },
        "compliance": {
            "what are compliance requirements": {
                "response": "Fraud detection compliance requirements include:\n\n**Indian Regulations:**\n• **RBI Guidelines**: Mandatory fraud reporting, customer awareness\n• **PCI DSS**: Card data security standards (if processing cards)\n• **PFSA**: Payment and Settlement Systems Act compliance\n• **DPDP Act 2023**: Data privacy and protection requirements\n• **NPCI UPI Guidelines**: Transaction monitoring and fraud prevention\n\n**Key Compliance Obligations:**\n1. **Fraud Reporting**: Report suspected fraud to RBI/RPS via online portals\n2. **Customer Notification**: Inform customers of suspicious transactions\n3. **Data Retention**: Maintain transaction logs for regulatory review\n4. **SLA Compliance**: Response time commitments for fraud alerts\n5. **Audit Trails**: Maintain complete transaction audit records\n6. **Incident Response**: Documented procedures for fraud incidents\n\n**Best Practices:**\n• Regular fraud risk assessments\n• Staff training on fraud patterns\n• Customer education programs\n• Secure data handling procedures\n• Incident documentation procedures",
                "confidence": 0.88,
                "sources": ["rbi_guidelines", "compliance_standards"],
            },
        },
        "model_insights": {
            "model performance": {
                "response": "Fraud detection model performance metrics:\n\n**Key Metrics:**\n• **Precision**: % of flagged transactions that are actually fraud (minimize false positives)\n• **Recall**: % of actual fraud correctly identified (minimize false negatives)\n• **F1 Score**: Harmonic mean of precision and recall\n• **AUC-ROC**: Area under ROC curve (overall discriminative ability)\n• **AUPRC**: Area under Precision-Recall curve (better for imbalanced data)\n\n**Typical UPI Fraud Model Performance:**\n• Precision: 85-92%\n• Recall: 78-88%\n• F1 Score: 0.82-0.90\n• AUC-ROC: 0.94-0.97\n\n**Interpretation:**\nHigher precision = fewer legitimate transactions blocked\nHigher recall = catch more actual fraud cases\nTrade-off depends on business risk tolerance\n\n**Model Confidence:**\nModels provide confidence scores (0-1) for each prediction. Higher = more certain.\nLow confidence predictions should be reviewed by human analysts.",
                "confidence": 0.90,
                "sources": ["model_performance", "evaluation_metrics"],
            },
            "how models work": {
                "response": "UPI fraud detection models use these approaches:\n\n**Ensemble Methods:**\nMultiple models combined for robust predictions:\n• **XGBoost**: Gradient boosting for structured features\n• **LightGBM**: Fast tree-based learning\n• **Transformer models**: Sequence-based pattern detection\n• **Neural networks**: Complex feature interactions\n\n**Feature Engineering:**\n• **Static features**: Amount, VPA patterns, time, device\n• **Behavioral features**: Transaction velocity, deviation from norms\n• **Geospatial features**: Location patterns, unusual shifts\n• **Network features**: Device fingerprinting, account relationships\n• **Temporal features**: Time-based patterns, periodicity\n• **Entity features**: Customer profiles, merchant categories\n\n**Decision Process:**\n1. Real-time feature extraction\n2. Parallel model inference\n3. Weighted scoring\n4. Confidence calculation\n5. Decision threshold application\n6. Risk level classification\n\n**Output:** APPROVE (low risk), REVIEW (medium), or DECLINE (high risk)",
                "confidence": 0.87,
                "sources": ["machine_learning", "fraud_detection"],
            },
        },
        "sms_detection": {
            "is this sms phishing": {
                "response": "To determine if an SMS is phishing, look for these warning signs:\n\n**Immediate Red Flags:**\n• **Urgent language**: 'URGENT', 'IMMEDIATE ACTION', 'LIMITED TIME'\n• **Threats**: 'Account blocked', 'OTP required', 'Verification needed'\n• **Too good offers**: 'Cashback', 'Prize', 'Free gift', 'Refund'\n• **Requests for codes**: OTP, UPI PIN, password, CVV\n• **Suspicious links**: Shortened URLs, unfamiliar domains\n• **Unrecognized sender**: Unknown number, wrong area code\n\n**Common Phishing Templates:**\n1. Bank impersonation: 'Unusual activity detected, click to secure account'\n2. Delivery scam: 'Package on hold, pay ₹xxx to release'\n3. Prize/lottery: 'You won! Claim at link (expires soon)'\n4. Support scam: 'Call now for assistance', 'Your account at risk'\n5. Verification: 'Verify your account to continue'\n\n**Safe Practices:**\n• Never click links in suspicious SMS\n• Never share OTP/PIN/passwords\n• Contact bank directly using saved numbers\n• Verify through official app/website\n• Report phishing attempts to your bank",
                "confidence": 0.91,
                "sources": ["phishing_detection", "sms_security"],
            },
        },
        "identification": {
            "how to identify fraud": {
                "response": "Identifying potential fraud requires vigilance for these patterns:\n\n**Transaction-Level Indicators:**\n• **Unusual amounts**: Spending beyond normal patterns\n• **New payees**: First-time recipients of large amounts\n• **Timing**: Transactions at unusual hours (late night, holidays)\n• **Velocity**: Multiple transactions in short timeframes\n• **Direction**: Unexpected collect requests, reverse payments\n\n**Communication Patterns:**\n• Urgency pressure: 'Act now', 'Limited time', 'Immediate'\n• Fear tactics: 'Account at risk', 'Verification needed'\n• Too-good offers: Cashback, prizes, 'free gifts'\n• Requests for sensitive info: OTP, PIN, passwords\n• Suspicious links or attachments\n\n**Device/Network Indicators:**\n• New devices not recognized\n• Unusual location patterns\n• Different IP or GPS coordinates\n• Multiple failed authentication attempts\n• Multiple accounts on same device\n\n**When to Act:**\n1. **Trust instincts** - if it feels suspicious, verify independently\n2. **Contact directly** - use verified phone numbers/emails\n3. **Don't click** - navigate to official sites manually\n4. **Document everything** - screenshots, messages, timestamps\n5. **Report immediately** - to bank, authorities, platform",
                "confidence": 0.89,
                "sources": ["fraud_identification", "threat_detection"],
            },
        },
    }

    @classmethod
    def get_response(cls, intent: str, query: str) -> Dict[str, Any]:
        """Get knowledge response for query."""
        # Check direct matches first
        for category, responses in cls.KNOWLEDGE_RESPONSES.items():
            for pattern, data in responses.items():
                if pattern in query.lower():
                    return data

        # Return generic response with categories
        return {
            "response": cls._get_general_response(intent, query),
            "confidence": 0.7,
            "sources": ["knowledge_base"],
        }

    @classmethod
    def _get_general_response(cls, intent: str, query: str) -> str:
        """Generate response based on intent with comprehensive pattern matching."""
        lower = query.lower()

        # Transaction analysis patterns
        if any(kw in lower for kw in ["transaction", "txn", "payment", "transfer", "amount", "send", "receive"]):
            if "fraud" in lower or "scam" in lower or "check" in lower or "risk" in lower:
                return cls.KNOWLEDGE_RESPONSES["transaction_analysis"]["analyze transaction"]["response"]
            return cls.KNOWLEDGE_RESPONSES["transaction_analysis"]["analyze transaction"]["response"]

        # UPI-related queries
        if "upi" in lower or "vpa" in lower or "bhim" in lower or "google pay" in lower or "phonepe" in lower or "paytm" in lower:
            if "scam" in lower or "fraud" in lower or "protect" in lower or "avoid" in lower:
                return cls.KNOWLEDGE_RESPONSES["upi_scam"]["what is upi scam"]["response"]
            if "how to protect" in lower or "how to avoid" in lower:
                return cls.KNOWLEDGE_RESPONSES["fraud_prevention"]["how to protect from upi scams"]["response"]

        # Job scam queries
        if "job" in lower or "employment" in lower or "hiring" in lower or "employment" in lower or "career" in lower:
            if "scam" in lower or "fraud" in lower:
                return cls.KNOWLEDGE_RESPONSES["job_scam"]["what is job scam"]["response"]
            if "how to avoid" in lower or "protect" in lower:
                return cls.KNOWLEDGE_RESPONSES["fraud_prevention"]["how to avoid scams"]["response"]

        # SMS/Phishing queries
        if "sms" in lower or "text" in lower or "phishing" in lower or "smishing" in lower:
            return cls.KNOWLEDGE_RESPONSES["sms_phishing"]["what is sms phishing"]["response"]

        # Protection/Prevention queries
        if "protect" in lower or "prevent" in lower or "avoid" in lower or "safety" in lower:
            if "upi" in lower or "payment" in lower:
                return cls.KNOWLEDGE_RESPONSES["fraud_prevention"]["how to protect from upi scams"]["response"]
            return cls.KNOWLEDGE_RESPONSES["fraud_prevention"]["how to avoid scams"]["response"]

        # Common fraud types
        if "types of scams" in lower or "types of fraud" in lower or "common scams" in lower or "kinds of scams" in lower:
            return cls.KNOWLEDGE_RESPONSES["fraud_detection"]["types of scams"]["response"]
        if "types of fraud" in lower or "common fraud" in lower:
            return cls.KNOWLEDGE_RESPONSES["fraud_detection"]["what are common fraud types"]["response"]

        # Model-related queries
        if "model" in lower and ("performance" in lower or "accuracy" in lower or "metrics" in lower):
            return cls.KNOWLEDGE_RESPONSES["model_insights"]["model performance"]["response"]
        if "model" in lower and ("work" in lower or "how" in lower or "explain" in lower):
            return cls.KNOWLEDGE_RESPONSES["model_insights"]["how models work"]["response"]

        # Compliance queries
        if "compliance" in lower or "compliant" in lower or "rbi" in lower or "regulation" in lower:
            return cls.KNOWLEDGE_RESPONSES["compliance"]["what are compliance requirements"]["response"]

        # Identification queries
        if "identify" in lower or "identify scam" in lower or "identify fraud" in lower:
            return cls.KNOWLEDGE_RESPONSES["identification"]["how to identify fraud"]["response"]

        # SMS phishing detection
        if "sms phishing" in lower or "text phishing" in lower:
            return cls.KNOWLEDGE_RESPONSES["sms_detection"]["is this sms phishing"]["response"]

        # Default response
        return (
            "I can help you with fraud detection, scam education, and security analysis. "
            "Ask me about:\n"
            "• UPI fraud prevention and common scams\n"
            "• SMS/phishing scam detection\n"
            "• Job scam identification\n"
            "• Transaction analysis and fraud patterns\n"
            "• How fraud detection models work\n"
            "• Compliance requirements (RBI, PCI DSS, etc.)\n"
            "• How to identify and avoid fraud\n\n"
            "What specific fraud or security question can I help you with?"
        )


class AIAssistant:
    """Main AI assistant for fraud detection support."""

    def __init__(
        self,
        rag: Optional[Any] = None,
        decision_engine: Optional[Any] = None,
        memory: Optional[ConversationMemory] = None,
    ):
        self.rag = rag
        self.decision_engine = decision_engine
        self.memory = memory or ConversationMemory()
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()

    async def process_request(
        self,
        messages: List[Dict],
        session_id: Optional[str] = None,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """Process AI chat request and generate response."""
        # Generate or use provided session ID
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        # Get the last user message
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return {
                "response": "Please provide a message to analyze.",
                "confidence": 0.0,
                "session_id": session_id,
            }

        last_message = user_messages[-1].get("content", "")

        # Store in memory
        self.memory.add_message(session_id, "user", last_message)

        # Classify intent
        intent = self.intent_classifier.classify(last_message)

        # Generate response based on intent
        response_data = self.response_generator.get_response(intent, last_message)

        # If RAG is available, augment with knowledge base
        ragged_sources = []
        if self.rag and self.rag._initialized:
            try:
                rag_results = self.rag.search(query=last_message, top_k=3)
                if rag_results:
                    for result in rag_results:
                        ragged_sources.append({
                            "title": result.chunk.title,
                            "content": result.chunk.content[:500],
                            "category": result.chunk.category,
                            "score": round(result.score, 4),
                        })
            except Exception:
                pass  # Continue without RAG if it fails

        # Build final response
        final_response = response_data["response"]

        # Add sources
        sources = []
        if include_sources:
            sources = response_data.get("sources", [])
            if ragged_sources:
                sources.extend(ragged_sources)

        # Store assistant response in memory
        self.memory.add_message(session_id, "assistant", final_response)

        return {
            "response": final_response,
            "sources": sources if sources else None,
            "confidence": response_data.get("confidence", 0.5),
            "session_id": session_id,
        }

    async def analyze_transaction(
        self,
        transaction_data: Dict,
    ) -> Dict[str, Any]:
        """Analyze a transaction for fraud."""
        if not self.decision_engine:
            return {"error": "Decision engine not available"}

        from upishield_core import UPITransaction, ScoreRequest

        # Create transaction object
        txn = UPITransaction(**transaction_data)

        # Score transaction
        request = ScoreRequest(
            transaction=txn,
            include_explanation=True,
            include_features=False,
        )

        context = await self.decision_engine.score(request.transaction)

        return {
            "decision": context.decision.value,
            "risk_score": context.final_score,
            "risk_level": context.risk_level.value,
            "confidence": context.ml_confidence,
            "explanation": context.explanation.model_dump() if context.explanation else None,
            "recommendation": self._get_recommendation(context),
        }

    def _get_recommendation(self, context: Any) -> str:
        """Get actionable recommendation from decision context."""
        if context.decision.value == "DECLINE":
            return "Do not proceed with this transaction. Potential fraud detected."
        elif context.decision.value == "REVIEW":
            return "Review this transaction carefully. Higher than normal risk detected."
        else:
            return "Transaction appears safe. Proceed with standard security precautions."