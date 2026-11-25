"""
RAG (Retrieval-Augmented Generation) Service for Fraud Knowledge Base.

Provides semantic search over fraud detection knowledge with vector embeddings.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    SentenceTransformer = None
    faiss = None


class KnowledgeChunk(BaseModel):
    """A chunk of knowledge from the fraud knowledge base."""
    id: str
    title: str
    content: str
    category: str
    tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None


class RAGQuery(BaseModel):
    """Query request for RAG search."""
    query: str
    top_k: int = 5
    category_filter: Optional[str] = None
    min_score: float = 0.3


class RAGResult(BaseModel):
    """Single RAG search result."""
    chunk: KnowledgeChunk
    score: float


class RAGResponse(BaseModel):
    """RAG search response."""
    results: List[RAGResult]
    query: str
    total_chunks_searched: int


class FraudKnowledgeBase:
    """Fraud detection knowledge base with vector search."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunks: List[KnowledgeChunk] = []
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the encoder and load/build index."""
        if not RAG_AVAILABLE:
            return False

        try:
            self.encoder = SentenceTransformer(self.model_name)
            self._load_knowledge_base()
            self._build_index()
            self._initialized = True
            return True
        except Exception as e:
            print(f"RAG initialization failed: {e}")
            return False

    def _load_knowledge_base(self):
        """Load fraud knowledge from JSON files or build default."""
        kb_path = Path(__file__).parent.parent / "data" / "fraud_knowledge.json"

        if kb_path.exists():
            with open(kb_path, 'r') as f:
                data = json.load(f)
                self.chunks = [KnowledgeChunk(**c) for c in data]
        else:
            # Build default knowledge base
            self.chunks = self._build_default_knowledge()
            # Save for future use
            kb_path.parent.mkdir(parents=True, exist_ok=True)
            with open(kb_path, 'w') as f:
                json.dump([c.model_dump() for c in self.chunks], f, indent=2)

    def _build_default_knowledge(self) -> List[KnowledgeChunk]:
        """Build comprehensive default fraud knowledge base."""
        return [
            # UPI Fraud Knowledge
            KnowledgeChunk(
                id="upi_fraud_overview",
                title="UPI Fraud Overview",
                content=(
                    "UPI (Unified Payments Interface) fraud refers to unauthorized or deceptive transactions "
                    "conducted through India's UPI payment system. Fraudsters exploit the instant, real-time "
                    "nature of UPI to trick users into authorizing payments. Key characteristic: transactions "
                    "are irrevocable once authorized with UPI PIN."
                ),
                category="upi_fraud",
                tags=["overview", "definition", "upi", "fraud"]
            ),
            KnowledgeChunk(
                id="upi_collect_request_fraud",
                title="Collect Request Fraud (Most Common UPI Scam)",
                content=(
                    "Fraudster sends a 'Request Money' (collect request) instead of 'Pay'. User thinks "
                    "they're receiving money, clicks 'Approve', but actually authorizes money OUT of "
                    "their account. Often disguised as: refunds, cashback, KYC verification, prize winnings, "
                    "or delivery fees. RED FLAG: Never approve collect requests you didn't initiate or expect."
                ),
                category="upi_fraud",
                tags=["collect_request", "pull_payment", "most_common", "red_flags"]
            ),
            KnowledgeChunk(
                id="upi_qr_code_scams",
                title="QR Code Scams",
                content=(
                    "'Scan this QR to receive ₹5000 cashback' - QR actually contains a collect request "
                    "(pull money). User scans, enters PIN, money debited instead of credited. RULE: Only "
                    "scan QR codes to PAY, never to receive money. Legitimate merchants never ask you to "
                    "scan QR to receive payments."
                ),
                category="upi_fraud",
                tags=["qr_code", "cashback_scam", "pull_payment", "red_flags"]
            ),
            KnowledgeChunk(
                id="upi_kyc_phishing",
                title="KYC/Bank Impersonation Phishing",
                content=(
                    "'Your UPI/KYC will expire, click link to update' - Link leads to phishing page "
                    "capturing UPI PIN, OTP, credentials. Banks NEVER send KYC links via SMS/WhatsApp. "
                    "Official bank communication comes from verified shortcodes (e.g., AXISBK, ICICIB), "
                    "not 10-digit numbers."
                ),
                category="upi_fraud",
                tags=["kyc", "phishing", "impersonation", "red_flags"]
            ),
            KnowledgeChunk(
                id="upi_fake_support",
                title="Fake Customer Support Scams",
                content=(
                    "Caller claims to be from bank/Google Pay/PhonePe/Paytm. Asks for UPI PIN, OTP, or to "
                    "install screen-sharing app (AnyDesk/TeamViewer/QuickSupport). LEGITIMATE SUPPORT "
                    "NEVER ASKS FOR PIN/OTP. Real support will never ask you to install remote access apps."
                ),
                category="upi_fraud",
                tags=["fake_support", "remote_access", "pin_theft", "red_flags"]
            ),
            KnowledgeChunk(
                id="upi_protection_tips",
                title="UPI Fraud Protection Tips",
                content=(
                    "1. Never share UPI PIN/OTP with anyone, ever. 2. Only approve collect requests you "
                    "initiated and expect. 3. Verify merchant name before clicking 'Pay'. 4. Use ScamShield "
                    "Live Scoring to check suspicious transactions. 5. Report fraud immediately: 1930 "
                    "(cybercrime helpline) or your bank. 6. Enable transaction alerts in banking app. "
                    "7. Set UPI transaction limits. 8. Don't scan QR codes from unknown sources."
                ),
                category="upi_fraud",
                tags=["protection", "prevention", "tips", "best_practices"]
            ),

            # Job Scam Knowledge
            KnowledgeChunk(
                id="job_scam_overview",
                title="Job Scam Overview",
                content=(
                    "Job scams trick job seekers into paying money or sharing personal data for fake "
                    "employment opportunities. Target freshers and desperate job seekers. Scammers exploit "
                    "trust in reputed company names and urgency of finding employment."
                ),
                category="job_scam",
                tags=["overview", "definition", "target_audience"]
            ),
            KnowledgeChunk(
                id="job_scam_upfront_fees",
                title="Upfront Fee Job Scams",
                content=(
                    "'Registration fee: ₹2,999' for job portal access, 'Training material cost', "
                    "'Background verification charges', 'Security deposit'. Legitimate employers NEVER "
                    "ask candidates to pay money to get hired. Any job requiring upfront payment is a scam."
                ),
                category="job_scam",
                tags=["upfront_fees", "registration_fee", "training_cost", "red_flags"]
            ),
            KnowledgeChunk(
                id="job_scam_reshipping_mule",
                title="Reshipping / Money Mule Job Scams",
                content=(
                    "'Receive packages, re-ship them' - actually handling stolen goods purchased with "
                    "stolen credit cards. 'Process payments for international clients' - money laundering. "
                    "Victims become unwitting accomplices in crime. Legal consequences possible."
                ),
                category="job_scam",
                tags=["reshipping", "money_mule", "money_laundering", "legal_risk"]
            ),
            KnowledgeChunk(
                id="job_scam_fake_training",
                title="Fake Training/Certification Scams",
                content=(
                    "'Pay ₹15,000 for AWS/Data Science certification + guaranteed job'. Certificates are "
                    "worthless, jobs don't exist. Often use fake logos of Coursera, edX, or company names. "
                    "Real certifications don't guarantee jobs; legitimate platforms don't promise employment."
                ),
                category="job_scam",
                tags=["fake_training", "certification", "guaranteed_job", "red_flags"]
            ),
            KnowledgeChunk(
                id="job_scam_impersonation",
                title="Company Impersonation Job Scams",
                content=(
                    "Fake offer letters from TCS, Infosys, Google, Amazon. HR email from @gmail.com, "
                    "@yahoo.com instead of company domain (@tcs.com, @infosys.com). Check: official "
                    "careers page, LinkedIn verification, domain email, no upfront fees."
                ),
                category="job_scam",
                tags=["impersonation", "fake_offer_letter", "email_verification", "red_flags"]
            ),

            # SMS Phishing (Smishing) Knowledge
            KnowledgeChunk(
                id="smishing_overview",
                title="SMS Phishing (Smishing) Overview",
                content=(
                    "Smishing = SMS + Phishing. Fraudulent text messages designed to steal credentials, "
                    "money, or install malware. Exploits urgency and trust in official-looking messages. "
                    "India sees millions of smishing attempts daily."
                ),
                category="smishing",
                tags=["overview", "definition", "sms", "phishing"]
            ),
            KnowledgeChunk(
                id="smishing_bank_alert_spoofing",
                title="Bank Alert Spoofing",
                content=(
                    "'A/c XXXX debited ₹15,000. If not you, click: bit.ly/fake-bank'. "
                    "'Your KYC expired. Update now: short.url/phishing'. "
                    "'UPI transaction failed. Retry: malicious-link.com'. "
                    "Official bank alerts come from verified shortcodes, not 10-digit numbers. "
                    "Banks never include links in SMS for security actions."
                ),
                category="smishing",
                tags=["bank_spoofing", "kyc_expiry", "shortened_urls", "red_flags"]
            ),
            KnowledgeChunk(
                id="smishing_delivery_scams",
                title="Delivery/Customs Scams",
                content=(
                    "'Your package held at customs. Pay ₹2,500: link'. "
                    "'Delivery failed. Reschedule: tracking-fake.com'. "
                    "Legitimate couriers (Blue Dart, Delhivery, India Post) don't ask for payment via "
                    "SMS links. Check tracking on official website/app."
                ),
                category="smishing",
                tags=["delivery", "customs", "courier", "red_flags"]
            ),
            KnowledgeChunk(
                id="smishing_prize_lottery",
                title="Prize/Lottery Scams",
                content=(
                    "'Congratulations! You won ₹10 Lakhs. Claim: claim-prize.xyz'. "
                    "'KBC/Jio/Airtel lucky draw winner. Contact: +91-XXXXXXXXXX'. "
                    "You can't win a lottery you didn't enter. Legitimate companies don't notify winners "
                    "via SMS from unknown numbers asking for money to claim prizes."
                ),
                category="smishing",
                tags=["prize", "lottery", "kbc", "fake_winnings", "red_flags"]
            ),
            KnowledgeChunk(
                id="smishing_authority_impersonation",
                title="Authority Impersonation (Tax/Police/Govt)",
                content=(
                    "'Income Tax refund ₹45,000. Verify: itr-refund.fake'. "
                    "'Police case filed against you. Call: 98765XXXXX'. "
                    "Government agencies don't communicate via SMS with links or ask for money to "
                    "resolve cases. Official notices come by post or official portal."
                ),
                category="smishing",
                tags=["tax", "police", "government", "authority_impersonation", "red_flags"]
            ),
            KnowledgeChunk(
                id="smishing_identification",
                title="How to Identify Smishing",
                content=(
                    "1. Sender: Unknown number, not official shortcode (e.g., not 'AXISBK', 'ICICIB', "
                    "'HDFCBK'). 2. URL: Shortened (bit.ly, tinyurl, t.co) or suspicious domains. 3. "
                    "Urgency: 'Immediate action required', 'within 30 minutes', 'account will be closed'. "
                    "4. Generic: 'Dear Customer' instead of your name. 5. Asks for: OTP, PIN, password, "
                    "card details, CVV. 6. Poor grammar/spelling in message."
                ),
                category="smishing",
                tags=["identification", "red_flags", "sender_verification", "url_analysis"]
            ),
            KnowledgeChunk(
                id="smishing_protection",
                title="Smishing Protection",
                content=(
                    "1. Never click links in unsolicited SMS. 2. Verify independently: Call bank on "
                    "official number (from card/website, not SMS). 3. Report: Forward to 1930 or use "
                    "'Report Spam' in messaging app. 4. Enable spam filtering in messaging app. 5. Don't "
                    "call numbers provided in suspicious SMS. 6. Check official app/website for alerts."
                ),
                category="smishing",
                tags=["protection", "prevention", "reporting", "best_practices"]
            ),

            # ScamShield Technical Knowledge
            KnowledgeChunk(
                id="scamshield_architecture",
                title="ScamShield Architecture Overview",
                content=(
                    "ScamShield uses a multi-layered fraud detection pipeline: "
                    "1. Feature Engineering: 50+ features computed from transaction (velocity, device, "
                    "network, amount patterns, user behavior). 2. ML Ensemble: FraudMLP (UPI), TextCNN "
                    "(SMS), Transformer (Job Scam) - all ONNX INT8 quantized for <50ms latency. "
                    "3. Rules Engine: Velocity (>5/min), new device + high amount, impossible travel, "
                    "high-risk MCC, known bad entities. 4. Threat Intel: Known bad VPAs, IPs, devices, "
                    "phone numbers from threat feeds. 5. DecisionEngine: ML (55%) + Rules (25%) + "
                    "Threat Intel (20%) weighted combination. Outputs: APPROVE/REVIEW/DECLINE with "
                    "risk score, confidence, and explanation."
                ),
                category="technical",
                tags=["architecture", "pipeline", "fraudmlp", "textcnn", "decision_engine", "latency"]
            ),
            KnowledgeChunk(
                id="scamshield_models",
                title="ScamShield ML Models",
                content=(
                    "Three specialized models: "
                    "1. FraudMLP (UPI): 50 features, feed-forward network, detects transaction anomalies. "
                    "2. Smishing CNN (SMS): TextCNN + BERT embeddings, analyzes sender, URLs, linguistic "
                    "features, urgency signals. 3. Job Scam Transformer: BERT-based, analyzes job posting "
                    "text for fraud patterns. All models exported to ONNX with INT8 quantization for "
                    "sub-50ms inference. Fallback: sklearn GradientBoostingClassifier if ONNX unavailable. "
                    "Deterministic hash fallback for edge cases."
                ),
                category="technical",
                tags=["models", "fraudmlp", "smishing_cnn", "job_scam_transformer", "onnx", "int8"]
            ),
            KnowledgeChunk(
                id="scamshield_scoring_api",
                title="ScamShield Scoring API",
                content=(
                    "POST /api/v1/score - Real-time transaction scoring. Request: transaction object with "
                    "txn_id, amount, payer_vpa, payee_vpa, merchant_category, device_id, ip, location, "
                    "velocity features, time features. Response: decision (APPROVE/REVIEW/DECLINE), "
                    "risk_level (LOW/MEDIUM/HIGH/CRITICAL), risk_score (0-1), confidence (0-1), "
                    "processing_time_ms, explanation with score_breakdown, model_scores, top_factors, "
                    "triggered_rules, threat_intel_summary, feature_importance, recommendation. "
                    "Target latency: <50ms p99. Batch endpoint: POST /api/v1/score/batch."
                ),
                category="technical",
                tags=["api", "scoring", "endpoint", "latency", "response_format"]
            ),
            KnowledgeChunk(
                id="scamshield_alert_rules",
                title="ScamShield Alert Rules",
                content=(
                    "Alert rules define conditions for triggering notifications. Rule structure: name, "
                    "condition (SQL-like expression on transaction features), severity (critical/high/medium/low), "
                    "enabled flag. Example conditions: 'risk_score > 0.8 AND amount > 50000', "
                    "'velocity_5m > 10', 'is_new_device AND amount > avg_amount_7d * 3'. "
                    "Rules evaluated in real-time during scoring. Notifications sent via configured "
                    "channels: email, webhook, Slack, SMS. Rules managed via /api/v1/alerts/rules."
                ),
                category="technical",
                tags=["alert_rules", "conditions", "severity", "notifications", "real_time"]
            ),
            KnowledgeChunk(
                id="scamshield_investigation",
                title="Fraud Investigation Workflow",
                content=(
                    "1. Alert received → 2. Triage: check risk_score, rules triggered, threat intel → "
                    "3. Create case: link related alerts, transactions → 4. Investigate: analyze patterns, "
                    "check user history, device fingerprint, network → 5. Decision: confirm fraud / false "
                    "positive → 6. Action: block user/device/IP, update rules, add to threat intel → "
                    "7. Close case with outcome. Best practices: SLA-based triage, evidence preservation, "
                    "cross-team collaboration, regular rule tuning based on investigation outcomes."
                ),
                category="technical",
                tags=["investigation", "workflow", "triage", "case_management", "best_practices"]
            ),
            KnowledgeChunk(
                id="scamshield_compliance",
                title="Compliance & Regulatory",
                content=(
                    "ScamShield helps meet: RBI Master Direction - Digital Payment Security Controls, "
                    "NPCI UPI Procedural Guidelines, PCI DSS Requirement 12.10 (Incident Response), "
                    "GDPR Article 33 (Breach Notification - 72hr). Audit-ready reports: Incident Register, "
                    "Risk Metrics (fraud rates, detection accuracy, false positive trends), Control "
                    "Effectiveness (rule performance, model drift indicators). Export: PDF, CSV, JSON."
                ),
                category="technical",
                tags=["compliance", "rbi", "npci", "pci_dss", "gdpr", "audit", "reports"]
            ),

            # General Fraud Prevention
            KnowledgeChunk(
                id="fraud_prevention_general",
                title="General Digital Payment Fraud Prevention",
                content=(
                    "Universal protection principles: 1. VERIFY independently - never trust unsolicited "
                    "communications. 2. NEVER share OTP, PIN, passwords, CVV. 3. CHECK sender - official "
                    "entities use verified shortcodes/domains. 4. THINK before clicking - urgency is a "
                    "manipulation tactic. 5. REPORT immediately - 1930 (cybercrime), bank, platform. "
                    "6. MONITOR accounts - enable alerts, review statements. 7. LIMIT exposure - set "
                    "transaction limits, use virtual cards. 8. EDUCATE others - share awareness."
                ),
                category="prevention",
                tags=["general", "best_practices", "awareness", "reporting"]
            ),
            KnowledgeChunk(
                id="fraud_reporting_india",
                title="Fraud Reporting in India",
                content=(
                    "1. National Cyber Crime Reporting Portal: cybercrime.gov.in (24/7) "
                    "2. Cyber Crime Helpline: 1930 (toll-free) "
                    "3. Bank's customer care (on back of card/official website) "
                    "4. NPCI UPI Help: upihelp.npci.org.in "
                    "5. Platform reporting: Google Pay/PhonePe/Paytm in-app help "
                    "6. Police: Local cyber crime police station "
                    "Key info to provide: Transaction ID, date/time, amount, VPA, screenshots, "
                    "communication records, any caller details."
                ),
                category="prevention",
                tags=["reporting", "india", "1930", "cybercrime", "npci", "police"]
            ),
        ]

    def _build_index(self):
        """Build FAISS index from chunk embeddings."""
        if not self.chunks or not self.encoder:
            return

        # Generate embeddings for chunks without them
        texts = [f"{c.title}. {c.content}" for c in self.chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized vectors
        self.index.add(embeddings.astype(np.float32))

        # Store embeddings in chunks
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i].tolist()

    def search(self, query: str, top_k: int = 5, category_filter: Optional[str] = None,
               min_score: float = 0.3) -> List[RAGResult]:
        """Search knowledge base for relevant chunks."""
        if not self._initialized or not self.index or not self.encoder:
            return []

        # Encode query
        query_embedding = self.encoder.encode([query], show_progress_bar=False)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k * 3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            # Apply category filter
            if category_filter and chunk.category != category_filter:
                continue

            # Apply minimum score
            if score < min_score:
                continue

            results.append(RAGResult(chunk=chunk, score=float(score)))

            if len(results) >= top_k:
                break

        return results


# Global RAG instance
_rag_instance: Optional[FraudKnowledgeBase] = None


def get_rag() -> FraudKnowledgeBase:
    """Get or create global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = FraudKnowledgeBase()
        _rag_instance.initialize()
    return _rag_instance