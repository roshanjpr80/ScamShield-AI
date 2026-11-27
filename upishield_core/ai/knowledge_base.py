"""
RAG Knowledge Base for Fraud Detection.

Implements vector storage and retrieval using ChromaDB with sentence-transformers
for semantic search over fraud patterns, regulations, case studies, and prevention tips.
"""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ============================================================
# Data Models
# ============================================================

class KnowledgeCategory(str, Enum):
    """Categories of fraud knowledge."""
    UPI_FRAUD = "upi_fraud"
    SMS_PHISHING = "sms_phishing"
    JOB_SCAM = "job_scam"
    DOCUMENT_FRAUD = "document_fraud"
    REGULATIONS = "regulations"
    PREVENTION = "prevention"
    CASE_STUDIES = "case_studies"
    IOC = "ioc"  # Indicators of Compromise
    GENERAL = "general"


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge from the fraud knowledge base."""
    id: str
    title: str
    content: str
    category: KnowledgeCategory
    tags: List[str] = field(default_factory=list)
    source: str = "internal"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category.value,
            "tags": self.tags,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Result from knowledge base search."""
    chunk: KnowledgeChunk
    score: float
    distance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "distance": self.distance,
        }


# Embedding Provider

class EmbeddingProvider:
    """Provides text embeddings using sentence-transformers."""

    _instance: Optional["EmbeddingProvider"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingProvider":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_name = model_name
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if self._model is None:
            self._model_name = model_name
            self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded embedding model: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model = None

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        if self._model is None:
            # Fallback: random embeddings for testing
            if isinstance(texts, str):
                texts = [texts]
            return np.random.rand(len(texts), 384).astype(np.float32)

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.embed(text)[0]

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return 384  # all-MiniLM-L6-v2 dimension

    @classmethod
    def get_instance(cls) -> "EmbeddingProvider":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ============================================================
# Vector Store (ChromaDB)
# ============================================================

class VectorStore:
    """ChromaDB vector store for fraud knowledge base."""

    def __init__(
        self,
        collection_name: str = "fraud_knowledge",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./data/chroma"
        self._client: Optional[chromadb.Client] = None
        self._collection = None
        self._embedding_provider = EmbeddingProvider.get_instance()

    def _get_client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        return self._client

    def _get_collection(self):
        """Get or create collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Fraud detection knowledge base"},
            )
        return self._collection

    def initialize(self) -> bool:
        """Initialize the vector store."""
        try:
            self._get_collection()
            logger.info(f"Vector store initialized: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False

    def add_chunks(self, chunks: List[KnowledgeChunk]) -> List[str]:
        """Add knowledge chunks to the vector store."""
        if not chunks:
            return []

        collection = self._get_collection()

        # Prepare data
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "title": chunk.title,
                "category": chunk.category.value,
                "tags": json.dumps(chunk.tags),
                "source": chunk.source,
                "created_at": chunk.created_at.isoformat(),
                "updated_at": chunk.updated_at.isoformat(),
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        # Generate embeddings
        embeddings = self._embedding_provider.embed(documents).tolist()

        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks to vector store")
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[KnowledgeCategory] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Search for relevant chunks."""
        collection = self._get_collection()

        # Generate query embedding
        query_embedding = self._embedding_provider.embed_single(query).tolist()

        # Build where clause for category filter
        where = {}
        if category_filter:
            where["category"] = category_filter.value

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        # Parse results
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert distance to similarity score (1 - distance for cosine)
                score = max(0.0, 1.0 - distance)

                if score < min_score:
                    continue

                metadata = results["metadatas"][0][i]
                chunk = KnowledgeChunk(
                    id=chunk_id,
                    title=metadata.get("title", ""),
                    content=results["documents"][0][i],
                    category=KnowledgeCategory(metadata.get("category", "general")),
                    tags=json.loads(metadata.get("tags", "[]")),
                    source=metadata.get("source", "internal"),
                    created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                    updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.utcnow().isoformat())),
                    metadata={k: v for k, v in metadata.items()
                             if k not in ["title", "category", "tags", "source", "created_at", "updated_at"]},
                )
                search_results.append(SearchResult(chunk=chunk, score=score, distance=distance))

        return search_results

    def get_chunk(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        """Get a specific chunk by ID."""
        collection = self._get_collection()
        results = collection.get(ids=[chunk_id], include=["documents", "metadatas"])

        if results["ids"] and results["ids"][0]:
            metadata = results["metadatas"][0]
            return KnowledgeChunk(
                id=results["ids"][0],
                title=metadata.get("title", ""),
                content=results["documents"][0],
                category=KnowledgeCategory(metadata.get("category", "general")),
                tags=json.loads(metadata.get("tags", "[]")),
                source=metadata.get("source", "internal"),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.utcnow().isoformat())),
            )
        return None

    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """Delete chunks by IDs."""
        collection = self._get_collection()
        collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    def count(self) -> int:
        """Get total chunk count."""
        collection = self._get_collection()
        return collection.count()

    def clear(self):
        """Clear all chunks."""
        collection = self._get_collection()
        client = self._get_client()
        client.delete_collection(self.collection_name)
        self._collection = None
        self._get_collection()  # Recreate empty collection
        logger.info("Vector store cleared")


# Knowledge Base Manager

class FraudKnowledgeBase:
    """Main knowledge base manager for fraud detection."""

    _instance: Optional["FraudKnowledgeBase"] = None

    def __new__(cls) -> "FraudKnowledgeBase":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        collection_name: str = "fraud_knowledge",
        persist_directory: Optional[str] = None,
    ):
        if not hasattr(self, "initialized"):
            self.vector_store = VectorStore(collection_name, persist_directory)
            self._initialized = False
            self._chunks: Dict[str, KnowledgeChunk] = {}
            self.initialized = True

    def initialize(self, seed_data: bool = True) -> bool:
        """Initialize the knowledge base with optional seed data."""
        success = self.vector_store.initialize()
        if success and seed_data:
            self._load_seed_data()
        self._initialized = success
        return success

    def _load_seed_data(self):
        """Load seed fraud knowledge data."""
        seed_chunks = self._create_seed_chunks()
        self.vector_store.add_chunks(seed_chunks)
        for chunk in seed_chunks:
            self._chunks[chunk.id] = chunk
        logger.info(f"Loaded {len(seed_chunks)} seed knowledge chunks")

    def _create_seed_chunks(self) -> List[KnowledgeChunk]:
        """Create comprehensive seed knowledge for fraud detection."""
        chunks = []

        # UPI Fraud Patterns
        chunks.extend([
            KnowledgeChunk(
                id="upi_001",
                title="UPI Collect Request Scam - Most Common Fraud",
                content="""The UPI Collect Request scam is the most prevalent UPI fraud in India.

How it works:
1. Fraudster sends a 'Request Money' (collect request) instead of 'Pay'
2. User thinks they're RECEIVING money but actually AUTHORIZING money OUT
3. Common pretexts: 'I'm sending you money', 'Cashback received', 'Refund approved', 'Prize money'

Red Flags:
- You receive a collect request from unknown VPA
- Message says 'money will be credited' but it's a request to pay
- Urgency pressure: 'approve within 5 minutes'
- Sender asks you to enter UPI PIN to 'receive' money

Protection:
- NEVER approve collect requests you didn't initiate
- Always verify the DIRECTION of payment before entering PIN
- Legitimate refunds come as CREDITS (push), not collect requests (pull)
- Use official app notifications, not SMS links""",
                category=KnowledgeCategory.UPI_FRAUD,
                tags=["collect_request", "social_engineering", "most_common", "upi_pin"],
                source="upi_fraud_research",
            ),
            KnowledgeChunk(
                id="upi_002",
                title="QR Code Payment Scams",
                content="""QR Code scams exploit the scan-to-pay feature.

Types:
1. **Reverse QR**: Fraudster shows QR saying 'Scan to receive ₹5000' - actually a payment TO them
2. **Tampered QR**: Legitimate merchant QR replaced with fraudster's QR
3. **Super QR**: Malicious QR on websites/ads that redirects to phishing

How it works:
- User scans QR code from unknown source
- App shows 'Pay ₹X to Merchant' but user thinks 'Receive'
- User enters UPI PIN authorizing payment OUT

Protection:
- ALWAYS check payment direction on UPI app screen
- Verify payee VPA matches expected merchant
- Don't scan QR codes from untrusted sources (WhatsApp, random websites)
- Use merchant's official payment link/App instead of physical QR
- If receiving money, the other person should scan YOUR QR or use your VPA directly""",
                category=KnowledgeCategory.UPI_FRAUD,
                tags=["qr_code", "reverse_qr", "payment_direction", "scan"],
                source="upi_fraud_research",
            ),
            KnowledgeChunk(
                id="upi_003",
                title="Fake UPI Customer Support Scams",
                content="""Impersonation of Google Pay, PhonePe, Paytm, BHIM support.

Common Tactics:
1. Caller claims 'suspicious activity' on your UPI account
2. Asks you to install remote access app (TeamViewer, AnyDesk, QuickSupport)
3. Requests UPI PIN, OTP, or screen sharing 'for verification'
4. May send fake SMS from 'official' number with verification link

Real vs Fake Support:
- **REAL**: Never asks for PIN, OTP, or remote access. Uses in-app chat only.
- **FAKE**: Asks for PIN/OTP, wants remote access, pressures urgency, uses personal numbers

Protection:
- Official support NEVER asks for UPI PIN or OTP
- Never install remote access apps for 'support'
- Use in-app help/support only
- If unsure, call official customer care from app/website
- Report suspicious calls to bank immediately""",
                category=KnowledgeCategory.UPI_FRAUD,
                tags=["customer_support", "impersonation", "remote_access", "upi_pin"],
                source="upi_fraud_research",
            ),
            KnowledgeChunk(
                id="upi_004",
                title="UPI SIM Swap / Account Takeover",
                content="""SIM swap enables complete account takeover.

Attack Flow:
1. Fraudster gathers victim's personal info (phishing, social media, data breaches)
2. Contacts mobile operator pretending to be victim
3. Gets new SIM issued with victim's number
4. Receives all OTPs for UPI/banking
5. Resets UPI PIN, adds new device, drains account

Warning Signs:
- Sudden loss of mobile network/service
- Unexpected 'SIM change' SMS from operator
- Unable to receive OTPs
- Unauthorized UPI transactions

Protection:
- Enable SIM lock/PIN with mobile operator
- Use biometric/device PIN for UPI apps (not just OTP)
- Monitor mobile service - report outages immediately
- Register for 'SIM change alerts' with operator if available
- Use authenticator apps (Google Authenticator) instead of SMS OTP where possible
- Set up UPI transaction alerts on email too""",
                category=KnowledgeCategory.UPI_FRAUD,
                tags=["sim_swap", "account_takeover", "otp_interception", "mobile_security"],
                source="upi_fraud_research",
            ),

            # SMS Phishing / Smishing
            KnowledgeChunk(
                id="sms_001",
                title="Bank Alert Spoofing Smishing",
                content="""Most common SMS phishing tactic - fake bank transaction alerts.

Template:
'BANK ALERT: Your A/c XXXXXX debited ₹15,000. If not you, click: bit.ly/xyz to block'

Why it works:
- Creates panic with 'unauthorized transaction'
- Shortened URL hides malicious destination
- Uses partial account number for credibility
- Urgent language demands immediate action

Red Flags:
- Banks NEVER send transaction links via SMS
- Official SMS uses sender ID (e.g., 'ICICIBK') not random numbers
- Links use URL shorteners (bit.ly, tinyurl) or suspicious domains
- Asks to 'verify', 'confirm', 'update', 'block' via link
- Poor grammar/spelling in sophisticated attacks too

Protection:
- NEVER click links in transaction SMS
- Check transactions in official banking app
- Call bank on OFFICIAL number (from card/website)
- Forward suspicious SMS to 1930 (India's cybercrime helpline)
- Enable transaction alerts in banking app (not just SMS)""",
                category=KnowledgeCategory.SMS_PHISHING,
                tags=["bank_spoofing", "transaction_alert", "url_shortener", "urgency"],
                source="smishing_research",
            ),
            KnowledgeChunk(
                id="sms_002",
                title="KYC Expiry / Verification Phishing",
                content="""Exploits mandatory KYC requirements for bank/UPI accounts.

Template:
'RBI/UPI/NPCI: Your KYC expired. Update now to avoid account block: link'
'Dear Customer, Complete KYC verification: link - Team PhonePe/Google Pay'

Why it works:
- KYC is real requirement (RBI mandate)
- Fear of account blocking creates urgency
- Uses official-sounding sender names
- Links to convincing phishing pages

Red Flags:
- KYC updates done in BANK APP/BRANCH, not SMS links
- RBI/NPCI never send individual KYC links
- Legitimate apps show in-app KYC banners, not SMS
- Links to domains like 'kyc-verify.xyz', not official domains

Protection:
- Complete KYC only in official banking/UPI app
- Check app for KYC status/banner
- Visit bank branch for document verification
- Report phishing to bank and 1930""",
                category=KnowledgeCategory.SMS_PHISHING,
                tags=["kyc", "rbi_impersonation", "account_block_threat", "document_phishing"],
                source="smishing_research",
            ),
            KnowledgeChunk(
                id="sms_003",
                title="Delivery / Parcel Scam Smishing",
                content="""Fake delivery notifications demanding payment.

Template:
'India Post: Your parcel held at customs. Pay ₹2,500 to release: link'
'BlueDart: COD amount ₹1,800 pending. Pay: link'
'Customs: International package requires duty payment: link'

Why it works:
- Legitimate courier communications exist
- Small amounts seem reasonable for 'customs/duty'
- Tracking numbers sometimes included for credibility
- Time pressure: 'parcel will be returned'

Red Flags:
- Legitimate couriers collect COD at DOOR, not via SMS links
- India Post doesn't demand payment via SMS for customs
- International customs handled by courier, not random links
- COD amount usually known at order time

Protection:
- Track parcels on OFFICIAL courier website/app
- Pay COD only to delivery person at door
- Contact courier on official customer care
- Never pay 'customs duty' via SMS link""",
                category=KnowledgeCategory.SMS_PHISHING,
                tags=["delivery_scam", "parcel", "customs_duty", "cod"],
                source="smishing_research",
            ),

            # Job Scams
            KnowledgeChunk(
                id="job_001",
                title="Upfront Fee Job Scam - Most Common",
                content="""Fraudulent job offers requiring payment before starting.

Common Variants:
1. **Registration Fee**: 'Pay ₹2,999 to register for interview'
2. **Training Fee**: 'Mandatory paid training before joining - ₹5,000-15,000'
3. **Equipment/Software**: 'Buy specific laptop/software from our vendor'
4. **Background Check**: 'Pay for verification - ₹1,500'
5. **Security Deposit**: 'Refundable deposit for equipment - ₹10,000+'

Red Flags:
- **LEGITIMATE EMPLOYERS NEVER ASK FOR MONEY**
- Payment via UPI/gift cards/crypto (untraceable)
- Personal email domains (@gmail.com, @yahoo.com) not company domain
- No official offer letter on company letterhead
- Pressure: 'offer expires today', 'limited slots'
- Interview on WhatsApp/Telegram, not video call

Protection:
- Research company independently (website, LinkedIn, Glassdoor)
- Verify email domain matches company website
- NEVER pay money to get a job
- Report fake listings to job portal (Naukri, LinkedIn, Indeed)
- Government jobs NEVER have application fees via UPI""",
                category=KnowledgeCategory.JOB_SCAM,
                tags=["upfront_fee", "training_fee", "registration_fee", "most_common", "personal_email"],
                source="job_scam_research",
            ),
            KnowledgeChunk(
                id="job_002",
                title="Money Mule / Reshipping Job Scams",
                content="""'Work from home' jobs that are actually money laundering.

How it works:
1. 'Job': Receive packages, repackage, ship to different address
2. Packages purchased with STOLEN credit cards
3. You become 'money mule' - handling stolen goods
4. Police trace stolen goods to YOUR address

Variants:
- 'Quality control inspector' - check and forward packages
- 'Logistics coordinator' - receive and reship
- 'Mystery shopper' - buy items, forward to 'client'

Red Flags:
- No interview or very casual 'interview'
- Immediate hiring without verification
- Work involves receiving/sending packages
- Ask for your address for 'shipping labels'
- Payment via crypto/UPI to personal accounts

Protection:
- NEVER accept packages for others
- Legitimate work-from-home has proper contracts
- Check company registration (MCA, ROC)
- If it involves handling goods/money for others = ILLEGAL""",
                category=KnowledgeCategory.JOB_SCAM,
                tags=["money_mule", "reshipping", "stolen_goods", "work_from_home", "illegal"],
                source="job_scam_research",
            ),
            KnowledgeChunk(
                id="job_003",
                title="Fake Offer Letter / Impersonation",
                content="""Scammers impersonate real companies with fake offer letters.

Tactics:
1. Copy real company logo, format
2. Use similar email: careers@company.com vs careers@company-services.com
3. Offer high salary for entry-level role
4. Request documents: Aadhaar, PAN, bank details, passport
5. May ask for 'joining fee' after sending offer

Red Flags:
- Email domain mismatch (check SPF/DKIM)
- Offer letter has grammatical errors, wrong formatting
- Salary unrealistic for role/experience
- Requests sensitive documents BEFORE joining
- No verifiable HR contact on company website
- Joining date too soon (within days)

Protection:
- Call company HR on OFFICIAL website number
- Verify email domain on company's career page
- Check LinkedIn for real employees
- Never share Aadhaar/PAN/bank details before verified offer
- Report to company's security team and job portal""",
                category=KnowledgeCategory.JOB_SCAM,
                tags=["fake_offer_letter", "impersonation", "document_theft", "email_spoofing"],
                source="job_scam_research",
            ),

            # Document Fraud
            KnowledgeChunk(
                id="doc_001",
                title="Forged KYC Document Detection",
                content="""Common KYC document forgeries in financial fraud.

Types:
1. **Aadhaar Card**: Morphing photo, changing DOB/address, fake QR
2. **PAN Card**: Editing name/DOB, fake hologram
3. **Passport**: Photo substitution, visa page tampering
4. **Bank Statements**: Editing transactions, balance amounts
5. **Salary Slips**: Inflating income, fake employer details

Detection Methods:
- **Error Level Analysis (ELA)**: Detects inconsistent compression (tampering)
- **Noise Analysis**: Inconsistent sensor noise patterns
- **Copy-Move Detection**: Cloned regions within image
- **Metadata Analysis**: Software traces, timestamps
- **QR/Barcode Validation**: Cross-reference with issuing authority
- **Font/Consistency Check**: Mismatched fonts, alignment

Key Indicators:
- Inconsistent lighting/shadows on photo
- Pixelation around edited areas
- Mismatched fonts within same document
- Incorrect security features (holograms, watermarks)
- Metadata shows recent edit software""",
                category=KnowledgeCategory.DOCUMENT_FRAUD,
                tags=["kyc_forgery", "aadhaar", "pan", "passport", "ela", "copy_move", "noise_analysis"],
                source="document_forensics_research",
            ),
            KnowledgeChunk(
                id="doc_002",
                title="Payment Screenshot Tampering Detection",
                content="""Fake payment screenshots used in UPI/refund scams.

How fraudsters create them:
1. Edit legitimate screenshot (Photoshop, inspect element)
2. Use fake screenshot generator apps/websites
3. Modify HTML of banking/UPI web portal

Detection:
- **ELA Analysis**: Edited areas show different compression
- **Font Analysis**: UPI apps use specific fonts; mismatches indicate editing
- **Timestamp/Transaction ID Validation**: Cross-check with actual transaction
- **UI Element Consistency**: App version, battery %, network indicators
- **Metadata**: Screenshot metadata vs claimed time

Red Flags:
- Transaction ID doesn't exist in UPI app
- Amount format mismatches (extra spaces, wrong currency symbol)
- Battery percentage unrealistic for claimed time
- Network indicator doesn't match location
- App version older than current release""",
                category=KnowledgeCategory.DOCUMENT_FRAUD,
                tags=["screenshot_tampering", "payment_proof", "fake_screenshot", "transaction_verification"],
                source="document_forensics_research",
            ),

            # Regulations
            KnowledgeChunk(
                id="reg_001",
                title="RBI Master Direction on Fraud Classification and Reporting",
                content="""Key RBI guidelines for fraud detection and reporting (Updated 2024).

Fraud Classification:
- **Category 1**: Fraud by bank employees
- **Category 2**: Fraud by outsiders (cyber, identity theft)
- **Category 3**: Fraud involving borrowers
- **Category 4**: Fraud in digital payment systems

Reporting Requirements:
1. **FMR-1**: Individual fraud report (within 3 weeks of detection)
2. **FMR-2**: Quarterly consolidated return
3. **FMR-3**: Annual fraud report
4. **Early Warning Signal (EWS)**: Report within 24 hours for frauds >₹5 crore

Digital Payment Fraud Specific:
- UPI frauds must be reported via NPCI's FRAUD portal
- Banks must have 24x7 fraud reporting helpline
- Customer liability limited to ₹0 if reported within 3 days (unauthorized transaction)
- Zero liability if bank's negligence (e.g., SIM swap without verification)

Compliance for Fraud Detection Systems:
- Real-time transaction monitoring mandatory
- Velocity checks, device fingerprinting, geo-location
- Alert generation for suspicious patterns
- Audit trail retention minimum 10 years
- Regular penetration testing of fraud systems""",
                category=KnowledgeCategory.REGULATIONS,
                tags=["rbi", "fraud_reporting", "fmr", "digital_payments", "customer_liability", "compliance"],
                source="rbi_guidelines",
            ),
            KnowledgeChunk(
                id="reg_002",
                title="NPCI UPI Procedural Guidelines for Fraud Prevention",
                content="""NPCI guidelines specific to UPI fraud prevention.

Key Requirements for Banks/PSPs:
1. **Transaction Monitoring**: Real-time scoring for every transaction
2. **Velocity Controls**: Configurable limits per user/device/VPA
3. **Device Binding**: Mandatory device registration for high-value
4. **Collect Request Controls**: Additional authentication for collect >₹2,000
5. **QR Code Security**: Dynamic QR with amount, merchant verification
6. **Customer Awareness**: Mandatory in-app fraud warnings

Fraud Reporting:
- Report to NPCI within 24 hours via FRAUD portal
- Share fraud VPA/device/IP with NPCI for ecosystem blocking
- Customer dispute resolution within 48 hours (NPCI mandate)

Technical Controls:
- Risk-based authentication (step-up for high risk)
- Behavioral biometrics for anomaly detection
- Negative database sharing across PSPs
- Mandatory OTP for collect requests > threshold

Penalties:
- Non-compliance attracts penalties under PSS Act
- Repeated violations: license review for PSPs""",
                category=KnowledgeCategory.REGULATIONS,
                tags=["npci", "upi", "psp", "velocity", "device_binding", "collect_request", "qr_code", "penalties"],
                source="npci_guidelines",
            ),
            KnowledgeChunk(
                id="reg_003",
                title="DPDP Act 2023 - Data Privacy in Fraud Detection",
                content="""Digital Personal Data Protection Act implications for fraud systems.

Key Principles:
1. **Consent**: Explicit consent for personal data processing
2. **Purpose Limitation**: Data only for fraud detection, not marketing
3. **Data Minimization**: Collect only necessary fraud signals
4. **Storage Limitation**: Retain only as long as necessary
5. **Security**: Reasonable security safeguards (encryption, access control)
6. **Accountability**: Data fiduciary responsible for compliance

Fraud Detection Specific:
- Device fingerprint, IP, location = personal data (requires consent/legitimate interest)
- Transaction data = financial data (special category, higher protection)
- Sharing fraud data with other entities = requires legal basis
- Cross-border data transfer = adequate protection required

Compliance Actions:
- Privacy notice explaining fraud data processing
- Data retention policy (e.g., 7 years for fraud records)
- Data Protection Impact Assessment (DPIA) for AI/ML models
- Right to access/correction/deletion (with fraud investigation exceptions)
- Data Protection Officer appointment if large scale processing""",
                category=KnowledgeCategory.REGULATIONS,
                tags=["dpdp", "data_privacy", "personal_data", "consent", "retention", "dpia", "cross_border"],
                source="dpdp_act",
            ),

            # Prevention Tips
            KnowledgeChunk(
                id="prev_001",
                title="UPI Security Best Practices - Complete Guide",
                content="""Comprehensive UPI fraud prevention checklist.

Daily Habits:
✅ Check UPI transaction history daily
✅ Enable transaction notifications (app + email)
✅ Set UPI transaction limits (per txn, daily)
✅ Use biometric/app lock for UPI apps
✅ Never share UPI PIN, OTP, QR code screenshots

Before Every Transaction:
✅ Verify payee name matches expected
✅ Check payment DIRECTION (Pay vs Receive/Collect)
✅ Confirm amount before entering PIN
✅ Don't approve if unsure - cancel and verify

Device Security:
✅ Install apps ONLY from Play Store/App Store
✅ Keep OS and apps updated
✅ Don't root/jailbreak device
✅ Use antivirus/mobile security
✅ Avoid public WiFi for financial transactions

If You Suspect Fraud:
1. **Immediately**: Change UPI PIN, disable UPI in app
2. **Within 1 hour**: Call bank 24x7 helpline
3. **Within 24 hours**: File police complaint (cybercrime.gov.in)
4. **Within 3 days**: Formal written complaint to bank (for zero liability)
5. **Ongoing**: Monitor credit report for identity theft

Emergency Numbers:
- Bank 24x7 helpline (on back of debit card)
- 1930 - National Cybercrime Helpline
- cybercrime.gov.in - Online reporting
- NPCI UPI help: 1800-120-1740""",
                category=KnowledgeCategory.PREVENTION,
                tags=["upi_security", "best_practices", "daily_habits", "emergency", "zero_liability"],
                source="prevention_guide",
            ),
            KnowledgeChunk(
                id="prev_002",
                title="SMS Phishing (Smishing) Detection Checklist",
                content="""Quick reference to identify phishing SMS.

ASK YOURSELF for every SMS:
□ Is the sender a known contact or official sender ID (e.g., ICICIBK)?
□ Does it create URGENCY ('immediate', 'urgent', 'expire today', 'limited time')?
□ Does it THREATEN ('blocked', 'closed', 'legal action', 'penalty')?
□ Does it offer TOO-GOOD-TO-BE-TRUE ('cashback', 'prize', 'lottery', 'refund')?
□ Does it ask for SENSITIVE INFO (OTP, PIN, password, CVV, Aadhaar, PAN)?
□ Does it contain a LINK (especially shortened: bit.ly, tinyurl, t.co)?
□ Is the language poor (spelling, grammar, formatting)?

IF YES TO ANY → HIGH PROBABILITY OF PHISHING

NEVER:
❌ Click links in suspicious SMS
❌ Reply to suspicious SMS
❌ Call numbers in suspicious SMS
❌ Share OTP/PIN/password with ANYONE
❌ Download apps from SMS links

ALWAYS:
✅ Verify independently (official app, website, saved number)
✅ Report to 1930 / cybercrime.gov.in
✅ Block sender
✅ Delete SMS""",
                category=KnowledgeCategory.PREVENTION,
                tags=["smishing", "phishing_checklist", "detection", "urgency", "threats", "links"],
                source="prevention_guide",
            ),
            KnowledgeChunk(
                id="prev_003",
                title="Job Scam Prevention - Complete Checklist",
                content="""Verify every job offer before proceeding.

BEFORE APPLYING:
□ Research company: Website, LinkedIn, Glassdoor, MCA registration
□ Check email domain: Must match company website (not @gmail.com)
□ Verify contact: Call company HR on official number
□ Check salary: Realistic for role, experience, location

DURING PROCESS:
□ Video interview with real employee (not just WhatsApp chat)
□ Official offer letter on company letterhead with authorized signatory
□ No requests for money at ANY stage
□ Background check by THIRD PARTY, not employer asking you to pay

RED FLAGS - WALK AWAY IMMEDIATELY:
🚨 Any upfront payment (registration, training, equipment, deposit)
🚨 Personal email for official communication
🚨 Pressure to decide quickly ('offer expires today')
🚨 Request for Aadhaar/PAN/bank details before offer
🚨 Interview only on WhatsApp/Telegram
🚨 'Work from home' handling packages/money
🚨 Crypto/UPI/gift card payments

LEGITIMATE PROCESS:
1. Apply on company career portal
2. Screening call → Technical interview → HR interview
3. Official offer letter via email from company domain
4. Background verification by authorized agency
5. Joining formalities on first day at office/remote

REPORT:
- Job portal (Naukri, LinkedIn, Indeed) - 'Report Job'
- Cybercrime.gov.in
- Company's official security/careers email""",
                category=KnowledgeCategory.PREVENTION,
                tags=["job_scam", "prevention", "verification", "red_flags", "legitimate_process", "reporting"],
                source="prevention_guide",
            ),

            # Case Studies
            KnowledgeChunk(
                id="case_001",
                title="Case Study: Large-Scale QR Code Scam Network (2023)",
                content="""Real fraud case demonstrating organized QR code fraud.

Incident:
- 200+ victims across 5 states
- Total loss: ₹2.3 crores
- Duration: 6 months before detection

Modus Operandi:
1. Fraudsters posed as 'OLX/Quikr buyers' for high-value items
2. Sent 'advance payment' QR codes via WhatsApp
3. QR actually requested payment FROM victim
4. Used social engineering: 'My UPI limit reached, use this QR'
5. Victims scanned, entered PIN, lost ₹10K-₹2L each

Investigation:
- Common VPA patterns identified (random alphanumeric @ybl, @oksbi)
- Device fingerprints linked 50+ fraudster devices
- IP geolocation clustered in 3 cities
- Money mule network: 200+ bank accounts used for layering

Resolution:
- NPCI blocked 150+ VPAs across ecosystem
- 12 arrests across 3 states
- ₹45 lakhs recovered
- Banks implemented collect request warnings

Lessons:
- QR code direction verification critical
- Cross-bank VPA sharing effective
- Device fingerprinting catches mule networks
- Customer education reduces success rate""",
                category=KnowledgeCategory.CASE_STUDIES,
                tags=["qr_code_scam", "organized_crime", "money_mule", "vpa_blocking", "device_fingerprint"],
                source="case_study_2023",
            ),
            KnowledgeChunk(
                id="case_002",
                title="Case Study: SIM Swap Leading to ₹85L UPI Fraud (2024)",
                content="""High-value account takeover via SIM swap.

Incident:
- Victim: Senior executive, multiple bank accounts
- Loss: ₹85 lakhs in 45 minutes
- Time: 2:00 AM (low monitoring)

Attack Flow:
1. Fraudster gathered personal info from LinkedIn, data breaches
2. Called telecom operator at 1:30 AM (low staff verification)
3. Social engineered SIM swap using victim's details
4. Received all banking OTPs on new SIM
5. Reset UPI pins, added new devices, transferred funds

Detection:
- Bank's velocity rule triggered (5 txns in 3 minutes)
- Device anomaly: New device, new location
- But transactions already processed (UPI instant)

Recovery:
- Only ₹12L recovered (transferred to crypto/offshore)
- Telecom operator liable for insufficient verification
- Victim got partial refund under RBI zero-liability (reported in 2 hrs)

Prevention Applied:
- Bank: Mandatory biometric for new device >₹50K
- Telecom: SIM swap requires in-person + OTP on old SIM
- UPI: Collect request limit ₹2K without step-up auth
- Customer: Email alerts + SMS for all UPI txns""",
                category=KnowledgeCategory.CASE_STUDIES,
                tags=["sim_swap", "account_takeover", "high_value", "night_attack", "velocity", "recovery"],
                source="case_study_2024",
            ),

            # IOCs
            KnowledgeChunk(
                id="ioc_001",
                title="Known Fraudulent VPA Patterns (Updated Weekly)",
                content="""Indicators of Compromise: Fraudulent VPA patterns.

Suspicious VPA Characteristics:
- Random alphanumeric: abc123xyz@ybl, user789@oksbi
- Generic names: payment@ybl, refund@oksbi, cashback@upi
- Impersonation: support@paytm, care@phonepe, help@googlepay
- Sequential: user001@upi, user002@upi, user003@upi
- New PSPs: @newpsp, @xyzbank (recently onboarded)

High-Risk PSP Handles:
- @ybl (Yes Bank - heavily targeted)
- @oksbi (SBI - high volume)
- @icici (ICICI)
- @hdfc (HDFC)
- @axl (Axis)
- @kotak (Kotak)
- @upi (generic)

Action:
- Block/flag VPAs matching patterns
- Share with NPCI negative database
- Monitor for new VPA registrations from suspicious devices/IPs
- Correlate with device fingerprint clusters""",
                category=KnowledgeCategory.IOC,
                tags=["vpa", "ioc", "fraudulent_vpa", "psp", "negative_database", "blocking"],
                source="threat_intel_feed",
            ),
            KnowledgeChunk(
                id="ioc_002",
                title="Known Phishing Domains and URLs",
                content="""Indicators of Compromise: Active phishing infrastructure.

Common Phishing Domain Patterns:
- Typosquatting: paytmm.com, phonepee.com, googlepayy.com
- Subdomain abuse: paytm.phishersite.com, security.icici.fake.com
- URL shorteners: bit.ly/xyz, tinyurl.com/abc, t.co/def
- Free hosting: github.io, netlify.app, vercel.app, pages.dev
- New TLDs: .xyz, .top, .online, .site, .click, .link

Active Campaign Indicators (Sample):
- kyc-verify-*.xyz (KYC phishing)
- upi-refund-*.top (Refund scam)
- job-offer-*.online (Job scam)
- parcel-track-*.click (Delivery scam)
- bank-alert-*.site (Bank spoofing)

IP Ranges Associated with Fraud:
- 203.0.113.0/24 (TEST-NET, used in examples)
- 198.51.100.0/24 (TEST-NET)
- Various VPN/proxy exit nodes
- Tor exit nodes

Action:
- Block domains at DNS/firewall level
- Share with threat intel platforms
- Monitor for new domain registrations (typosquatting)
- Use threat intel feeds for real-time blocking""",
                category=KnowledgeCategory.IOC,
                tags=["phishing_domains", "ioc", "typosquatting", "url_shorteners", "ip_ranges", "blocking"],
                source="threat_intel_feed",
            ),
        ])

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[KnowledgeCategory] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Search the knowledge base."""
        if not self._initialized:
            self.initialize()
        return self.vector_store.search(query, top_k, category_filter, min_score)

    def get_chunk(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        """Get a specific knowledge chunk."""
        return self.vector_store.get_chunk(chunk_id)

    def add_chunk(self, chunk: KnowledgeChunk) -> str:
        """Add a new knowledge chunk."""
        self.vector_store.add_chunks([chunk])
        self._chunks[chunk.id] = chunk
        return chunk.id

    def add_chunks(self, chunks: List[KnowledgeChunk]) -> List[str]:
        """Add multiple knowledge chunks."""
        ids = self.vector_store.add_chunks(chunks)
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        return ids

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_chunks": self.vector_store.count(),
            "categories": {cat.value: 0 for cat in KnowledgeCategory},
            "initialized": self._initialized,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
        }

    def rebuild_index(self):
        """Rebuild the vector index from stored chunks."""
        chunks = list(self._chunks.values())
        if chunks:
            self.vector_store.clear()
            self.vector_store.add_chunks(chunks)
            logger.info(f"Rebuilt index with {len(chunks)} chunks")


# Singleton Instance

_knowledge_base: Optional[FraudKnowledgeBase] = None


def get_knowledge_base() -> FraudKnowledgeBase:
    """Get singleton knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = FraudKnowledgeBase()
    return _knowledge_base