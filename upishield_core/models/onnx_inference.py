"""
ONNX Runtime inference for text classification models (SMS/Job scam detection).

Optimized for production inference with INT8 quantization support.
"""

import joblib
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class SmishingCategory(str, Enum):
    """Smishing scam categories."""
    PHISHING = "phishing"
    LOTTERY = "lottery"
    ROMANCE = "romance"
    INVESTMENT = "investment"
    TECH_SUPPORT = "tech_support"
    FAKE_JOB = "fake_job"
    UPI_FRAUD = "upi_fraud"
    GOVERNMENT_IMPERSONATION = "government_impersonation"
    DELIVERY = "delivery"
    LEGITIMATE = "legitimate"


class JobFraudCategory(str, Enum):
    """Job fraud categories."""
    LEGITIMATE = "legitimate"
    FAKE_JOB_FEE = "fake_job_fee"
    WORK_FROM_HOME_SCAM = "wfh_scam"
    DATA_ENTRY_TYPING = "data_entry_typing"
    HIGH_SALARY_NO_EXP = "high_salary_no_exp"
    IMPERSONATION = "impersonation"
    PYRAMID_MLM = "pyramid_mlm"
    CRYPTO_INVESTMENT = "crypto_investment"
    PERSONAL_INFO_THEFT = "info_theft"


# Category to index mapping
CATEGORY_TO_IDX = {cat.value: i for i, cat in enumerate(SmishingCategory)}
IDX_TO_CATEGORY = {i: cat.value for i, cat in enumerate(SmishingCategory)}
NUM_CATEGORIES = len(SmishingCategory)

JOB_CATEGORY_TO_IDX = {cat.value: i for i, cat in enumerate(JobFraudCategory)}
JOB_IDX_TO_CATEGORY = {i: cat.value for i, cat in enumerate(JobFraudCategory)}
NUM_JOB_CATEGORIES = len(JobFraudCategory)


@dataclass
class SmishingPrediction:
    """Structured prediction result."""
    is_fraud: bool
    fraud_score: float
    category: str
    category_probs: Dict[str, float]
    confidence: float
    explanation: str
    metadata: Dict[str, Any]


@dataclass
class JobFraudPrediction:
    """Structured prediction result for job fraud detection."""
    is_fraud: bool
    fraud_score: float
    category: str
    category_probs: Dict[str, float]
    confidence: float
    risk_level: str
    red_flags: List[str]
    explanation: str
    metadata: Dict[str, Any]


class TextCNNInference:
    """
    ONNX Runtime inference for TextCNN models (SMS/Job scam detection).
    """

    SMS_CATEGORIES = [c.value for c in SmishingCategory]
    JOB_CATEGORIES = [c.value for c in JobFraudCategory]

    SMS_CATEGORY_DESCRIPTIONS = {
        SmishingCategory.PHISHING: "Attempts to steal credentials or personal info",
        SmishingCategory.LOTTERY: "Fake prize/lottery winnings requiring payment",
        SmishingCategory.ROMANCE: "Romance/relationship scams requesting money",
        SmishingCategory.INVESTMENT: "Fake investment opportunities with guaranteed returns",
        SmishingCategory.TECH_SUPPORT: "Fake tech support claiming device issues",
        SmishingCategory.FAKE_JOB: "Fraudulent job offers requiring fees",
        SmishingCategory.UPI_FRAUD: "UPI payment fraud including PIN theft",
        SmishingCategory.GOVERNMENT_IMPERSONATION: "Impersonation of government agencies",
        SmishingCategory.DELIVERY: "Fake delivery/customs fee scams",
        SmishingCategory.LEGITIMATE: "Legitimate messages",
    }

    JOB_RED_FLAGS = [
        "registration_fee",
        "training_fee",
        "equipment_deposit",
        "unrealistic_salary",
        "no_experience_required",
        "urgent_hiring",
        "work_from_home",
        "personal_info_request",
    ]

    JOB_RED_FLAG_DESCRIPTIONS = {
        "registration_fee": "Requests payment for registration/processing",
        "training_fee": "Asks for training/certification fees",
        "equipment_deposit": "Requires equipment/security deposit",
        "unrealistic_salary": "Salary significantly above market rate",
        "no_experience_required": "High-paying role with zero experience needed",
        "urgent_hiring": "Pressure tactics - immediate joining required",
        "work_from_home": "WFH offered for roles that typically require office",
        "personal_info_request": "Requests sensitive personal info upfront",
    }

    JOB_CATEGORY_DESCRIPTIONS = {
        JobFraudCategory.FAKE_JOB_FEE: "Job requires upfront payment for registration/training",
        JobFraudCategory.WORK_FROM_HOME_SCAM: "Too-good-to-be-true work from home offer",
        JobFraudCategory.DATA_ENTRY_TYPING: "Data entry/typing job with unrealistic pay",
        JobFraudCategory.HIGH_SALARY_NO_EXP: "Unrealistically high salary for no experience",
        JobFraudCategory.IMPERSONATION: "Impersonates a legitimate company",
        JobFraudCategory.PYRAMID_MLM: "Pyramid scheme or MLM recruitment",
        JobFraudCategory.CRYPTO_INVESTMENT: "Crypto investment scheme disguised as job",
        JobFraudCategory.PERSONAL_INFO_THEFT: "Attempts to harvest personal information",
    }

    def __init__(
        self,
        model_dir: str = "./models",
        model_name: str = "smishing_cnn.onnx",
        vocab_path: str = "sms_vocab.pkl",
        max_length: int = 200,
        num_classes: int = NUM_CATEGORIES,
        categories: List[str] = None,
        category_descriptions: Dict = None,
    ):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / model_name
        self.vocab_path = self.model_dir / vocab_path
        self.max_length = max_length
        self.num_classes = num_classes
        self.categories = categories or self.SMS_CATEGORIES
        self.category_descriptions = category_descriptions or self.SMS_CATEGORY_DESCRIPTIONS

        self.session: Optional[ort.InferenceSession] = None
        self.vocab: Dict[str, int] = {}
        self._loaded = False

    def load(self) -> bool:
        """Load ONNX model and vocabulary."""
        try:
            # Load vocabulary
            if self.vocab_path.exists():
                self.vocab = joblib.load(self.vocab_path)
            else:
                raise FileNotFoundError(f"Vocabulary not found at {self.vocab_path}")

            # Load ONNX model
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 2
            session_options.inter_op_num_threads = 1
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_pattern = True
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"]
            )

            # Warm up
            dummy_input = np.random.randint(0, len(self.vocab), (1, self.max_length), dtype=np.int64)
            _ = self.session.run(None, {"input_ids": dummy_input})

            self._loaded = True
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            self._loaded = False
            return False

    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization matching training."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [self.vocab.get(t, 1) for t in tokens]  # 1 = <UNK>

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to fixed-length array."""
        tokens = self._tokenize(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))  # 0 = <PAD>
        return np.array(tokens, dtype=np.int64).reshape(1, -1)

    def _run_inference(self, texts: List[str]) -> List[float]:
        """Run ONNX inference (batch size fixed to 1) and return fraud probabilities."""
        scores = []
        for text in texts:
            encoded = self._encode(text)
            outputs = self.session.run(None, {"input_ids": encoded})
            scores.append(float(np.squeeze(outputs[0])))
        return scores

    def _classify_category(self, text: str, fraud_prob: float) -> str:
        """Determine scam sub-type from keyword heuristics on the text."""
        if fraud_prob < 0.5:
            return SmishingCategory.LEGITIMATE.value

        t = text.lower()
        checks = [
            (SmishingCategory.LOTTERY, ["lottery", "prize", "winner", "won", "congratulations", "draw", "lucky"]),
            (SmishingCategory.UPI_FRAUD, ["upi", "vpa", "bank account", "pin", "otp", "transfer", "payment", "refund"]),
            (SmishingCategory.GOVERNMENT_IMPERSONATION, ["kyc", "aadhaar", "income tax", "pan card", "government", "voter id", "epic"]),
            (SmishingCategory.INVESTMENT, ["investment", "return", "profit", "double your", "stock", "crypto", "guaranteed"]),
            (SmishingCategory.TECH_SUPPORT, ["virus", "hack", "device", "technical", "support", "android", "pc"]),
            (SmishingCategory.DELIVERY, ["delivery", "courier", "parcel", "shipment", "customs", "post office"]),
            (SmishingCategory.FAKE_JOB, ["job", "hiring", "salary", "earn", "work from home"]),
            (SmishingCategory.ROMANCE, ["love", "marriage", "match", "relationship", "matrimony"]),
            (SmishingCategory.PHISHING, ["click", "verify", "update", "urgent", "suspended", "link", "http", "www."]),
        ]
        for category, kws in checks:
            if any(kw in t for kw in kws):
                return category.value
        return SmishingCategory.PHISHING.value

    def predict(self, texts: List[str], batch_size: int = 32) -> List[SmishingPrediction]:
        """Predict categories for texts."""
        if not self._loaded:
            self.load()

        if not self._loaded:
            raise RuntimeError("Model not loaded")

        fraud_probs = self._run_inference(texts)

        predictions = []
        for text, fraud_prob in zip(texts, fraud_probs):
            pred_category = self._classify_category(text, fraud_prob)
            is_fraud = pred_category != SmishingCategory.LEGITIMATE.value

            # Build per-category probabilities scaled by binary score
            cat_probs_dict = {c: 0.0 for c in self.categories}
            if is_fraud:
                cat_probs_dict[pred_category] = fraud_prob
                cat_probs_dict[SmishingCategory.PHISHING.value] = max(
                    0.0, (1.0 - fraud_prob) * 0.3
                ) if pred_category != SmishingCategory.PHISHING.value else fraud_prob
                remaining = 1.0 - fraud_prob
                for c in self.categories:
                    if cat_probs_dict[c] == 0.0:
                        cat_probs_dict[c] = remaining / (len(self.categories) - 1)
            else:
                cat_probs_dict[SmishingCategory.LEGITIMATE.value] = 1.0 - fraud_prob
                remaining = fraud_prob
                for c in self.categories:
                    if c != SmishingCategory.LEGITIMATE.value:
                        cat_probs_dict[c] = remaining / (len(self.categories) - 1)

            confidence = float(max(fraud_prob, 1.0 - fraud_prob))

            predictions.append(SmishingPrediction(
                is_fraud=is_fraud,
                fraud_score=float(fraud_prob),
                category=pred_category,
                category_probs=cat_probs_dict,
                confidence=confidence,
                explanation=self._generate_explanation(pred_category, cat_probs_dict),
                metadata={
                    "text_length": len(text),
                    "model": "TextCNN_ONNX",
                    "fraud_prob": float(fraud_prob),
                }
            ))

        return predictions

    def predict_single(self, text: str) -> SmishingPrediction:
        """Predict for a single text."""
        return self.predict([text])[0]

    def _generate_explanation(self, category: str, probs: Dict[str, float]) -> str:
        """Generate human-readable explanation."""
        if category == SmishingCategory.LEGITIMATE.value:
            return "Message appears legitimate. No scam patterns detected."

        desc = self.category_descriptions.get(category, "Suspicious message detected")
        top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_cats = [f"{cat} ({prob:.1%})" for cat, prob in top_probs if cat != SmishingCategory.LEGITIMATE.value]

        return f"{desc}. Top predictions: {', '.join(top_cats)}"


class JobScamInference(TextCNNInference):
    """ONNX Runtime inference for Job Scam detection."""

    def __init__(self, model_dir: str = "./models"):
        super().__init__(
            model_dir=model_dir,
            model_name="job_scam_cnn.onnx",
            vocab_path="job_vocab.pkl",
            max_length=200,
            num_classes=NUM_JOB_CATEGORIES,
            categories=self.JOB_CATEGORIES,
            category_descriptions=self.JOB_CATEGORY_DESCRIPTIONS,
        )

    def _classify_job_category(self, text: str, fraud_prob: float) -> str:
        """Determine job scam sub-type from keyword heuristics."""
        if fraud_prob < 0.5:
            return JobFraudCategory.LEGITIMATE.value

        t = text.lower()
        checks = [
            (JobFraudCategory.FAKE_JOB_FEE, ["registration fee", "processing fee", "joining fee", "registration charge", "training fee", "certification fee", "course fee", "equipment deposit", "security deposit", "laptop deposit"]),
            (JobFraudCategory.WORK_FROM_HOME_SCAM, ["work from home", "remote", "wfh", "work from anywhere", "home based"]),
            (JobFraudCategory.DATA_ENTRY_TYPING, ["data entry", "typing", "form filling", "copy paste", "captcha"]),
            (JobFraudCategory.HIGH_SALARY_NO_EXP, ["fresher", "no experience", "zero experience", "0 year", "fresh graduate"]),
            (JobFraudCategory.IMPERSONATION, ["hr@", "careers@", "jobs@", "recruitment@", "official", "authorized"]),
            (JobFraudCategory.PYRAMID_MLM, ["network marketing", "mlm", "multi level", "downline", "upline", "binary plan", "referral income", "team building"]),
            (JobFraudCategory.CRYPTO_INVESTMENT, ["crypto", "bitcoin", "ethereum", "trading", "investment", "blockchain", "defi", "nft"]),
            (JobFraudCategory.PERSONAL_INFO_THEFT, ["aadhaar", "pan card", "bank details", "passport", "ssn", "id proof", "personal information", "kyc document"]),
        ]
        for category, kws in checks:
            if any(kw in t for kw in kws):
                return category.value
        return JobFraudCategory.FAKE_JOB_FEE.value

    def predict(self, texts: List[str], batch_size: int = 32) -> List[JobFraudPrediction]:
        """Predict job fraud categories for texts."""
        if not self._loaded:
            self.load()

        if not self._loaded:
            raise RuntimeError("Model not loaded")

        fraud_probs = self._run_inference(texts)

        predictions = []
        for text, fraud_prob in zip(texts, fraud_probs):
            pred_category = self._classify_job_category(text, fraud_prob)
            is_fraud = pred_category != JobFraudCategory.LEGITIMATE.value

            # Build per-category probabilities scaled by binary score
            cat_probs_dict = {c: 0.0 for c in self.categories}
            if is_fraud:
                cat_probs_dict[pred_category] = fraud_prob
                remaining = 1.0 - fraud_prob
                for c in self.categories:
                    if cat_probs_dict[c] == 0.0:
                        cat_probs_dict[c] = remaining / (len(self.categories) - 1)
            else:
                cat_probs_dict[JobFraudCategory.LEGITIMATE.value] = 1.0 - fraud_prob
                remaining = fraud_prob
                for c in self.categories:
                    if c != JobFraudCategory.LEGITIMATE.value:
                        cat_probs_dict[c] = remaining / (len(self.categories) - 1)

            confidence = float(max(fraud_prob, 1.0 - fraud_prob))

            # Detect red flags from text
            red_flags = self._detect_red_flags(text)

            # Risk level
            if fraud_prob >= 0.8:
                risk_level = "CRITICAL"
            elif fraud_prob >= 0.6:
                risk_level = "HIGH"
            elif fraud_prob >= 0.35:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            predictions.append(JobFraudPrediction(
                is_fraud=is_fraud,
                fraud_score=float(fraud_prob),
                category=pred_category,
                category_probs=cat_probs_dict,
                confidence=confidence,
                risk_level=risk_level,
                red_flags=red_flags,
                explanation=self._generate_job_explanation(pred_category, cat_probs_dict, red_flags),
                metadata={
                    "text_length": len(text),
                    "model": "JobCNN_ONNX",
                    "fraud_prob": float(fraud_prob),
                    "red_flag_probs": {flag: 1.0 if flag in red_flags else 0.0 for flag in self.JOB_RED_FLAGS},
                }
            ))

        return predictions

    def predict_single(self, text: str) -> JobFraudPrediction:
        """Predict for a single text."""
        return self.predict([text])[0]

    def _detect_red_flags(self, text: str) -> List[str]:
        """Detect red flags from job posting text."""
        text_lower = text.lower()
        flags = []

        if any(kw in text_lower for kw in ["registration fee", "processing fee", "joining fee", "registration charge"]):
            flags.append("registration_fee")
        if any(kw in text_lower for kw in ["training fee", "certification fee", "course fee", "training charge"]):
            flags.append("training_fee")
        if any(kw in text_lower for kw in ["equipment deposit", "security deposit", "laptop deposit", "equipment fee"]):
            flags.append("equipment_deposit")

        # Check for unrealistic salary
        salary_matches = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:per month|/month|monthly|salary|pay)', text_lower)
        for match in salary_matches:
            salary = float(match.replace(',', ''))
            if salary > 1500000:
                flags.append("unrealistic_salary")
                break

        if ("fresher" in text_lower or "0 years" in text_lower or "no experience" in text_lower):
            if any(kw in text_lower for kw in ["50000", "100000", "lakh", "high salary", "package"]):
                flags.append("no_experience_required")

        if any(kw in text_lower for kw in ["urgent", "immediate", "asap", "today", "hurry", "quickly"]):
            flags.append("urgent_hiring")

        if any(kw in text_lower for kw in ["work from home", "remote", "wfh", "work from anywhere"]):
            flags.append("work_from_home")

        if any(kw in text_lower for kw in ["aadhaar", "pan card", "bank details", "passport", "ssn", "id proof", "personal information"]):
            flags.append("personal_info_request")

        return flags

    def _generate_job_explanation(self, category: str, probs: Dict[str, float], red_flags: List[str]) -> str:
        """Generate human-readable explanation for job fraud."""
        if category == JobFraudCategory.LEGITIMATE.value:
            return "Job posting appears legitimate. No major fraud indicators detected."

        base = self.JOB_CATEGORY_DESCRIPTIONS.get(category, "Suspicious job posting detected")
        top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_cats = [f"{cat} ({prob:.1%})" for cat, prob in top_probs if cat != JobFraudCategory.LEGITIMATE.value]

        flag_desc = ""
        if red_flags:
            flag_desc = f" Red flags: {', '.join([self.JOB_RED_FLAG_DESCRIPTIONS.get(f, f) for f in red_flags[:3]])}."

        return f"{base}. Top predictions: {', '.join(top_cats)}.{flag_desc}"


# Singleton instances
_sms_detector: Optional[TextCNNInference] = None
_job_detector: Optional[JobScamInference] = None


def get_sms_detector(model_dir: str = "./models") -> TextCNNInference:
    """Get or create SMS detector singleton."""
    global _sms_detector
    if _sms_detector is None:
        _sms_detector = TextCNNInference(model_dir=model_dir)
        _sms_detector.load()
    return _sms_detector


def get_job_detector(model_dir: str = "./models") -> JobScamInference:
    """Get or create Job scam detector singleton."""
    global _job_detector
    if _job_detector is None:
        _job_detector = JobScamInference(model_dir=model_dir)
        _job_detector.load()
    return _job_detector