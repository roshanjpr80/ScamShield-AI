"""
Job Posting Fraud Detection Model.

Transformer-based classifier for detecting fraudulent job postings including:
- Fake job offers with registration/training fees
- Work-from-home scams
- Data entry/typing job scams
- High salary no experience scams
- Impersonation of legitimate companies
"""

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

# Lazy imports for heavy ML libraries
_torch = None
_torch_nn = None
_torch_nn_functional = None
_torch_utils_data = None
_transformers = None

def _get_torch():
    global _torch, _torch_nn, _torch_nn_functional, _torch_utils_data
    if _torch is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        _torch = torch
        _torch_nn = nn
        _torch_nn_functional = F
        _torch_utils_data = (Dataset, DataLoader)
    return _torch, _torch_nn, _torch_nn_functional, _torch_utils_data

def _get_transformers():
    global _transformers
    if _transformers is None:
        from transformers import (
            AutoTokenizer,
            AutoModel,
            AutoConfig,
            get_linear_schedule_with_warmup,
        )
        _transformers = (AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup)
    return _transformers


class JobFraudCategory(str, Enum):
    """Job fraud categories."""
    LEGITIMATE = "legitimate"
    FAKE_JOB_FEE = "fake_job_fee"           # Registration/training fee scams
    WORK_FROM_HOME_SCAM = "wfh_scam"         # Too-good-to-be-true WFH offers
    DATA_ENTRY_TYPING = "data_entry_typing"  # Data entry/typing job scams
    HIGH_SALARY_NO_EXP = "high_salary_no_exp" # Unrealistic salary for no experience
    IMPERSONATION = "impersonation"          # Impersonating legitimate companies
    PYRAMID_MLM = "pyramid_mlm"              # Pyramid/MLM schemes
    CRYPTO_INVESTMENT = "crypto_investment"  # Crypto investment job scams
    PERSONAL_INFO_THEFT = "info_theft"       # Personal information harvesting


# Category to index mapping
JOB_CATEGORY_TO_IDX = {cat.value: i for i, cat in enumerate(JobFraudCategory)}
JOB_IDX_TO_CATEGORY = {i: cat.value for i, cat in enumerate(JobFraudCategory)}
NUM_JOB_CATEGORIES = len(JobFraudCategory)


@dataclass
class JobFraudPrediction:
    """Structured prediction result for job fraud detection."""
    is_fraud: bool
    fraud_score: float
    category: str
    category_probs: Dict[str, float]
    confidence: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    red_flags: List[str]
    explanation: str
    metadata: Dict[str, Any]


class JobPostingDataset:
    """PyTorch dataset for job posting fraud detection."""

    def __init__(
        self,
        titles: List[str],
        descriptions: List[str],
        companies: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256,
        include_structured: bool = True,
        salaries_min: List[float] = None,
        salaries_max: List[float] = None,
        experience_required: List[str] = None,
        education: List[str] = None,
        platforms: List[str] = None,
    ):
        _, _, _, (Dataset, DataLoader) = _get_torch()
        self.titles = titles
        self.descriptions = descriptions
        self.companies = companies
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_structured = include_structured

        # Structured features
        self.salaries_min = salaries_min or [0.0] * len(titles)
        self.salaries_max = salaries_max or [0.0] * len(titles)
        self.experience_required = experience_required or [""] * len(titles)
        self.education = education or [""] * len(titles)
        self.platforms = platforms or [""] * len(titles)

    def __len__(self) -> int:
        return len(self.titles)

    def _combine_text(self, idx: int) -> str:
        """Combine title, company, description into single text."""
        parts = []
        if self.titles[idx]:
            parts.append(f"Title: {self.titles[idx]}")
        if self.companies[idx]:
            parts.append(f"Company: {self.companies[idx]}")
        if self.descriptions[idx]:
            parts.append(f"Description: {self.descriptions[idx]}")
        return " | ".join(parts)

    def _extract_structured_features(self, idx: int) -> np.ndarray:
        """Extract numerical structured features."""
        # Salary features
        salary_min = self.salaries_min[idx] if idx < len(self.salaries_min) else 0.0
        salary_max = self.salaries_max[idx] if idx < len(self.salaries_max) else 0.0
        salary_avg = (salary_min + salary_max) / 2 if salary_max > 0 else salary_min

        # Experience
        exp_text = self.experience_required[idx] if idx < len(self.experience_required) else ""
        exp_years = 0
        exp_match = re.search(r'(\d+)', exp_text)
        if exp_match:
            exp_years = int(exp_match.group(1))

        # Education level encoding
        edu = self.education[idx].lower() if idx < len(self.education) else ""
        edu_map = {"10th": 1, "12th": 2, "diploma": 3, "b.tech": 4, "b.e": 4, "b.sc": 4,
                   "b.com": 4, "m.tech": 5, "m.e": 5, "m.sc": 5, "mba": 5, "phd": 6}
        edu_level = 0
        for k, v in edu_map.items():
            if k in edu:
                edu_level = max(edu_level, v)

        # Platform risk
        platform = self.platforms[idx].lower() if idx < len(self.platforms) else ""
        high_risk_platforms = ["whatsapp", "telegram", "facebook", "instagram", "sms", "email", "unknown"]
        platform_risk = 1.0 if any(p in platform for p in high_risk_platforms) else 0.0

        return np.array([
            salary_avg / 1000000.0,  # Normalize to lakhs
            exp_years / 20.0,         # Normalize
            edu_level / 6.0,          # Normalize
            platform_risk,
        ], dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        torch, _, _, _ = _get_torch()
        text = self._combine_text(idx)
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

        if self.include_structured:
            struct_features = self._extract_structured_features(idx)
            item["structured_features"] = torch.tensor(struct_features, dtype=torch.float32)

        return item


class JobFraudTransformer:
    """Transformer-based job fraud classifier with structured feature fusion."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_categories: int = NUM_JOB_CATEGORIES,
        hidden_dim: int = 768,
        structured_dim: int = 4,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        torch, nn, F, _ = _get_torch()
        AutoTokenizer, AutoModel, AutoConfig, _ = _get_transformers()

        super().__init__()
        self.model_name = model_name
        self.num_categories = num_categories

        # Load pretrained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.config.hidden_size

        # Structured feature encoder
        self.structured_encoder = nn.Sequential(
            nn.Linear(structured_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Multi-task heads
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories),
        )

        # Binary fraud detection head
        self.fraud_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Red flag detection head (multi-label)
        self.red_flag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 8),  # 8 red flag types
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        structured_features: "torch.Tensor" = None,
    ) -> Dict[str, "torch.Tensor"]:
        """Forward pass."""
        torch, _, F, _ = _get_torch()
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        # Encode structured features
        if structured_features is not None:
            struct_encoded = self.structured_encoder(structured_features)
            fused = torch.cat([pooled_output, struct_encoded], dim=-1)
            fused = self.fusion(fused)
        else:
            fused = pooled_output

        # Multi-task predictions
        category_logits = self.category_head(fused)
        fraud_logits = self.fraud_head(fused)
        confidence = self.confidence_head(fused)
        red_flags = self.red_flag_head(fused)

        return {
            "category_logits": category_logits,
            "fraud_logits": fraud_logits,
            "confidence": confidence,
            "red_flags": red_flags,
            "embeddings": fused,
        }


class JobFraudDetector:
    """
    High-level interface for job posting fraud detection.

    Features:
    - Transformer-based text classification
    - Structured feature fusion (salary, experience, education, platform)
    - Red flag detection (multi-label)
    - ONNX export for production
    - Explainable predictions
    """

    RED_FLAGS = [
        "registration_fee",
        "training_fee",
        "equipment_deposit",
        "unrealistic_salary",
        "no_experience_required",
        "urgent_hiring",
        "work_from_home",
        "personal_info_request",
    ]

    RED_FLAG_DESCRIPTIONS = {
        "registration_fee": "Requests payment for registration/processing",
        "training_fee": "Asks for training/certification fees",
        "equipment_deposit": "Requires equipment/security deposit",
        "unrealistic_salary": "Salary significantly above market rate",
        "no_experience_required": "High-paying role with zero experience needed",
        "urgent_hiring": "Pressure tactics - immediate joining required",
        "work_from_home": "WFH offered for roles that typically require office",
        "personal_info_request": "Requests sensitive personal info upfront",
    }

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        device: str = None,
        model_dir: str = "./models/job_fraud",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        torch, _, _, _ = _get_torch()
        AutoTokenizer, _, _, _ = _get_transformers()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Optional[JobFraudTransformer] = None
        self.is_trained = False

    def build_model(
        self,
        hidden_dim: int = 768,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> JobFraudTransformer:
        """Build the transformer model."""
        self.model = JobFraudTransformer(
            model_name=self.model_name,
            num_categories=NUM_JOB_CATEGORIES,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        ).to(self.device)
        return self.model

    def load_model(self, model_path: str = None) -> JobFraudTransformer:
        """Load a trained model."""
        torch, _, _, _ = _get_torch()
        path = Path(model_path) if model_path else self.model_dir / "best_model.pt"

        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.model = JobFraudTransformer(
            model_name=checkpoint.get("model_name", self.model_name),
            num_categories=checkpoint.get("num_categories", NUM_JOB_CATEGORIES),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True

        return self.model

    def train(
        self,
        train_data: Dict[str, List],
        val_data: Dict[str, List] = None,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        class_weights: torch.Tensor = None,
        early_stopping_patience: int = 3,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the job fraud detection model.

        Args:
            train_data: Dict with keys: titles, descriptions, companies, labels,
                       salaries_min, salaries_max, experience_required, education, platforms
            val_data: Same structure as train_data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Warmup steps
            max_grad_norm: Gradient clipping
            class_weights: Class weights for imbalanced data
            early_stopping_patience: Early stopping patience
            save_best: Save best model

        Returns:
            Training history dict
        """
        if self.model is None:
            self.build_model()

        # Create datasets
        train_dataset = JobPostingDataset(
            titles=train_data["titles"],
            descriptions=train_data["descriptions"],
            companies=train_data["companies"],
            labels=train_data["labels"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            salaries_min=train_data.get("salaries_min"),
            salaries_max=train_data.get("salaries_max"),
            experience_required=train_data.get("experience_required"),
            education=train_data.get("education"),
            platforms=train_data.get("platforms"),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_loader = None
        if val_data:
            val_dataset = JobPostingDataset(
                titles=val_data["titles"],
                descriptions=val_data["descriptions"],
                companies=val_data["companies"],
                labels=val_data["labels"],
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                salaries_min=val_data.get("salaries_min"),
                salaries_max=val_data.get("salaries_max"),
                experience_required=val_data.get("experience_required"),
                education=val_data.get("education"),
                platforms=val_data.get("platforms"),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # Optimizer and scheduler
        torch, nn, _, _ = _get_torch()
        _, _, _, get_linear_schedule_with_warmup = _get_transformers()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Loss functions
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        category_criterion = nn.CrossEntropyLoss(weight=class_weights)
        fraud_criterion = nn.BCEWithLogitsLoss()
        confidence_criterion = nn.MSELoss()
        red_flag_criterion = nn.BCELoss()

        # Red flag labels for training data (heuristic)
        train_red_flags = self._generate_red_flag_labels(train_data)
        if val_data:
            val_red_flags = self._generate_red_flag_labels(val_data)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                structured = batch.get("structured_features", None)
                if structured is not None:
                    structured = structured.to(self.device)

                # Binary fraud labels
                fraud_labels = (labels != JOB_CATEGORY_TO_IDX[JobFraudCategory.LEGITIMATE]).float()

                # Red flag labels for this batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_red_flags))
                batch_red_flags = torch.tensor(
                    train_red_flags[start_idx:end_idx],
                    dtype=torch.float32
                ).to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask, structured)
                category_logits = outputs["category_logits"]
                fraud_logits = outputs["fraud_logits"].squeeze(-1)
                confidence = outputs["confidence"].squeeze(-1)
                red_flag_logits = outputs["red_flags"]

                # Losses
                category_loss = category_criterion(category_logits, labels)
                fraud_loss = fraud_criterion(fraud_logits, fraud_labels)

                with torch.no_grad():
                    preds = category_logits.argmax(dim=-1)
                    confidence_target = (preds == labels).float()
                confidence_loss = confidence_criterion(confidence, confidence_target)

                red_flag_loss = red_flag_criterion(red_flag_logits, batch_red_flags)

                # Combined loss
                loss = (
                    category_loss +
                    0.3 * fraud_loss +
                    0.1 * confidence_loss +
                    0.5 * red_flag_loss
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_correct += (category_logits.argmax(dim=-1) == labels).sum().item()
                train_total += labels.size(0)

                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "acc": train_correct / train_total,
                })

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader:
                val_loss, val_acc, val_f1 = self._validate(
                    val_loader, val_red_flags,
                    category_criterion, fraud_criterion,
                    confidence_criterion, red_flag_criterion
                )
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_f1"].append(val_f1)

                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, "
                      f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    if save_best:
                        self.save_model(self.model_dir / "best_model.pt")
                        print(f"  Saved best model (val_f1={val_f1:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}")

        self.is_trained = True
        return history

    def _generate_red_flag_labels(self, data: Dict[str, List]) -> List[List[float]]:
        """Generate heuristic red flag labels from job data."""
        n = len(data["titles"])
        red_flags = []

        for i in range(n):
            flags = [0.0] * len(self.RED_FLAGS)
            title = data["titles"][i].lower() if i < len(data["titles"]) else ""
            desc = data["descriptions"][i].lower() if i < len(data["descriptions"]) else ""
            company = data["companies"][i].lower() if i < len(data["companies"]) else ""
            platform = data.get("platforms", [""])[i].lower() if i < len(data.get("platforms", [""])) else ""
            salary_min = data.get("salaries_min", [0])[i] if i < len(data.get("salaries_min", [0])) else 0
            salary_max = data.get("salaries_max", [0])[i] if i < len(data.get("salaries_max", [0])) else 0
            exp_req = data.get("experience_required", [""])[i].lower() if i < len(data.get("experience_required", [""])) else ""
            edu = data.get("education", [""])[i].lower() if i < len(data.get("education", [""])) else ""

            combined = f"{title} {desc} {company}"

            # Check each red flag
            if any(kw in combined for kw in ["registration fee", "processing fee", "joining fee", "registration charge"]):
                flags[0] = 1.0
            if any(kw in combined for kw in ["training fee", "certification fee", "course fee", "training charge"]):
                flags[1] = 1.0
            if any(kw in combined for kw in ["equipment deposit", "security deposit", "laptop deposit", "equipment fee"]):
                flags[2] = 1.0
            if salary_max > 1500000 or (salary_min > 100000 and "fresher" in combined):
                flags[3] = 1.0
            if ("fresher" in combined or "0 years" in exp_req or "no experience" in combined) and salary_min > 50000:
                flags[4] = 1.0
            if any(kw in combined for kw in ["urgent", "immediate", "asap", "today", "hurry"]):
                flags[5] = 1.0
            if "work from home" in combined or "remote" in combined or "wfh" in combined:
                flags[6] = 1.0
            if any(kw in combined for kw in ["aadhaar", "pan card", "bank details", "passport", "ssn", "id proof"]):
                flags[7] = 1.0

            # Platform-based flags
            if platform in ["whatsapp", "telegram", "facebook", "instagram", "sms", "email"]:
                flags[5] = 1.0  # urgent hiring
                flags[6] = 1.0  # work from home

            red_flags.append(flags)

        return red_flags

    def _validate(
        self,
        val_loader: DataLoader,
        val_red_flags: List[List[float]],
        category_criterion: nn.CrossEntropyLoss,
        fraud_criterion: nn.BCEWithLogitsLoss,
        confidence_criterion: nn.MSELoss,
        red_flag_criterion: nn.BCELoss,
    ) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                structured = batch.get("structured_features", None)
                if structured is not None:
                    structured = structured.to(self.device)

                fraud_labels = (labels != JOB_CATEGORY_TO_IDX[JobFraudCategory.LEGITIMATE]).float()

                start_idx = batch_idx * val_loader.batch_size
                end_idx = min(start_idx + val_loader.batch_size, len(val_red_flags))
                batch_red_flags = torch.tensor(
                    val_red_flags[start_idx:end_idx],
                    dtype=torch.float32
                ).to(self.device)

                outputs = self.model(input_ids, attention_mask, structured)
                category_logits = outputs["category_logits"]
                fraud_logits = outputs["fraud_logits"].squeeze(-1)
                confidence = outputs["confidence"].squeeze(-1)
                red_flag_logits = outputs["red_flags"]

                category_loss = category_criterion(category_logits, labels)
                fraud_loss = fraud_criterion(fraud_logits, fraud_labels)

                preds = category_logits.argmax(dim=-1)
                confidence_target = (preds == labels).float()
                confidence_loss = confidence_criterion(confidence, confidence_target)

                red_flag_loss = red_flag_criterion(red_flag_logits, batch_red_flags)

                loss = (
                    category_loss +
                    0.3 * fraud_loss +
                    0.1 * confidence_loss +
                    0.5 * red_flag_loss
                )

                val_loss += loss.item()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        return avg_val_loss, val_acc, val_f1

    def predict(
        self,
        titles: List[str],
        descriptions: List[str],
        companies: List[str],
        salaries_min: List[float] = None,
        salaries_max: List[float] = None,
        experience_required: List[str] = None,
        education: List[str] = None,
        platforms: List[str] = None,
        batch_size: int = 16,
    ) -> List[JobFraudPrediction]:
        """
        Predict job fraud for postings.

        Args:
            titles: Job titles
            descriptions: Job descriptions
            companies: Company names
            salaries_min: Minimum salaries
            salaries_max: Maximum salaries
            experience_required: Experience requirements
            education: Education requirements
            platforms: Platform where posted
            batch_size: Batch size for inference

        Returns:
            List of JobFraudPrediction objects
        """
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model not trained or loaded")

        self.model.eval()
        n = len(titles)

        # Pad lists to same length
        salaries_min = salaries_min or [0.0] * n
        salaries_max = salaries_max or [0.0] * n
        experience_required = experience_required or [""] * n
        education = education or [""] * n
        platforms = platforms or [""] * n

        _, _, _, (Dataset, DataLoader) = _get_torch()
        torch, _, F, _ = _get_torch()
        dataset = JobPostingDataset(
            titles=titles,
            descriptions=descriptions,
            companies=companies,
            labels=[0] * n,  # Dummy labels
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            salaries_min=salaries_min,
            salaries_max=salaries_max,
            experience_required=experience_required,
            education=education,
            platforms=platforms,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        predictions = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                structured = batch.get("structured_features", None)
                if structured is not None:
                    structured = structured.to(self.device)

                outputs = self.model(input_ids, attention_mask, structured)
                category_logits = outputs["category_logits"]
                fraud_logits = outputs["fraud_logits"].squeeze(-1)
                confidence = outputs["confidence"].squeeze(-1)
                red_flag_logits = outputs["red_flags"]

                category_probs = F.softmax(category_logits, dim=-1)
                fraud_probs = torch.sigmoid(fraud_logits)
                red_flag_probs = red_flag_logits

                for i in range(len(titles)):
                    if len(predictions) >= n:
                        break

                    cat_probs = category_probs[i].cpu().numpy()
                    pred_idx = cat_probs.argmax()
                    pred_category = JOB_IDX_TO_CATEGORY[pred_idx]
                    pred_prob = cat_probs[pred_idx]
                    fraud_prob = fraud_probs[i].item()
                    conf = confidence[i].item()
                    rf_probs = red_flag_probs[i].cpu().numpy()

                    is_fraud = pred_category != JobFraudCategory.LEGITIMATE
                    fraud_score = fraud_prob if is_fraud else (1 - cat_probs[JOB_CATEGORY_TO_IDX[JobFraudCategory.LEGITIMATE]])

                    # Risk level
                    if fraud_score >= 0.8:
                        risk_level = "CRITICAL"
                    elif fraud_score >= 0.6:
                        risk_level = "HIGH"
                    elif fraud_score >= 0.35:
                        risk_level = "MEDIUM"
                    else:
                        risk_level = "LOW"

                    # Red flags
                    detected_flags = [
                        self.RED_FLAGS[j] for j, p in enumerate(rf_probs) if p > 0.5
                    ]

                    cat_probs_dict = {
                        JOB_IDX_TO_CATEGORY[j]: float(cat_probs[j]) for j in range(NUM_JOB_CATEGORIES)
                    }

                    predictions.append(JobFraudPrediction(
                        is_fraud=is_fraud,
                        fraud_score=float(fraud_score),
                        category=pred_category,
                        category_probs=cat_probs_dict,
                        confidence=float(conf),
                        risk_level=risk_level,
                        red_flags=detected_flags,
                        explanation=self._generate_explanation(pred_category, cat_probs_dict, detected_flags),
                        metadata={
                            "model": self.model_name,
                            "fraud_prob": float(fraud_prob),
                            "red_flag_probs": {self.RED_FLAGS[j]: float(rf_probs[j]) for j in range(len(self.RED_FLAGS))},
                        }
                    ))

        return predictions

    def predict_single(
        self,
        title: str,
        description: str,
        company: str,
        salary_min: float = 0,
        salary_max: float = 0,
        experience: str = "",
        education: str = "",
        platform: str = "",
    ) -> JobFraudPrediction:
        """Predict for a single job posting."""
        return self.predict(
            titles=[title],
            descriptions=[description],
            companies=[company],
            salaries_min=[salary_min],
            salaries_max=[salary_max],
            experience_required=[experience],
            education=[education],
            platforms=[platform],
        )[0]

    def _generate_explanation(
        self,
        category: str,
        probs: Dict[str, float],
        red_flags: List[str],
    ) -> str:
        """Generate human-readable explanation."""
        if category == JobFraudCategory.LEGITIMATE:
            return "Job posting appears legitimate. No major fraud indicators detected."

        explanations = {
            JobFraudCategory.FAKE_JOB_FEE: "Job requires upfront payment for registration/training",
            JobFraudCategory.WORK_FROM_HOME_SCAM: "Too-good-to-be-true work from home offer",
            JobFraudCategory.DATA_ENTRY_TYPING: "Data entry/typing job with unrealistic pay",
            JobFraudCategory.HIGH_SALARY_NO_EXP: "Unrealistically high salary for no experience",
            JobFraudCategory.IMPERSONATION: "Impersonates a legitimate company",
            JobFraudCategory.PYRAMID_MLM: "Pyramid scheme or MLM recruitment",
            JobFraudCategory.CRYPTO_INVESTMENT: "Crypto investment scheme disguised as job",
            JobFraudCategory.PERSONAL_INFO_THEFT: "Attempts to harvest personal information",
        }

        base = explanations.get(category, "Suspicious job posting detected")
        top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_cats = [f"{cat} ({prob:.1%})" for cat, prob in top_probs if cat != JobFraudCategory.LEGITIMATE]

        flag_desc = ""
        if red_flags:
            flag_desc = f" Red flags: {', '.join([self.RED_FLAG_DESCRIPTIONS.get(f, f) for f in red_flags[:3]])}."

        return f"{base}. Top predictions: {', '.join(top_cats)}.{flag_desc}"

    def save_model(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save")

        checkpoint = {
            "model_name": self.model_name,
            "num_categories": NUM_JOB_CATEGORIES,
            "model_state_dict": self.model.state_dict(),
            "category_mapping": JOB_CATEGORY_TO_IDX,
            "max_length": self.max_length,
        }
        torch.save(checkpoint, path)

    def export_onnx(self, output_path: str, opset_version: int = 17):
        """Export model to ONNX format."""
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model not trained or loaded")

        self.model.eval()

        dummy_input_ids = torch.randint(0, 1000, (1, self.max_length)).to(self.device)
        dummy_attention_mask = torch.ones(1, self.max_length).to(self.device)
        dummy_structured = torch.randn(1, 4).to(self.device)

        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask, dummy_structured),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask", "structured_features"],
            output_names=["category_logits", "fraud_logits", "confidence", "red_flags"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "structured_features": {0: "batch_size"},
                "category_logits": {0: "batch_size"},
                "fraud_logits": {0: "batch_size"},
                "confidence": {0: "batch_size"},
                "red_flags": {0: "batch_size"},
            },
        )
        print(f"ONNX model exported to {output_path}")

    @classmethod
    def from_pretrained(cls, model_path: str, model_name: str = "distilbert-base-uncased"):
        """Load a pretrained detector."""
        detector = cls(model_name=model_name)
        detector.load_model(model_path)
        return detector


def compute_job_class_weights(labels: List[int], num_classes: int = NUM_JOB_CATEGORIES) -> torch.Tensor:
    """Compute class weights for imbalanced job fraud dataset."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = torch.ones(num_classes)
    for i in range(num_classes):
        if counts.get(i, 0) > 0:
            weights[i] = total / (num_classes * counts[i])
    return weights


def evaluate_job_model(
    detector: JobFraudDetector,
    titles: List[str],
    descriptions: List[str],
    companies: List[str],
    labels: List[int],
    salaries_min: List[float] = None,
    salaries_max: List[float] = None,
    experience_required: List[str] = None,
    education: List[str] = None,
    platforms: List[str] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Evaluate job fraud model and return detailed metrics."""
    predictions = detector.predict(
        titles, descriptions, companies,
        salaries_min, salaries_max,
        experience_required, education, platforms,
        batch_size
    )
    pred_labels = [JOB_CATEGORY_TO_IDX[p.category] for p in predictions]

    report = classification_report(
        labels, pred_labels,
        target_names=[JOB_IDX_TO_CATEGORY[i] for i in range(NUM_JOB_CATEGORIES)],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(labels, pred_labels)

    # Red flag evaluation
    all_detected_flags = []
    for p in predictions:
        all_detected_flags.extend(p.red_flags)

    from collections import Counter
    flag_counts = Counter(all_detected_flags)

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "red_flag_distribution": dict(flag_counts),
    }