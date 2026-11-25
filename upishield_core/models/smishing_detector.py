"""
SMS Smishing (SMS Phishing) Detection Model.

Transformer-based multi-class classifier for detecting SMS scams including:
- Phishing
- Lottery/Prize scams
- Romance scams
- Investment scams
- Tech support scams
- UPI fraud
- Government impersonation
- Delivery scams
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm


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


# Category to index mapping
CATEGORY_TO_IDX = {cat.value: i for i, cat in enumerate(SmishingCategory)}
IDX_TO_CATEGORY = {i: cat.value for i, cat in enumerate(SmishingCategory)}
NUM_CATEGORIES = len(SmishingCategory)


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


class SmishingDataset(Dataset):
    """PyTorch dataset for smishing detection."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SmishingTransformer(nn.Module):
    """Transformer-based smishing classifier with multi-task heads."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_categories: int = NUM_CATEGORIES,
        hidden_dim: int = 768,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
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

        # Multi-task heads
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories),
        )

        # Binary fraud detection head (auxiliary task)
        self.fraud_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning category logits, fraud logits, and confidence."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch, hidden_size]

        # Multi-task predictions
        category_logits = self.category_head(pooled_output)
        fraud_logits = self.fraud_head(pooled_output)
        confidence = self.confidence_head(pooled_output)

        return {
            "category_logits": category_logits,
            "fraud_logits": fraud_logits,
            "confidence": confidence,
            "embeddings": pooled_output,
        }


class SmishingDetector:
    """
    High-level interface for SMS smishing detection.

    Supports:
    - Training from scratch
    - Fine-tuning pretrained models
    - ONNX export for production inference
    - Ensemble with traditional ML models
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        device: str = None,
        model_dir: str = "./models/smishing",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Optional[SmishingTransformer] = None
        self.is_trained = False

        # Category descriptions for explanations
        self.category_descriptions = {
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

    def build_model(
        self,
        hidden_dim: int = 768,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> SmishingTransformer:
        """Build the transformer model."""
        self.model = SmishingTransformer(
            model_name=self.model_name,
            num_categories=NUM_CATEGORIES,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        ).to(self.device)
        return self.model

    def load_model(self, model_path: str = None) -> SmishingTransformer:
        """Load a trained model."""
        path = Path(model_path) if model_path else self.model_dir / "best_model.pt"

        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.model = SmishingTransformer(
            model_name=checkpoint.get("model_name", self.model_name),
            num_categories=checkpoint.get("num_categories", NUM_CATEGORIES),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True

        return self.model

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str] = None,
        val_labels: List[int] = None,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        class_weights: torch.Tensor = None,
        early_stopping_patience: int = 3,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the smishing detection model.

        Args:
            train_texts: Training text samples
            train_labels: Training labels (category indices)
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Warmup steps for scheduler
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
        train_dataset = SmishingDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_loader = None
        if val_texts and val_labels:
            val_dataset = SmishingDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # Optimizer and scheduler
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

        # Training history
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
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Binary fraud labels (1 if not legitimate)
                fraud_labels = (labels != CATEGORY_TO_IDX[SmishingCategory.LEGITIMATE]).float()

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                category_logits = outputs["category_logits"]
                fraud_logits = outputs["fraud_logits"].squeeze(-1)
                confidence = outputs["confidence"].squeeze(-1)

                # Multi-task loss
                category_loss = category_criterion(category_logits, labels)
                fraud_loss = fraud_criterion(fraud_logits, fraud_labels)

                # Confidence target: 1 for correct predictions, 0 for incorrect
                with torch.no_grad():
                    preds = category_logits.argmax(dim=-1)
                    confidence_target = (preds == labels).float()

                confidence_loss = confidence_criterion(confidence, confidence_target)

                # Combined loss
                loss = category_loss + 0.3 * fraud_loss + 0.1 * confidence_loss

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
                    val_loader, category_criterion, fraud_criterion, confidence_criterion
                )
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_f1"].append(val_f1)

                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, "
                      f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

                # Early stopping and best model saving
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

    def _validate(
        self,
        val_loader: DataLoader,
        category_criterion: nn.CrossEntropyLoss,
        fraud_criterion: nn.BCEWithLogitsLoss,
        confidence_criterion: nn.MSELoss,
    ) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                fraud_labels = (labels != CATEGORY_TO_IDX[SmishingCategory.LEGITIMATE]).float()

                outputs = self.model(input_ids, attention_mask)
                category_logits = outputs["category_logits"]
                fraud_logits = outputs["fraud_logits"].squeeze(-1)
                confidence = outputs["confidence"].squeeze(-1)

                category_loss = category_criterion(category_logits, labels)
                fraud_loss = fraud_criterion(fraud_logits, fraud_labels)

                preds = category_logits.argmax(dim=-1)
                confidence_target = (preds == labels).float()
                confidence_loss = confidence_criterion(confidence, confidence_target)

                loss = category_loss + 0.3 * fraud_loss + 0.1 * confidence_loss

                val_loss += loss.item()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        return avg_val_loss, val_acc, val_f1

    def predict(self, texts: List[str], batch_size: int = 32) -> List[SmishingPrediction]:
        """
        Predict smishing categories for texts.

        Args:
            texts: List of SMS texts to classify
            batch_size: Batch size for inference

        Returns:
            List of SmishingPrediction objects
        """
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model not trained or loaded. Call train() or load_model() first.")

        self.model.eval()
        dataset = SmishingDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        predictions = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                category_logits = outputs["category_logits"]
                fraud_logits = outputs["fraud_logits"].squeeze(-1)
                confidence = outputs["confidence"].squeeze(-1)

                # Probabilities
                category_probs = F.softmax(category_logits, dim=-1)
                fraud_probs = torch.sigmoid(fraud_logits)

                for i in range(len(texts)):
                    if len(predictions) >= len(texts):
                        break

                    cat_probs = category_probs[i].cpu().numpy()
                    pred_idx = cat_probs.argmax()
                    pred_category = IDX_TO_CATEGORY[pred_idx]
                    pred_prob = cat_probs[pred_idx]
                    fraud_prob = fraud_probs[i].item()
                    conf = confidence[i].item()

                    is_fraud = pred_category != SmishingCategory.LEGITIMATE
                    fraud_score = fraud_prob if is_fraud else (1 - cat_probs[CATEGORY_TO_IDX[SmishingCategory.LEGITIMATE]])

                    # Build category prob dict
                    cat_probs_dict = {
                        IDX_TO_CATEGORY[j]: float(cat_probs[j]) for j in range(NUM_CATEGORIES)
                    }

                    predictions.append(SmishingPrediction(
                        is_fraud=is_fraud,
                        fraud_score=float(fraud_score),
                        category=pred_category,
                        category_probs=cat_probs_dict,
                        confidence=float(conf),
                        explanation=self._generate_explanation(pred_category, cat_probs_dict),
                        metadata={
                            "text_length": len(texts[len(predictions)-1]),
                            "model": self.model_name,
                            "fraud_prob": float(fraud_prob),
                        }
                    ))

        return predictions

    def predict_single(self, text: str) -> SmishingPrediction:
        """Predict for a single text."""
        return self.predict([text])[0]

    def _generate_explanation(self, category: str, probs: Dict[str, float]) -> str:
        """Generate human-readable explanation."""
        if category == SmishingCategory.LEGITIMATE:
            return "Message appears legitimate. No scam patterns detected."

        desc = self.category_descriptions.get(category, "Suspicious message detected")
        top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_cats = [f"{cat} ({prob:.1%})" for cat, prob in top_probs if cat != SmishingCategory.LEGITIMATE]

        return f"{desc}. Top predictions: {', '.join(top_cats)}"

    def save_model(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save")

        checkpoint = {
            "model_name": self.model_name,
            "num_categories": NUM_CATEGORIES,
            "model_state_dict": self.model.state_dict(),
            "category_mapping": CATEGORY_TO_IDX,
            "max_length": self.max_length,
        }
        torch.save(checkpoint, path)

    def export_onnx(self, output_path: str, opset_version: int = 17):
        """Export model to ONNX format for production inference."""
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model not trained or loaded")

        self.model.eval()

        # Dummy input
        dummy_input_ids = torch.randint(0, 1000, (1, self.max_length)).to(self.device)
        dummy_attention_mask = torch.ones(1, self.max_length).to(self.device)

        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["category_logits", "fraud_logits", "confidence"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "category_logits": {0: "batch_size"},
                "fraud_logits": {0: "batch_size"},
                "confidence": {0: "batch_size"},
            },
        )
        print(f"ONNX model exported to {output_path}")

    @classmethod
    def from_pretrained(cls, model_path: str, model_name: str = "distilbert-base-uncased"):
        """Load a pretrained detector."""
        detector = cls(model_name=model_name)
        detector.load_model(model_path)
        return detector


def compute_class_weights(labels: List[int], num_classes: int = NUM_CATEGORIES) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = torch.ones(num_classes)
    for i in range(num_classes):
        if counts.get(i, 0) > 0:
            weights[i] = total / (num_classes * counts[i])
    return weights


def evaluate_model(
    detector: SmishingDetector,
    texts: List[str],
    labels: List[int],
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate model and return detailed metrics."""
    predictions = detector.predict(texts, batch_size)
    pred_labels = [CATEGORY_TO_IDX[p.category] for p in predictions]

    report = classification_report(
        labels, pred_labels,
        target_names=[IDX_TO_CATEGORY[i] for i in range(NUM_CATEGORIES)],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(labels, pred_labels)

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"],
    }