#!/usr/bin/env python
"""
Train fraud detection models and export to ONNX for UPIShield.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import warnings
warnings.filterwarnings("ignore")


# --- Model Definitions ---
class FraudMLP(nn.Module):
    """Multi-layer perceptron for tabular fraud detection."""
    def __init__(self, input_dim=50, hidden_dims=[64, 32, 16], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TextCNN(nn.Module):
    """CNN for text classification (SMS/Job scam detection)."""
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, filter_sizes=[3, 4, 5], num_classes=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch, num_filters, seq_len-fs+1)]
        x = [torch.max_pool1d(c, c.size(2)).squeeze(2) for c in x]  # [(batch, num_filters)]
        x = torch.cat(x, dim=1)  # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)


# --- Data Preparation ---
def prepare_tabular_data(csv_path, target_col="is_fraud", test_size=0.2):
    """Load and prepare tabular data for fraud MLP."""
    df = pd.read_csv(csv_path)

    # Feature engineering
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_night'] = df['hour'].isin([0,1,2,3,4,5]).astype(int)
    df['amount_log'] = np.log1p(df['amount'])
    df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
    df['is_high_value'] = (df['amount'] > 50000).astype(int)
    df['txn_type_encoded'] = pd.factorize(df['txn_type'])[0]
    df['merchant_cat_encoded'] = pd.factorize(df['merchant_category'])[0]

    # Hash-based features for categorical
    for col in ['payer_vpa', 'payee_vpa', 'device_id', 'client_ip']:
        df[f'{col}_hash'] = df[col].apply(lambda x: hash(str(x)) % 10000 / 10000.0)

    # Select features
    feature_cols = [
        'amount', 'amount_log', 'hour', 'day_of_week', 'is_night',
        'is_round_amount', 'is_high_value', 'txn_type_encoded',
        'merchant_cat_encoded', 'payer_vpa_hash', 'payee_vpa_hash',
        'device_id_hash', 'client_ip_hash'
    ]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def prepare_text_data(csv_path, text_col="text", target_col="label", max_vocab=20000, max_len=200):
    """Load and prepare text data for CNN."""
    df = pd.read_csv(csv_path)

    # Simple tokenizer
    from collections import Counter
    import re

    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    all_tokens = []
    for text in df[text_col]:
        all_tokens.extend(tokenize(text))

    vocab = Counter(all_tokens).most_common(max_vocab - 2)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(vocab):
        word2idx[word] = i + 2

    def encode(text):
        tokens = tokenize(text)
        idxs = [word2idx.get(t, 1) for t in tokens]
        if len(idxs) > max_len:
            idxs = idxs[:max_len]
        else:
            idxs = idxs + [0] * (max_len - len(idxs))
        return idxs

    X = np.array([encode(t) for t in df[text_col]])
    y = df[target_col].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, word2idx


# --- Training Functions ---
def train_fraud_mlp(X_train, y_train, X_test, y_test, input_dim, epochs=30, batch_size=256):
    """Train the tabular fraud MLP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = FraudMLP(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_auc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(yb.numpy())

        auc = roc_auc_score(all_labels, all_preds)
        scheduler.step(1 - auc)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "models/fraud_mlp_best.pth")

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, AUC={auc:.4f}, Best={best_auc:.4f}")

    print(f"Best AUC: {best_auc:.4f}")
    model.load_state_dict(torch.load("models/fraud_mlp_best.pth"))
    return model


def train_text_cnn(X_train, y_train, X_test, y_test, vocab_size, epochs=20, batch_size=128):
    """Train text CNN for SMS/Job scam detection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = TextCNN(vocab_size=vocab_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_ds = TensorDataset(torch.LongTensor(X_train), torch.FloatTensor(y_train))
    test_ds = TensorDataset(torch.LongTensor(X_test), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_auc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(yb.numpy())

        auc = roc_auc_score(all_labels, all_preds)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "models/text_cnn_best.pth")

        if epoch % 3 == 0:
            print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, AUC={auc:.4f}, Best={best_auc:.4f}")

    print(f"Best AUC: {best_auc:.4f}")
    model.load_state_dict(torch.load("models/text_cnn_best.pth"))
    return model


def train_sklearn_ensemble(X_train, y_train, X_test, y_test):
    """Train sklearn GradientBoosting as fallback/ensemble."""
    print("Training sklearn GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Sklearn GB AUC: {auc:.4f}")
    print(classification_report(y_test, (preds > 0.5).astype(int)))

    joblib.dump(model, "models/sklearn_fraud_gb.pkl")
    return model


# --- ONNX Export ---
def export_to_onnx(model, input_shape, output_path, model_name="model"):
    """Export PyTorch model to ONNX."""
    import sys
    import io

    # Suppress ONNX verbose logging that causes Unicode issues on Windows
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        model.eval()
        dummy_input = torch.randn(input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # Use dynamic_shapes instead of dynamic_axes for newer PyTorch
            verbose=False
        )
    finally:
        sys.stdout = old_stdout

    print(f"Exported {model_name} to {output_path}")

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified: {output_path}")


def export_text_cnn_onnx(model, vocab_size, max_len, output_path):
    """Export TextCNN to ONNX."""
    import sys
    import io

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        model.eval()
        dummy_input = torch.randint(0, vocab_size, (1, max_len), dtype=torch.long)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['output'],
            verbose=False
        )
    finally:
        sys.stdout = old_stdout

    print(f"Exported TextCNN to {output_path}")

    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified: {output_path}")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--upi-data", default="data/upi_transactions.csv", help="UPI transactions CSV")
    parser.add_argument("--sms-data", default="data/sms_phishing.csv", help="SMS phishing CSV")
    parser.add_argument("--job-data", default="data/job_scams.csv", help="Job scams CSV")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--models-dir", default="models", help="Output models directory")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print("=" * 60)
    print("TRAINING FRAUD DETECTION MODELS")
    print("=" * 60)

    # 1. Train UPI Fraud MLP (tabular)
    print("\n[1/4] Training UPI Fraud MLP...")
    if os.path.exists(args.upi_data):
        X_train, X_test, y_train, y_test, scaler, feat_cols = prepare_tabular_data(args.upi_data)
        joblib.dump(scaler, f"{args.models_dir}/upi_scaler.pkl")
        joblib.dump(feat_cols, f"{args.models_dir}/upi_features.pkl")

        mlp = train_fraud_mlp(X_train, y_train, X_test, y_test,
                              input_dim=X_train.shape[1], epochs=args.epochs, batch_size=args.batch_size)
        export_to_onnx(mlp, (1, X_train.shape[1]), f"{args.models_dir}/xgb_fraud_int8.onnx", "UPI Fraud MLP")

        # Also train sklearn ensemble as backup
        train_sklearn_ensemble(X_train, y_train, X_test, y_test)
    else:
        print(f"  SKIPPED: {args.upi_data} not found")

    # 2. Train SMS Phishing CNN
    print("\n[2/4] Training SMS Phishing CNN...")
    if os.path.exists(args.sms_data):
        X_train, X_test, y_train, y_test, vocab = prepare_text_data(args.sms_data)
        joblib.dump(vocab, f"{args.models_dir}/sms_vocab.pkl")

        cnn = train_text_cnn(X_train, y_train, X_test, y_test,
                             vocab_size=len(vocab), epochs=args.epochs, batch_size=args.batch_size)
        export_text_cnn_onnx(cnn, len(vocab), X_train.shape[1], f"{args.models_dir}/smishing_cnn.onnx")
    else:
        print(f"  SKIPPED: {args.sms_data} not found")

    # 3. Train Job Scam CNN
    print("\n[3/4] Training Job Scam CNN...")
    if os.path.exists(args.job_data):
        X_train, X_test, y_train, y_test, vocab = prepare_text_data(args.job_data)
        joblib.dump(vocab, f"{args.models_dir}/job_vocab.pkl")

        cnn = train_text_cnn(X_train, y_train, X_test, y_test,
                             vocab_size=len(vocab), epochs=args.epochs, batch_size=args.batch_size)
        export_text_cnn_onnx(cnn, len(vocab), X_train.shape[1], f"{args.models_dir}/job_scam_cnn.onnx")
    else:
        print(f"  SKIPPED: {args.job_data} not found")

    print("\n[4/4] Done!")
    print(f"\nModels saved to: {args.models_dir}/")
    print("Required files for UPIShield:")
    print(f"  - {args.models_dir}/xgb_fraud_int8.onnx (UPI fraud)")
    print(f"  - {args.models_dir}/smishing_cnn.onnx (SMS phishing)")
    print(f"  - {args.models_dir}/job_scam_cnn.onnx (Job scams)")
    print(f"  - {args.models_dir}/sklearn_fraud_gb.pkl (Sklearn fallback)")
    print(f"  - {args.models_dir}/upi_scaler.pkl (Feature scaler)")


if __name__ == "__main__":
    main()