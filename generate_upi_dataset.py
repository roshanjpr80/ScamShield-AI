#!/usr/bin/env python
"""
Generate synthetic UPI transaction dataset for fraud detection training.
Creates realistic transaction patterns with controllable fraud rate.
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import random
import os


# --- Configuration ---
UPI_DOMAINS = ["upi", "oksbi", "okhdfc", "okicici", "okaxis", "ybl", "ibl", "paytm", "phonepe", "gpay", "amazonpay"]
MERCHANT_CATEGORIES = [
    "personal", "ecommerce", "food_delivery", "transport", "utilities",
    "entertainment", "healthcare", "education", "financial", "retail",
    "grocery", "fuel", "shopping", "travel", "insurance"
]
LOCATIONS = [
    "Mumbai, MH", "Delhi, DL", "Bangalore, KA", "Hyderabad, TS", "Chennai, TN",
    "Kolkata, WB", "Pune, MH", "Ahmedabad, GJ", "Jaipur, RJ", "Surat, GJ",
    "Lucknow, UP", "Kanpur, UP", "Nagpur, MH", "Indore, MP", "Thane, MH"
]
DEVICE_TYPES = ["android", "ios", "web"]
TXN_TYPES = ["P2P", "P2M", "M2P", "COLLECT", "REFUND"]


# --- Fraud Pattern Generators ---
def generate_normal_user(user_id, base_date, n_txns=20):
    """Generate normal user behavior patterns."""
    txns = []
    current_date = base_date
    for _ in range(n_txns):
        # Normal hours: 8 AM - 10 PM
        hour = random.randint(8, 22)
        minute = random.randint(0, 59)
        current_date += timedelta(minutes=random.randint(5, 180))
        current_date = current_date.replace(hour=hour, minute=minute)

        amount = round(np.random.lognormal(mean=5.5, sigma=0.8), 2)  # ~250 mean
        amount = min(max(amount, 10), 10000)

        txns.append({
            "txn_id": f"txn_{hashlib.md5(f'{user_id}{current_date}'.encode()).hexdigest()[:12]}",
            "payer_vpa": user_id,
            "payee_vpa": f"{random.choice(['merchant', 'friend', 'family', 'shop'])}{random.randint(1,100)}@{random.choice(UPI_DOMAINS)}",
            "amount": amount,
            "currency": "INR",
            "txn_type": random.choice(["P2P", "P2M"]),
            "status": "SUCCESS",
            "timestamp": current_date.isoformat(),
            "device_id": f"device_{hashlib.md5(user_id.encode()).hexdigest()[:8]}",
            "client_ip": f"{random.randint(10,200)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "merchant_name": f"Merchant_{random.randint(1,500)}",
            "location": random.choice(LOCATIONS),
            "is_fraud": 0
        })
    return txns


def generate_fraud_user(user_id, base_date, n_txns=10):
    """Generate fraudulent behavior patterns."""
    txns = []
    current_date = base_date
    fraud_patterns = [
        "high_velocity",      # Many txns in short time
        "high_amount",        # Unusually large amounts
        "night_activity",     # Transactions at 2-5 AM
        "new_device",         # Sudden device change
        "collect_request",    # Fraudulent collect requests
        "round_amounts",      # Suspicious round numbers
        "geo_anomaly",        # Impossible travel
    ]

    for i in range(n_txns):
        pattern = random.choice(fraud_patterns)

        if pattern == "high_velocity":
            current_date += timedelta(minutes=random.randint(1, 5))
            amount = round(random.uniform(100, 5000), 2)
        elif pattern == "high_amount":
            current_date += timedelta(minutes=random.randint(30, 120))
            amount = round(random.uniform(50000, 200000), 2)
        elif pattern == "night_activity":
            current_date += timedelta(hours=random.randint(1, 4))
            current_date = current_date.replace(hour=random.randint(1, 4), minute=random.randint(0, 59))
            amount = round(random.uniform(500, 50000), 2)
        elif pattern == "collect_request":
            current_date += timedelta(minutes=random.randint(10, 60))
            amount = round(random.uniform(1000, 100000), 2)
        elif pattern == "round_amounts":
            current_date += timedelta(minutes=random.randint(15, 90))
            amount = float(random.choice([1000, 5000, 10000, 25000, 50000, 100000]))
        else:
            current_date += timedelta(minutes=random.randint(10, 180))
            amount = round(random.uniform(500, 50000), 2)

        txn_type = "COLLECT" if pattern == "collect_request" else random.choice(["P2P", "P2M"])

        txns.append({
            "txn_id": f"txn_{hashlib.md5(f'{user_id}{current_date}{i}'.encode()).hexdigest()[:12]}",
            "payer_vpa": user_id,
            "payee_vpa": f"scammer{random.randint(1,50)}@{random.choice(UPI_DOMAINS)}",
            "amount": amount,
            "currency": "INR",
            "txn_type": txn_type,
            "status": "PENDING",
            "timestamp": current_date.isoformat(),
            "device_id": f"device_{hashlib.md5(f'{user_id}{i}'.encode()).hexdigest()[:8]}" if pattern == "new_device" else f"device_{hashlib.md5(user_id.encode()).hexdigest()[:8]}",
            "client_ip": f"{random.randint(1,200)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            "merchant_category": "financial" if pattern == "collect_request" else random.choice(MERCHANT_CATEGORIES),
            "merchant_name": "Fake KYC Verification" if pattern == "collect_request" else f"Unknown_{random.randint(1,100)}",
            "location": random.choice(LOCATIONS),
            "is_fraud": 1,
            "fraud_pattern": pattern
        })
    return txns


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic UPI fraud dataset")
    parser.add_argument("--samples", type=int, default=10000, help="Total transactions to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.05, help="Fraction of fraud transactions (0-1)")
    parser.add_argument("--users", type=int, default=1000, help="Number of unique users")
    parser.add_argument("--output", type=str, default="data/upi_transactions.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_fraud = int(args.samples * args.fraud_rate)
    n_normal = args.samples - n_fraud

    n_fraud_users = max(1, int(args.users * args.fraud_rate * 2))
    n_normal_users = args.users - n_fraud_users

    print(f"Generating {args.samples} transactions ({n_normal} normal, {n_fraud} fraud)")
    print(f"Users: {n_normal_users} normal, {n_fraud_users} fraud")

    all_txns = []
    base_date = datetime(2026, 1, 1)

    # Normal users
    for u in range(n_normal_users):
        user_id = f"user{u}@upi"
        txns = generate_normal_user(user_id, base_date + timedelta(days=random.randint(0, 300)),
                                    n_txns=max(1, n_normal // n_normal_users))
        all_txns.extend(txns)

    # Fraud users
    for u in range(n_fraud_users):
        user_id = f"fraud{u}@upi"
        txns = generate_fraud_user(user_id, base_date + timedelta(days=random.randint(0, 300)),
                                   n_txns=max(1, n_fraud // n_fraud_users))
        all_txns.extend(txns)

    # Shuffle and trim
    random.shuffle(all_txns)
    all_txns = all_txns[:args.samples]

    df = pd.DataFrame(all_txns)
    df.to_csv(args.output, index=False)

    print(f"\nSaved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3f}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample:")
    print(df.head())


if __name__ == "__main__":
    main()