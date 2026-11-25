#!/usr/bin/env python
"""
Generate synthetic SMS phishing (smishing) dataset.
"""
import argparse
import pandas as pd
import random
import os


# Legitimate SMS templates
LEGIT_TEMPLATES = [
    "Dear {name}, your {bank} account ending in {last4} has been credited with Rs.{amount}. Available balance: Rs.{balance}.",
    "Your OTP for {bank} transaction of Rs.{amount} is {otp}. Do not share with anyone. Valid for 5 mins.",
    "{bank} Alert: Rs.{amount} debited from your account ending in {last4} on {date}. If not you, call {phone}.",
    "Your {service} bill of Rs.{amount} is due on {date}. Pay now: {url}",
    "Thank you for using {service}. Your order #{order_id} has been shipped. Track: {url}",
    "{bank}: Your account statement for {month} is ready. Download: {url}",
    "UPI Payment of Rs.{amount} to {payee} successful. Ref: {ref_id}",
    "Your {wallet} wallet balance is Rs.{balance}. Recharge now: {url}",
]

# Smishing templates
SMISHING_TEMPLATES = [
    # KYC/Bank verification scams
    "URGENT: Your {bank} account will be blocked in 2 hours. Complete KYC immediately: {short_url}",
    "Dear Customer, your {bank} ATM card is deactivated. Reactivate now: {short_url}",
    "RBI Alert: Your {bank} account needs verification. Click: {short_url} to avoid closure.",
    "Your {bank} account has suspicious activity. Verify now: {short_url}",

    # UPI/Payment scams
    "You have received Rs.{amount} via UPI. Claim now: {short_url}",
    "UPI: Rs.{amount} debited from your account. If not you, report: {short_url}",
    "Cashback of Rs.{amount} approved! Claim your reward: {short_url}",
    "Your UPI PIN has been reset. Set new PIN: {short_url}",

    # Job/Income scams
    "Work from home! Earn Rs.{amount}/day. No experience needed. Contact WhatsApp: {phone}",
    "Part-time job: Data entry, earn Rs.{amount}/month. Register: {short_url}",
    "Earn Rs.{amount} weekly from home. Click to apply: {short_url}",

    # Lottery/Prize scams
    "Congratulations! You won Rs.{amount} in {company} lucky draw. Claim: {short_url}",
    "You are selected for {company} gift card worth Rs.{amount}. Redeem: {short_url}",
    "Your mobile number won Rs.{amount}. Collect prize: {short_url}",

    # Loan/Credit scams
    "Pre-approved loan of Rs.{amount} at 0% interest. Apply now: {short_url}",
    "Instant loan up to Rs.{amount} in 5 mins. No documents. Link: {short_url}",
    "Credit card limit increase approved! Activate: {short_url}",

    # Delivery/Parcel scams
    "Your parcel from {company} is out for delivery. Track: {short_url}",
    "Delivery failed for your order. Reschedule: {short_url}",
    "Customs duty of Rs.{amount} pending for your parcel. Pay: {short_url}",

    # Government scheme scams
    "PM Scheme: You are eligible for Rs.{amount}. Apply: {short_url}",
    "Income Tax Refund of Rs.{amount} approved. Verify bank details: {short_url}",
    "Aadhaar update required. Complete now: {short_url}",
]


BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB", "Canara", "Union Bank", "Yes Bank"]
SERVICES = ["Amazon", "Flipkart", "Swiggy", "Zomato", "Uber", "Ola", "Paytm", "PhonePe", "Google Pay", "Jio", "Airtel"]
COMPANIES = ["Tata", "Reliance", "Amazon", "Flipkart", "Government of India", "RBI", "Income Tax Dept", "UIDAI"]


def fill_template(template, is_smishing=False):
    """Fill template with random values."""
    return template.format(
        name=random.choice(["Rajesh", "Priya", "Amit", "Sunita", "Vikram", "Anjali", "Customer", "User"]),
        bank=random.choice(BANKS),
        last4=f"{random.randint(1000, 9999)}",
        amount=f"{random.randint(100, 50000)}",
        balance=f"{random.randint(1000, 500000)}",
        otp=f"{random.randint(100000, 999999)}",
        date=f"{random.randint(1, 28)}/{random.randint(1, 12)}/2026",
        phone=f"+91-{random.randint(7000000000, 9999999999)}",
        service=random.choice(SERVICES),
        url=f"https://{random.choice(SERVICES).lower()}.com/track/{random.randint(100000, 999999)}",
        order_id=f"ORD{random.randint(100000, 999999)}",
        month=random.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
        wallet=random.choice(["Paytm", "PhonePe", "Google Pay", "Amazon Pay"]),
        payee=f"{random.choice(['Merchant', 'Friend', 'Family'])} {random.randint(1, 100)}",
        ref_id=f"UPI{random.randint(100000000000, 999999999999)}",
        short_url=f"https://bit.ly/{random.randint(100000, 999999)}",
        company=random.choice(COMPANIES),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SMS phishing dataset")
    parser.add_argument("--samples", type=int, default=5000, help="Total SMS samples")
    parser.add_argument("--fraud-rate", type=float, default=0.3, help="Fraction of smishing (0-1)")
    parser.add_argument("--output", type=str, default="data/sms_phishing.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_smishing = int(args.samples * args.fraud_rate)
    n_legit = args.samples - n_smishing

    data = []

    # Legitimate SMS
    for _ in range(n_legit):
        template = random.choice(LEGIT_TEMPLATES)
        text = fill_template(template)
        data.append({"text": text, "label": 0, "type": "legitimate"})

    # Smishing
    for _ in range(n_smishing):
        template = random.choice(SMISHING_TEMPLATES)
        text = fill_template(template, is_smishing=True)
        # Categorize smishing type
        if "KYC" in template or "blocked" in template or "verification" in template:
            stype = "bank_verification"
        elif "UPI" in template or "Cashback" in template or "debited" in template:
            stype = "upi_payment"
        elif "job" in template.lower() or "earn" in template.lower() or "Work from home" in template:
            stype = "job_scam"
        elif "won" in template or "Congratulations" in template or "gift card" in template:
            stype = "lottery_prize"
        elif "loan" in template.lower() or "credit" in template.lower():
            stype = "loan_credit"
        elif "parcel" in template or "delivery" in template or "Customs" in template:
            stype = "delivery_parcel"
        else:
            stype = "government_scheme"
        data.append({"text": text, "label": 1, "type": stype})

    random.shuffle(data)
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)

    print(f"Saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Smishing rate: {df['label'].mean():.3f}")
    print(f"\nType distribution:")
    print(df['type'].value_counts())
    print(f"\nSample legitimate:")
    print(df[df['label']==0]['text'].iloc[0])
    print(f"\nSample smishing:")
    print(df[df['label']==1]['text'].iloc[0])


if __name__ == "__main__":
    main()