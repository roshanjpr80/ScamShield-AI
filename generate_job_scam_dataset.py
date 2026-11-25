#!/usr/bin/env python
"""
Generate synthetic job scam dataset.
"""
import argparse
import pandas as pd
import random
import os


# Legitimate job posting templates
LEGIT_JOB_TEMPLATES = [
    "{company} is hiring {role} in {location}. Experience: {exp} years. Skills: {skills}. Salary: {salary}. Apply at careers.{company}.com",
    "We are looking for a {role} to join our team in {location}. {exp}+ years experience required. {skills} proficiency needed. Competitive package.",
    "Job Opening: {role} at {company}, {location}. Requirements: {exp} years in {skills}. Full-time position. Apply via LinkedIn.",
    "{company} seeks {role} for {location} office. Must have {exp} years experience with {skills}. Benefits include health insurance, PF.",
    "Urgent hiring: {role} at {location}. {exp} years exp in {skills}. Salary {salary}. Walk-in interview on {date}.",
]

# Job scam templates
SCAM_JOB_TEMPLATES = [
    # Registration fee scams
    "Work from home! {role} position. Earn Rs.{salary}/month. No experience needed. Registration fee: Rs.{fee}. Contact WhatsApp: {phone}",
    "Part-time {role} job. Earn Rs.{salary}/week. Flexible hours. Pay training fee Rs.{fee} to start. Call {phone}",
    "Data entry jobs available. Earn Rs.{salary}/day. Work from home. One-time registration Rs.{fee}. WhatsApp {phone}",

    # Advance payment scams
    "Hiring {role} for {company}. Salary Rs.{salary}/month. Send resume to {email}. Processing fee Rs.{fee} required.",
    "Immediate joining for {role}. Salary Rs.{salary}. Document verification charge Rs.{fee}. Contact {phone}",

    # Fake company/impersonation
    "{company} recruitment: {role} positions open. Salary Rs.{salary}. Apply at {fake_url}. Fee: Rs.{fee}",
    "We are {company} HR. You are selected for {role}. Salary Rs.{salary}. Pay security deposit Rs.{fee} via UPI to {upi}",

    # Work from home / easy money
    "Earn Rs.{salary}/month from home! Simple {role} work. No skills needed. Just pay Rs.{fee} for training kit. WhatsApp {phone}",
    "Online {role} jobs. Daily payout Rs.{salary}. Zero investment* (*conditions apply). Register: {fake_url}",

    # Government job scams
    "Government job alert: {role} vacancies. Salary Rs.{salary}. Apply before {date}. Form fee Rs.{fee}. Link: {fake_url}",
    "Railway/SSC/Banking jobs. Guaranteed selection. Pay Rs.{fee} for coaching + placement. Contact {phone}",

    # International job scams
    "Abroad job opportunity: {role} in {country}. Salary Rs.{salary} (equivalent). Visa processing fee Rs.{fee}. Contact {phone}",
    "Work in {country} as {role}. High salary + accommodation. Agent fee Rs.{fee}. WhatsApp {phone}",

    # Cryptocurrency/Investment scams disguised as jobs
    "Crypto trading {role} position. Earn Rs.{salary}/day. Training provided. Deposit Rs.{fee} to start trading. Link: {fake_url}",
]

COMPANIES = ["TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "Accenture", "Capgemini", "Cognizant", "IBM", "Microsoft", "Google", "Amazon", "Flipkart", "Paytm", "PhonePe", "Zomato", "Swiggy", "Ola", "Uber", "Byju's", "Unacademy"]
ROLES = ["Software Engineer", "Data Analyst", "Data Entry Operator", "Customer Support", "Sales Executive", "Marketing Associate", "HR Recruiter", "Accountant", "Content Writer", "Graphic Designer", "UI/UX Designer", "DevOps Engineer", "QA Tester", "Business Analyst", "Product Manager"]
LOCATIONS = ["Bangalore", "Hyderabad", "Pune", "Mumbai", "Delhi", "Chennai", "Kolkata", "Ahmedabad", "Gurgaon", "Noida", "Kochi", "Jaipur", "Indore", "Lucknow", "Bhubaneswar"]
COUNTRIES = ["USA", "Canada", "UK", "Germany", "Australia", "Singapore", "UAE", "Qatar", "Saudi Arabia", "Japan"]
SKILLS = ["Python", "Java", "SQL", "Excel", "React", "Node.js", "AWS", "Docker", "Kubernetes", "Machine Learning", "Tableau", "Power BI", "Communication", "Problem Solving"]


def fill_legit(template):
    return template.format(
        company=random.choice(COMPANIES),
        role=random.choice(ROLES),
        location=random.choice(LOCATIONS),
        exp=random.randint(0, 8),
        skills=", ".join(random.sample(SKILLS, random.randint(2, 4))),
        salary=f"{random.randint(3, 50)} LPA",
        date=f"{random.randint(1, 28)}/{random.randint(1, 12)}/2026",
    )


def fill_scam(template):
    salary_monthly = random.randint(15000, 200000)
    fee = random.choice([500, 999, 1999, 2500, 4999, 9999, 15000, 25000])
    phone = f"+91-{random.randint(6000000000, 9999999999)}"
    email = f"hr@{random.choice(['jobs', 'career', 'hiring', 'recruitment', 'talent'])}{random.randint(1, 999)}.com"
    fake_url = f"https://{random.choice(['apply', 'jobs', 'career', 'hiring', 'register'])}{random.randint(1, 999)}.{random.choice(['com', 'in', 'xyz', 'online', 'site'])}"
    upi = f"{random.choice(['hr', 'payroll', 'finance', 'recruitment'])}@{random.choice(['upi', 'oksbi', 'okhdfc', 'paytm'])}"
    country = random.choice(COUNTRIES)
    date = f"{random.randint(1, 28)}/{random.randint(1, 12)}/2026"

    return template.format(
        company=random.choice(COMPANIES),
        role=random.choice(ROLES),
        location=random.choice(LOCATIONS),
        salary=salary_monthly,
        fee=fee,
        phone=phone,
        email=email,
        fake_url=fake_url,
        upi=upi,
        country=country,
        date=date,
    )


def categorize_scam(text):
    text_lower = text.lower()
    if "registration fee" in text_lower or "training fee" in text_lower or "processing fee" in text_lower:
        return "registration_fee"
    if "security deposit" in text_lower or "document verification" in text_lower:
        return "advance_payment"
    if "government" in text_lower or "railway" in text_lower or "ssc" in text_lower or "banking job" in text_lower:
        return "government_job_scam"
    if "abroad" in text_lower or "visa" in text_lower or "overseas" in text_lower:
        return "international_job_scam"
    if "crypto" in text_lower or "trading" in text_lower or "deposit" in text_lower:
        return "crypto_investment_scam"
    if "work from home" in text_lower or "easy money" in text_lower or "zero investment" in text_lower:
        return "wfh_easy_money"
    return "other_scam"


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic job scam dataset")
    parser.add_argument("--samples", type=int, default=3000, help="Total job postings")
    parser.add_argument("--fraud-rate", type=float, default=0.4, help="Fraction of scams (0-1)")
    parser.add_argument("--output", type=str, default="data/job_scams.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_scam = int(args.samples * args.fraud_rate)
    n_legit = args.samples - n_scam

    data = []

    # Legitimate jobs
    for _ in range(n_legit):
        template = random.choice(LEGIT_JOB_TEMPLATES)
        text = fill_legit(template)
        data.append({"text": text, "label": 0, "type": "legitimate"})

    # Scam jobs
    for _ in range(n_scam):
        template = random.choice(SCAM_JOB_TEMPLATES)
        text = fill_scam(template)
        stype = categorize_scam(text)
        data.append({"text": text, "label": 1, "type": stype})

    random.shuffle(data)
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)

    print(f"Saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Scam rate: {df['label'].mean():.3f}")
    print(f"\nType distribution:")
    print(df['type'].value_counts())
    print(f"\nSample legitimate:")
    print(df[df['label']==0]['text'].iloc[0][:200])
    print(f"\nSample scam:")
    print(df[df['label']==1]['text'].iloc[0][:200])


if __name__ == "__main__":
    main()