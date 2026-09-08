"""Generates synthetic AML transaction dataset and customer KYC store for testing and development."""

import json
from pathlib import Path
import pandas as pd

from src.config import settings
from src.risk_engine.ml_model import MLRiskClassifier


def generate_seed_customers() -> dict:
    """Generates 20 sample customer KYC demographic profiles."""
    customers = {
        "CUST-101": {
            "customer_id": "CUST-101",
            "full_name": "Alice Johnson",
            "occupation": "High School Teacher",
            "monthly_income": 4500.0,
            "expected_monthly_turnover": 5000.0,
            "kyc_risk_tier": "LOW",
            "pep_status": False,
            "country_of_residence": "USA",
            "account_open_date": "2021-03-15",
        },
        "CUST-102": {
            "customer_id": "CUST-102",
            "full_name": "Brian Miller",
            "occupation": "Accountant",
            "monthly_income": 7200.0,
            "expected_monthly_turnover": 7500.0,
            "kyc_risk_tier": "LOW",
            "pep_status": False,
            "country_of_residence": "USA",
            "account_open_date": "2020-01-22",
        },
        "CUST-103": {
            "customer_id": "CUST-103",
            "full_name": "Robert Davis",
            "occupation": "Retail Store Manager",
            "monthly_income": 6500.0,
            "expected_monthly_turnover": 8000.0,
            "kyc_risk_tier": "MEDIUM",
            "pep_status": False,
            "country_of_residence": "USA",
            "account_open_date": "2020-05-10",
        },
        "CUST-104": {
            "customer_id": "CUST-104",
            "full_name": "Marcus Sterling",
            "occupation": "Commercial Real Estate Broker",
            "monthly_income": 25000.0,
            "expected_monthly_turnover": 50000.0,
            "kyc_risk_tier": "MEDIUM",
            "pep_status": False,
            "country_of_residence": "USA",
            "account_open_date": "2018-09-12",
        },
        "CUST-105": {
            "customer_id": "CUST-105",
            "full_name": "Elena Rostova",
            "occupation": "Independent Contractor",
            "monthly_income": 5000.0,
            "expected_monthly_turnover": 6000.0,
            "kyc_risk_tier": "HIGH",
            "pep_status": False,
            "country_of_residence": "USA",
            "account_open_date": "2022-04-01",
        },
        "CUST-106": {
            "customer_id": "CUST-106",
            "full_name": "Sophia Martinez",
            "occupation": "Graphic Designer",
            "monthly_income": 5800.0,
            "expected_monthly_turnover": 6000.0,
            "kyc_risk_tier": "LOW",
            "pep_status": False,
            "country_of_residence": "USA",
            "account_open_date": "2021-08-19",
        },
        "CUST-107": {
            "customer_id": "CUST-107",
            "full_name": "Kaveh Rahimi",
            "occupation": "Import-Export Merchant",
            "monthly_income": 12000.0,
            "expected_monthly_turnover": 20000.0,
            "kyc_risk_tier": "HIGH",
            "pep_status": False,
            "country_of_residence": "USA",
            "account_open_date": "2019-11-20",
        },
    }

    # Populate CUST-108 through CUST-120
    occupations = [
        ("Nurse", 6200.0, "LOW", False),
        ("Software Developer", 11000.0, "LOW", False),
        ("Attorney", 16000.0, "MEDIUM", False),
        ("Restaurant Owner", 14000.0, "MEDIUM", False),
        ("Foreign Diplomat", 18000.0, "HIGH", True),
        ("Logistics Coordinator", 5500.0, "LOW", False),
        ("E-commerce Seller", 9000.0, "MEDIUM", False),
        ("Civil Engineer", 8500.0, "LOW", False),
        ("Financial Consultant", 15000.0, "MEDIUM", False),
        ("Art Dealer", 22000.0, "HIGH", False),
        ("Construction Supervisor", 7500.0, "LOW", False),
        ("Auto Dealer", 13000.0, "MEDIUM", False),
        ("Government Official", 9500.0, "HIGH", True),
    ]

    for idx, (occ, inc, tier, pep) in enumerate(occupations, start=108):
        cid = f"CUST-{idx}"
        customers[cid] = {
            "customer_id": cid,
            "full_name": f"Customer {idx} Fullname",
            "occupation": occ,
            "monthly_income": inc,
            "expected_monthly_turnover": inc * 1.2,
            "kyc_risk_tier": tier,
            "pep_status": pep,
            "country_of_residence": "USA",
            "account_open_date": f"202{idx % 4}-0{(idx % 9) + 1:02d}-15",
        }

    return customers


def generate_seed_transactions() -> list:
    """Generates 100 synthetic transaction records with known AML patterns."""
    txs = []

    # 1. TX-1001: Clean low-risk payment for Alice Johnson (CUST-101)
    txs.append({
        "transaction_id": "TX-1001",
        "timestamp": "2026-09-01T12:00:00",
        "customer_id": "CUST-101",
        "counterparty_id": "CUST-201",
        "transaction_type": "PAYMENT",
        "amount": 45.50,
        "oldbalanceOrg": 3500.00,
        "newbalanceOrig": 3454.50,
        "oldbalanceDest": 12000.00,
        "newbalanceDest": 12045.50,
        "counterparty_country": "USA",
        "channel": "MOBILE_APP",
        "is_suspicious_ground_truth": 0,
    })

    # 2. TX-1002: Another clean transaction for CUST-102
    txs.append({
        "transaction_id": "TX-1002",
        "timestamp": "2026-09-01T13:00:00",
        "customer_id": "CUST-102",
        "counterparty_id": "CUST-202",
        "transaction_type": "PAYMENT",
        "amount": 120.00,
        "oldbalanceOrg": 5000.00,
        "newbalanceOrig": 4880.00,
        "oldbalanceDest": 8000.00,
        "newbalanceDest": 8120.00,
        "counterparty_country": "USA",
        "channel": "ONLINE_BANKING",
        "is_suspicious_ground_truth": 0,
    })

    # 3. TX-1003: Structuring transaction for Robert Davis (CUST-103)
    txs.append({
        "transaction_id": "TX-1003",
        "timestamp": "2026-09-01T14:10:00",
        "customer_id": "CUST-103",
        "counterparty_id": "CUST-204",
        "transaction_type": "TRANSFER",
        "amount": 9500.00,
        "oldbalanceOrg": 12000.00,
        "newbalanceOrig": 2500.00,
        "oldbalanceDest": 500.00,
        "newbalanceDest": 10000.00,
        "counterparty_country": "USA",
        "channel": "BRANCH",
        "is_suspicious_ground_truth": 1,
    })

    # 4. TX-1004: Prior structuring transaction for CUST-103 (history testing)
    txs.append({
        "transaction_id": "TX-1004",
        "timestamp": "2026-08-31T09:30:00",
        "customer_id": "CUST-103",
        "counterparty_id": "CUST-204",
        "transaction_type": "TRANSFER",
        "amount": 9600.00,
        "oldbalanceOrg": 21600.00,
        "newbalanceOrig": 12000.00,
        "oldbalanceDest": 100.00,
        "newbalanceDest": 9700.00,
        "counterparty_country": "USA",
        "channel": "BRANCH",
        "is_suspicious_ground_truth": 1,
    })

    # 5. TX-1005: High-value wire transfer for CUST-104
    txs.append({
        "transaction_id": "TX-1005",
        "timestamp": "2026-09-02T10:00:00",
        "customer_id": "CUST-104",
        "counterparty_id": "CUST-205",
        "transaction_type": "WIRE",
        "amount": 150000.00,
        "oldbalanceOrg": 300000.00,
        "newbalanceOrig": 150000.00,
        "oldbalanceDest": 10000.00,
        "newbalanceDest": 160000.00,
        "counterparty_country": "GBR",
        "channel": "WIRE_DESK",
        "is_suspicious_ground_truth": 1,
    })

    # 6. TX-1006: Rapid account balance drain for CUST-105
    txs.append({
        "transaction_id": "TX-1006",
        "timestamp": "2026-09-02T13:20:00",
        "customer_id": "CUST-105",
        "counterparty_id": "CUST-206",
        "transaction_type": "CASH_OUT",
        "amount": 49000.00,
        "oldbalanceOrg": 50000.00,
        "newbalanceOrig": 1000.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 49000.00,
        "counterparty_country": "USA",
        "channel": "ATM",
        "is_suspicious_ground_truth": 1,
    })

    # 7. TX-1007: Normal retail transaction for CUST-106
    txs.append({
        "transaction_id": "TX-1007",
        "timestamp": "2026-09-02T15:00:00",
        "customer_id": "CUST-106",
        "counterparty_id": "CUST-207",
        "transaction_type": "PAYMENT",
        "amount": 85.00,
        "oldbalanceOrg": 4200.00,
        "newbalanceOrig": 4115.00,
        "oldbalanceDest": 1500.00,
        "newbalanceDest": 1585.00,
        "counterparty_country": "USA",
        "channel": "POS",
        "is_suspicious_ground_truth": 0,
    })

    # 8. TX-1008: Sanctioned country wire for Kaveh Rahimi (CUST-107)
    txs.append({
        "transaction_id": "TX-1008",
        "timestamp": "2026-09-02T16:40:00",
        "customer_id": "CUST-107",
        "counterparty_id": "CUST-208",
        "transaction_type": "WIRE",
        "amount": 15000.00,
        "oldbalanceOrg": 30000.00,
        "newbalanceOrig": 15000.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 15000.00,
        "counterparty_country": "IRN",
        "channel": "WIRE_DESK",
        "is_suspicious_ground_truth": 1,
    })

    # 9. TX-1009 to TX-1100: Generated mix of normal transactions and realistic typologies
    channels = ["MOBILE_APP", "ONLINE_BANKING", "BRANCH", "ATM", "WIRE_DESK", "POS"]
    tx_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "WIRE"]
    countries = ["USA", "USA", "USA", "CAN", "GBR", "DEU", "FRA", "CYM"]

    for i in range(1009, 1101):
        tx_id = f"TX-{i}"
        cust_num = 101 + ((i - 1009) % 20)
        cust_id = f"CUST-{cust_num}"
        cp_id = f"CUST-{(cust_num + 50)}"

        # Make some transactions suspicious and others normal
        is_suspicious = 1 if (i % 5 == 0 or i % 7 == 0) else 0

        if is_suspicious:
            if i % 3 == 0:
                # Structuring
                amt = 9200.00 + ((i % 7) * 110.00)
                ttype = "TRANSFER"
                country = "USA"
            elif i % 3 == 1:
                # High Value
                amt = 25000.00 + ((i % 5) * 5000.00)
                ttype = "WIRE"
                country = "CYM" if i % 2 == 0 else "USA"
            else:
                # Rapid Drain
                amt = 18000.00
                ttype = "CASH_OUT"
                country = "USA"
        else:
            amt = round(15.00 + ((i * 17) % 450) + ((i % 10) * 0.5), 2)
            ttype = tx_types[i % len(tx_types)]
            country = countries[i % len(countries)]

        start_bal = max(amt * 1.5, 3000.00 + (i * 20.0))
        end_bal = start_bal - amt if ttype in ["PAYMENT", "TRANSFER", "CASH_OUT", "WIRE"] else start_bal + amt

        day = 1 + ((i - 1000) // 15)
        hour = 8 + (i % 12)
        minute = (i * 7) % 60
        ts = f"2026-09-{day:02d}T{hour:02d}:{minute:02d}:00"

        txs.append({
            "transaction_id": tx_id,
            "timestamp": ts,
            "customer_id": cust_id,
            "counterparty_id": cp_id,
            "transaction_type": ttype,
            "amount": round(amt, 2),
            "oldbalanceOrg": round(start_bal, 2),
            "newbalanceOrig": round(end_bal, 2),
            "oldbalanceDest": 1000.00,
            "newbalanceDest": round(1000.00 + amt, 2),
            "counterparty_country": country,
            "channel": channels[i % len(channels)],
            "is_suspicious_ground_truth": is_suspicious,
        })

    return txs


def seed_all() -> None:
    """Creates synthetic datasets on disk and trains the Random Forest model."""
    raw_dir = settings.DATA_DIR / "raw"
    processed_dir = settings.DATA_DIR / "processed"
    models_dir = settings.BASE_DIR / "models"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Seed Customer KYC profiles
    customers = generate_seed_customers()
    with open(settings.CUSTOMERS_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(customers, f, indent=2)
    print(f"Created customer store: {settings.CUSTOMERS_DATA_PATH} ({len(customers)} records)")

    # 2. Seed Transaction CSV
    txs = generate_seed_transactions()
    df = pd.DataFrame(txs)
    df.to_csv(settings.RAW_DATA_PATH, index=False)
    print(f"Created transaction CSV: {settings.RAW_DATA_PATH} ({len(df)} records)")

    # 3. Train and persist baseline ML model
    classifier = MLRiskClassifier(model_path=models_dir / "risk_classifier.joblib")
    metrics = classifier.train(df)
    print(f"Trained ML Risk Classifier: {classifier.model_path}")
    print(f"Metrics: Accuracy={metrics['accuracy']}, Precision={metrics['precision']}, Recall={metrics['recall']}, F1={metrics['f1_score']}")


if __name__ == "__main__":
    seed_all()
