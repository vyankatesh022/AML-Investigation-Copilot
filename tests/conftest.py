"""Shared test fixtures for the FinGuard AI test suite."""

import pytest
from src.data.loader import TransactionDataLoader, CustomerDataLoader
from src.risk_engine.rules import RuleEngine


@pytest.fixture
def clean_transaction():
    return {
        "transaction_id": "TX-TEST-CLEAN",
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
    }


@pytest.fixture
def structuring_transaction():
    return {
        "transaction_id": "TX-TEST-STRUCT",
        "timestamp": "2026-09-01T14:10:00",
        "customer_id": "CUST-103",
        "counterparty_id": "CUST-204",
        "transaction_type": "TRANSFER",
        "amount": 9950.00,
        "oldbalanceOrg": 12000.00,
        "newbalanceOrig": 2050.00,
        "oldbalanceDest": 500.00,
        "newbalanceDest": 10450.00,
        "counterparty_country": "USA",
        "channel": "BRANCH",
        "is_suspicious_ground_truth": 1,
    }


@pytest.fixture
def high_amount_transaction():
    return {
        "transaction_id": "TX-TEST-HIGH",
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
    }


@pytest.fixture
def rapid_drain_transaction():
    return {
        "transaction_id": "TX-TEST-DRAIN",
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
    }


@pytest.fixture
def sanctioned_country_transaction():
    return {
        "transaction_id": "TX-TEST-SANCTION",
        "timestamp": "2026-09-02T16:40:00",
        "customer_id": "CUST-107",
        "counterparty_id": "CUST-208",
        "transaction_type": "WIRE",
        "amount": 8500.00,
        "oldbalanceOrg": 15000.00,
        "newbalanceOrig": 6500.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 8500.00,
        "counterparty_country": "IRN",
        "channel": "ONLINE_BANKING",
        "is_suspicious_ground_truth": 1,
    }


@pytest.fixture
def sample_customer():
    return {
        "customer_id": "CUST-101",
        "full_name": "Alice Johnson",
        "occupation": "High School Teacher",
        "monthly_income": 4500.0,
        "expected_monthly_turnover": 5000.0,
        "kyc_risk_tier": "LOW",
        "pep_status": False,
        "country_of_residence": "USA",
        "account_open_date": "2021-03-15",
    }


@pytest.fixture
def rule_engine():
    return RuleEngine()


@pytest.fixture
def tx_loader():
    return TransactionDataLoader()


@pytest.fixture
def customer_loader():
    return CustomerDataLoader()
