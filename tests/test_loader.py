"""Unit tests for transaction data loaders, customer store, and preprocessors."""

import pytest
import pandas as pd
from pydantic import ValidationError

from src.data.loader import (
    TransactionDataLoader,
    CustomerDataLoader,
    TransactionRecord,
    CustomerProfile,
)
from src.data.preprocessor import TransactionPreprocessor


def test_transactions_csv_loading(tx_loader):
    """Transaction loader must successfully read records from CSV."""
    records = tx_loader.get_all_transactions()
    assert len(records) >= 20
    first = records[0]
    assert "transaction_id" in first
    assert "amount" in first
    assert "oldbalanceOrg" in first
    assert tx_loader.get_transaction_count() == len(records)


def test_missing_dataset_file_raises_error(tmp_path):
    """Loader raises FileNotFoundError with helpful message when dataset is missing."""
    non_existent_file = tmp_path / "missing_transactions.csv"
    loader = TransactionDataLoader(data_path=non_existent_file)
    with pytest.raises(FileNotFoundError, match="Transaction dataset file not found"):
        loader.load_data()


def test_missing_required_columns_raises_error(tmp_path):
    """Loader raises ValueError when CSV lacks required columns."""
    bad_csv = tmp_path / "bad_transactions.csv"
    bad_df = pd.DataFrame([{"transaction_id": "TX-1", "amount": 100.0}])
    bad_df.to_csv(bad_csv, index=False)

    loader = TransactionDataLoader(data_path=bad_csv)
    with pytest.raises(ValueError, match="missing required columns"):
        loader.load_data()


def test_duplicate_transactions_deduplicated(tmp_path):
    """Duplicate transaction IDs are removed, keeping the first occurrence."""
    csv_path = tmp_path / "duplicates.csv"
    df = pd.DataFrame([
        {
            "transaction_id": "TX-DUP-1",
            "timestamp": "2026-09-01T10:00:00",
            "customer_id": "CUST-1",
            "counterparty_id": "CUST-2",
            "transaction_type": "PAYMENT",
            "amount": 100.0,
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 400.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1100.0,
        },
        {
            "transaction_id": "TX-DUP-1",  # Duplicate ID
            "timestamp": "2026-09-01T10:05:00",
            "customer_id": "CUST-1",
            "counterparty_id": "CUST-2",
            "transaction_type": "PAYMENT",
            "amount": 200.0,
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 300.0,
            "oldbalanceDest": 1100.0,
            "newbalanceDest": 1300.0,
        },
    ])
    df.to_csv(csv_path, index=False)

    loader = TransactionDataLoader(data_path=csv_path)
    loaded_df = loader.load_data()
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["amount"] == 100.0  # Kept first


def test_invalid_amount_filtered(tmp_path):
    """Non-positive amounts (<= 0) are dropped during loading."""
    csv_path = tmp_path / "invalid_amount.csv"
    df = pd.DataFrame([
        {
            "transaction_id": "TX-ZERO",
            "timestamp": "2026-09-01T10:00:00",
            "customer_id": "CUST-1",
            "counterparty_id": "CUST-2",
            "transaction_type": "PAYMENT",
            "amount": 0.0,  # Invalid zero amount
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 500.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1000.0,
        },
        {
            "transaction_id": "TX-VALID",
            "timestamp": "2026-09-01T10:05:00",
            "customer_id": "CUST-1",
            "counterparty_id": "CUST-2",
            "transaction_type": "PAYMENT",
            "amount": 50.0,
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 450.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1050.0,
        },
    ])
    df.to_csv(csv_path, index=False)

    loader = TransactionDataLoader(data_path=csv_path)
    loaded_df = loader.load_data()
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["transaction_id"] == "TX-VALID"


def test_get_transaction_by_id(tx_loader):
    """Querying a specific transaction returns validated dictionary."""
    tx = tx_loader.get_transaction_by_id("TX-1003")
    assert tx is not None
    assert tx["transaction_id"] == "TX-1003"
    assert tx["amount"] == 9500.00
    assert tx["customer_id"] == "CUST-103"


def test_missing_transaction_returns_none(tx_loader):
    """Querying a non-existent transaction returns None."""
    tx = tx_loader.get_transaction_by_id("TX-DOES-NOT-EXIST")
    assert tx is None


def test_get_customer_transactions_and_history(tx_loader):
    """Retrieving transactions for a customer returns sorted history."""
    # CUST-103 has TX-1003 and TX-1004
    history = tx_loader.get_transactions_for_customer("CUST-103")
    assert len(history) >= 2
    # Verify sorted by timestamp descending
    assert history[0]["timestamp"] >= history[1]["timestamp"]

    # Test limit parameter
    limited = tx_loader.get_customer_history("CUST-103", limit=1)
    assert len(limited) == 1


def test_non_existent_customer_transactions(tx_loader):
    """Querying transactions for unknown customer returns empty list."""
    history = tx_loader.get_transactions_for_customer("CUST-UNKNOWN-999")
    assert history == []


def test_customer_loader_profile_lookup(customer_loader):
    """Customer loader fetches existing profile with correct KYC tier."""
    cust = customer_loader.get_customer_by_id("CUST-101")
    assert cust is not None
    assert cust["customer_id"] == "CUST-101"
    assert cust["full_name"] == "Alice Johnson"
    assert cust["kyc_risk_tier"] == "LOW"
    assert cust["pep_status"] is False


def test_missing_customer_returns_none(customer_loader):
    """Non-existent customer ID returns None."""
    cust = customer_loader.get_customer_by_id("CUST-NONEXISTENT")
    assert cust is None


def test_missing_customer_dataset_file(tmp_path):
    """Missing customer store raises FileNotFoundError."""
    missing_json = tmp_path / "missing_customers.json"
    loader = CustomerDataLoader(data_path=missing_json)
    with pytest.raises(FileNotFoundError, match="Customer dataset file not found"):
        loader.load_data()


def test_pydantic_invalid_timestamp_validation(clean_transaction):
    """Pydantic raises validation error for malformed timestamp."""
    bad_tx = dict(clean_transaction)
    bad_tx["timestamp"] = "not-a-timestamp"
    with pytest.raises(ValidationError):
        TransactionRecord(**bad_tx)


def test_preprocessor_feature_extraction(clean_transaction):
    """Feature preprocessor computes balance ratios and night flags."""
    enriched = TransactionPreprocessor.extract_features(clean_transaction)
    assert "amount_to_balance_ratio" in enriched
    assert "balance_drain_ratio" in enriched
    assert "is_night_transaction" in enriched
    assert "is_high_risk_country" in enriched

    # clean_transaction has amount 45.50 and oldbalanceOrg 3500.0
    expected_ratio = round(45.50 / (3500.00 + 1.0), 4)
    assert enriched["amount_to_balance_ratio"] == expected_ratio
    assert enriched["is_high_risk_country"] == 0
