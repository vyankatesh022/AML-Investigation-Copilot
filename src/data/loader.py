"""Data loader utilities for loading, validating, and accessing AML transaction data and customer profiles."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.config import settings


# Mandatory columns that must exist in any transaction dataset
REQUIRED_TRANSACTION_COLUMNS: Set[str] = {
    "transaction_id",
    "timestamp",
    "customer_id",
    "counterparty_id",
    "transaction_type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
}


class TransactionRecord(BaseModel):
    """Schema for validating an individual transaction record."""

    transaction_id: str = Field(..., min_length=1, description="Unique transaction ID")
    timestamp: str = Field(..., description="ISO 8601 transaction timestamp")
    customer_id: str = Field(..., min_length=1, description="Originator customer ID")
    counterparty_id: str = Field(..., min_length=1, description="Beneficiary customer ID")
    transaction_type: str = Field(..., min_length=1, description="Transaction type, e.g. TRANSFER, CASH_OUT")
    amount: float = Field(..., gt=0.0, description="Transaction amount must be strictly greater than 0")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Sender balance prior to transaction")
    newbalanceOrig: float = Field(..., ge=0.0, description="Sender balance following transaction")
    oldbalanceDest: float = Field(..., ge=0.0, description="Recipient balance prior to transaction")
    newbalanceDest: float = Field(..., ge=0.0, description="Recipient balance following transaction")
    counterparty_country: str = Field(default="USA", description="Recipient country ISO code")
    channel: str = Field(default="ONLINE_BANKING", description="Channel used to execute transaction")
    is_suspicious_ground_truth: int = Field(default=0, description="Binary ground truth label (0 or 1)")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        """Ensures the timestamp follows ISO-8601 format."""
        try:
            datetime.fromisoformat(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid timestamp format '{v}'. Expected ISO format (YYYY-MM-DDTHH:MM:SS)") from exc
        return v


class CustomerProfile(BaseModel):
    """Schema for validating a customer KYC demographic profile."""

    customer_id: str = Field(..., min_length=1)
    full_name: str = Field(..., min_length=1)
    occupation: str = Field(..., min_length=1)
    monthly_income: float = Field(..., ge=0.0)
    expected_monthly_turnover: float = Field(..., ge=0.0)
    kyc_risk_tier: str = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")
    pep_status: bool = Field(default=False)
    country_of_residence: str = Field(default="USA")
    account_open_date: str = Field(...)


class TransactionDataLoader:
    """Loads, cleans, validates, and provides access to transaction datasets."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path or settings.RAW_DATA_PATH)
        self._df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads transaction CSV data, validates required columns, cleans fields, and removes duplicates."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Transaction dataset file not found at: {self.data_path}. "
                "Please verify that the file exists and the path is correct."
            )

        try:
            raw_df = pd.read_csv(self.data_path)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV dataset from {self.data_path}: {exc}") from exc

        # Check required columns
        missing_columns = REQUIRED_TRANSACTION_COLUMNS - set(raw_df.columns)
        if missing_columns:
            raise ValueError(
                f"Transaction dataset is missing required columns: {sorted(list(missing_columns))}"
            )

        # Basic cleaning: strip whitespace from string fields
        df = raw_df.copy()
        str_columns = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
        for col in str_columns:
            df[col] = df[col].astype(str).str.strip()

        # Ensure amounts and balances are numeric
        numeric_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where transaction_id, customer_id, or amount is missing / null
        initial_len = len(df)
        df = df.dropna(subset=["transaction_id", "customer_id", "amount", "timestamp"])

        # Validate that amount > 0
        df = df[df["amount"] > 0]

        # Handle duplicates: keep first occurrence of transaction_id
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")

        self._df = df.reset_index(drop=True)
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        """Cached property for the validated DataFrame."""
        if self._df is None:
            return self.load_data()
        return self._df

    def get_transaction_count(self) -> int:
        """Returns the total number of validated transactions in the dataset."""
        return len(self.df)

    def get_all_transactions(self) -> List[Dict[str, Any]]:
        """Returns all validated transaction records as dictionaries."""
        return self.df.to_dict(orient="records")

    def get_transaction_by_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves and validates a single transaction record by its transaction ID."""
        clean_id = str(tx_id).strip()
        records = self.df[self.df["transaction_id"] == clean_id]
        if records.empty:
            return None

        row = records.iloc[0].to_dict()
        validated = TransactionRecord(**row)
        return validated.model_dump()

    def get_transactions_for_customer(self, customer_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves all historical transactions for a given customer ID, sorted by timestamp descending."""
        clean_cust_id = str(customer_id).strip()
        records = self.df[self.df["customer_id"] == clean_cust_id]
        if records.empty:
            return []

        sorted_records = records.sort_values(by="timestamp", ascending=False)
        if limit and limit > 0:
            sorted_records = sorted_records.head(limit)

        return sorted_records.to_dict(orient="records")

    def get_customer_history(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Convenience method for retrieving limited recent transaction history for an account."""
        return self.get_transactions_for_customer(customer_id=customer_id, limit=limit)


class CustomerDataLoader:
    """Loads, validates, and provides access to customer KYC profiles."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path or settings.CUSTOMERS_DATA_PATH)
        self._customers: Optional[Dict[str, Dict[str, Any]]] = None

    def load_data(self) -> Dict[str, Dict[str, Any]]:
        """Loads and parses the customer JSON store."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Customer dataset file not found at: {self.data_path}. "
                "Please verify that the file exists and the path is correct."
            )

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to parse customer JSON data: {exc}") from exc

        if not isinstance(raw_data, dict):
            raise ValueError("Customer dataset must be a JSON dictionary of customer_id -> profile")

        self._customers = raw_data
        return self._customers

    @property
    def customers(self) -> Dict[str, Dict[str, Any]]:
        """Cached property for the customer dictionary."""
        if self._customers is None:
            return self.load_data()
        return self._customers

    def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves and validates a customer profile by ID."""
        clean_id = str(customer_id).strip()
        raw_customer = self.customers.get(clean_id)
        if not raw_customer:
            return None

        validated = CustomerProfile(**raw_customer)
        return validated.model_dump()

    def get_all_customers(self) -> Dict[str, Dict[str, Any]]:
        """Returns all customer profiles in the store."""
        return self.customers
