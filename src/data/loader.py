"""Data loader utilities for transactions and customer profiles."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

from src.config import settings


class TransactionRecord(BaseModel):
    """Schema for a single financial transaction record."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: str = Field(..., description="ISO 8601 transaction timestamp")
    customer_id: str = Field(..., description="Originator / Sender customer identifier")
    counterparty_id: str = Field(..., description="Beneficiary / Receiver customer identifier")
    transaction_type: str = Field(..., description="Type of transaction, e.g. TRANSFER, CASH_OUT")
    amount: float = Field(..., ge=0.0, description="Monetary amount of transaction")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Sender balance prior to transaction")
    newbalanceOrig: float = Field(..., ge=0.0, description="Sender balance after transaction")
    oldbalanceDest: float = Field(..., ge=0.0, description="Recipient balance prior to transaction")
    newbalanceDest: float = Field(..., ge=0.0, description="Recipient balance after transaction")
    counterparty_country: str = Field(default="USA", description="Country of recipient / destination")
    channel: str = Field(default="ONLINE_BANKING", description="Channel used to execute transaction")
    is_suspicious_ground_truth: int = Field(default=0, description="Binary ground truth label (0 or 1)")


class CustomerProfile(BaseModel):
    """Schema for a customer KYC and demographic profile."""

    customer_id: str
    full_name: str
    occupation: str
    monthly_income: float
    expected_monthly_turnover: float
    kyc_risk_tier: str  # LOW, MEDIUM, HIGH
    pep_status: bool
    country_of_residence: str
    account_open_date: str


class TransactionDataLoader:
    """Loads, validates, and queries financial transaction records."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path or settings.RAW_DATA_PATH)
        self._df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Reads CSV file into a pandas DataFrame."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Transaction dataset not found at: {self.data_path}")
        self._df = pd.read_csv(self.data_path)
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        """Cached property for DataFrame."""
        if self._df is None:
            return self.load_data()
        return self._df

    def get_all_transactions(self) -> List[Dict[str, Any]]:
        """Returns all transaction records as dictionaries."""
        return self.df.to_dict(orient="records")

    def get_transaction_by_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Finds a single transaction by its unique ID."""
        records = self.df[self.df["transaction_id"] == tx_id]
        if records.empty:
            return None
        row = records.iloc[0].to_dict()
        # Validate through Pydantic
        validated = TransactionRecord(**row)
        return validated.model_dump()

    def get_transactions_for_customer(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves historical transactions for a given customer."""
        records = self.df[self.df["customer_id"] == customer_id]
        if records.empty:
            return []
        sorted_records = records.sort_values(by="timestamp", ascending=False).head(limit)
        return sorted_records.to_dict(orient="records")


class CustomerDataLoader:
    """Loads and queries customer KYC profile data."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path or settings.CUSTOMERS_DATA_PATH)
        self._customers: Optional[Dict[str, Dict[str, Any]]] = None

    def load_data(self) -> Dict[str, Dict[str, Any]]:
        """Reads JSON customer store into memory."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Customer dataset not found at: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self._customers = json.load(f)
        return self._customers

    @property
    def customers(self) -> Dict[str, Dict[str, Any]]:
        """Cached property for customers dictionary."""
        if self._customers is None:
            return self.load_data()
        return self._customers

    def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves customer profile by ID."""
        raw_customer = self.customers.get(customer_id)
        if not raw_customer:
            return None
        validated = CustomerProfile(**raw_customer)
        return validated.model_dump()

    def get_all_customers(self) -> Dict[str, Dict[str, Any]]:
        """Returns all customer profiles."""
        return self.customers
