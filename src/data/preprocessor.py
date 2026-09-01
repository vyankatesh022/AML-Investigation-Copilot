"""Feature engineering and preprocessing utilities for transaction data."""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd

from src.config import settings


class TransactionPreprocessor:
    """Extracts derived features and cleans transaction data."""

    @staticmethod
    def extract_features(tx: Dict[str, Any], history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enriches a single transaction dictionary with analytical features."""
        enriched = dict(tx)

        # Parse timestamp
        dt = datetime.fromisoformat(str(tx.get("timestamp", "2026-01-01T00:00:00")))
        enriched["hour_of_day"] = dt.hour
        enriched["day_of_week"] = dt.weekday()
        enriched["is_night_transaction"] = int(dt.hour >= 23 or dt.hour <= 5)

        # Balance dynamics
        amount = float(tx.get("amount", 0.0))
        old_orig = float(tx.get("oldbalanceOrg", 0.0))
        new_orig = float(tx.get("newbalanceOrig", 0.0))
        old_dest = float(tx.get("oldbalanceDest", 0.0))
        new_dest = float(tx.get("newbalanceDest", 0.0))

        # Ratio of transaction amount to originator starting balance
        enriched["amount_to_balance_ratio"] = (
            round(amount / (old_orig + 1.0), 4) if old_orig >= 0 else 0.0
        )

        # Account drain ratio: how much of the balance was emptied out
        balance_drop = max(0.0, old_orig - new_orig)
        enriched["balance_drain_ratio"] = (
            round(balance_drop / (old_orig + 1.0), 4) if old_orig > 0 else 0.0
        )

        # Destination balance change discrepancy
        dest_increase = max(0.0, new_dest - old_dest)
        enriched["dest_balance_discrepancy"] = round(abs(amount - dest_increase), 2)

        # Country risk check
        country = str(tx.get("counterparty_country", "USA")).upper()
        enriched["is_high_risk_country"] = int(country in settings.HIGH_RISK_JURISDICTIONS)

        # Historical behavior deviation (if history provided)
        if history and len(history) > 0:
            amounts = [float(h.get("amount", 0.0)) for h in history]
            avg_amount = sum(amounts) / len(amounts)
            enriched["historical_avg_amount"] = round(avg_amount, 2)
            enriched["amount_vs_history_ratio"] = (
                round(amount / (avg_amount + 1.0), 2) if avg_amount > 0 else 1.0
            )
            enriched["recent_velocity_count"] = len(history)
        else:
            enriched["historical_avg_amount"] = amount
            enriched["amount_vs_history_ratio"] = 1.0
            enriched["recent_velocity_count"] = 1

        return enriched

    @classmethod
    def preprocess_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a DataFrame of raw transactions into an enriched feature DataFrame."""
        rows = [cls.extract_features(row) for row in df.to_dict(orient="records")]
        return pd.DataFrame(rows)
