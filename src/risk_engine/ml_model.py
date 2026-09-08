"""Machine learning risk classification model for AML transaction monitoring."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import settings
from src.data.preprocessor import TransactionPreprocessor

# Defined feature columns used as inputs for the ML model
MODEL_FEATURE_COLUMNS: List[str] = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "amount_to_balance_ratio",
    "balance_drain_ratio",
    "dest_balance_discrepancy",
    "is_high_risk_country",
    "is_night_transaction",
    "is_transfer",
    "is_cash_out",
    "is_wire",
    "is_payment",
    "is_deposit",
]


class MLExplanation(BaseModel):
    """Structured explainability and transparency metadata for machine learning risk inference."""

    model_name: str = Field(default="RandomForestClassifier (Tabular AML Risk Classifier)")
    model_version: str = Field(default="1.0.0")
    anomaly_probability: float = Field(..., description="Estimated statistical probability of anomaly (0.0 to 1.0)")
    prediction_label: str = Field(..., description="'Anomalous' or 'Normal'")
    risk_level: str = Field(..., description="'LOW', 'MEDIUM', or 'HIGH'")
    is_anomalous: bool = Field(..., description="Whether anomaly probability meets or exceeds 0.50 threshold")
    input_features: Dict[str, float] = Field(default_factory=dict, description="Complete 15-feature numeric vector evaluated")
    top_signals: List[str] = Field(default_factory=list, description="Top feature signals ranked by model importance")
    signal_details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed feature names, observed values, and importances")
    risk_interpretation: str = Field(..., description="Contextual interpretation distinguishing statistical anomaly from confirmed crime")
    known_limitations: str = Field(
        default=(
            "Supervised Random Forest trained on synthetic tabular AML features. An anomaly score "
            "quantifies deviation from normal baseline distributions and does NOT confirm money laundering "
            "or financial crime. Network and multi-hop mule account detection requires graph analysis."
        )
    )


class MLRiskClassifier:
    """Trains, saves, loads, and executes Scikit-Learn risk classification for transactions."""

    def __init__(self, model_path: Optional[Path] = None):
        self.models_dir = settings.BASE_DIR / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path or (self.models_dir / "risk_classifier.joblib")
        self.model: Optional[RandomForestClassifier] = None
        self.feature_columns = MODEL_FEATURE_COLUMNS

    @classmethod
    def extract_model_features(cls, tx: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Transforms a raw transaction dictionary into the exact numeric feature set required by the model."""
        enriched = TransactionPreprocessor.extract_features(tx, history=history)
        tx_type = str(tx.get("transaction_type", "PAYMENT")).upper()

        feature_dict = {
            "amount": float(enriched.get("amount", 0.0)),
            "oldbalanceOrg": float(enriched.get("oldbalanceOrg", 0.0)),
            "newbalanceOrig": float(enriched.get("newbalanceOrig", 0.0)),
            "oldbalanceDest": float(enriched.get("oldbalanceDest", 0.0)),
            "newbalanceDest": float(enriched.get("newbalanceDest", 0.0)),
            "amount_to_balance_ratio": float(enriched.get("amount_to_balance_ratio", 0.0)),
            "balance_drain_ratio": float(enriched.get("balance_drain_ratio", 0.0)),
            "dest_balance_discrepancy": float(enriched.get("dest_balance_discrepancy", 0.0)),
            "is_high_risk_country": float(enriched.get("is_high_risk_country", 0)),
            "is_night_transaction": float(enriched.get("is_night_transaction", 0)),
            "is_transfer": 1.0 if tx_type == "TRANSFER" else 0.0,
            "is_cash_out": 1.0 if tx_type == "CASH_OUT" else 0.0,
            "is_wire": 1.0 if tx_type == "WIRE" else 0.0,
            "is_payment": 1.0 if tx_type == "PAYMENT" else 0.0,
            "is_deposit": 1.0 if tx_type == "DEPOSIT" else 0.0,
        }
        return feature_dict

    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extracts the feature matrix X and target vector y from a transaction DataFrame."""
        rows = [self.extract_model_features(row) for row in df.to_dict(orient="records")]
        X = pd.DataFrame(rows)[self.feature_columns]
        y = df["is_suspicious_ground_truth"].astype(int)
        return X, y

    def train(self, df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
        """Trains the Random Forest model and calculates validation metrics."""
        X, y = self.prepare_dataset(df)

        if len(df) < 10:
            raise ValueError(f"Dataset too small ({len(df)} records) for training. Minimum 10 records required.")

        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
        )

        clf = RandomForestClassifier(
            n_estimators=40,
            max_depth=4,
            min_samples_split=2,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # Predictions on test split
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        self.model = clf
        self.save_model()
        return metrics

    def save_model(self) -> None:
        """Serializes the trained model to disk."""
        if self.model is None:
            raise ValueError("No model has been trained to save.")
        joblib.dump(self.model, self.model_path)

    def load_model(self) -> RandomForestClassifier:
        """Loads a pre-trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")
        self.model = joblib.load(self.model_path)
        return self.model

    def ensure_model_loaded(self, fallback_df: Optional[pd.DataFrame] = None) -> None:
        """Loads existing model, or trains a new one if not yet created."""
        if self.model is not None:
            return

        if self.model_path.exists():
            self.load_model()
        elif fallback_df is not None:
            self.train(fallback_df)
        else:
            raise FileNotFoundError(
                f"Model not found at {self.model_path} and no training data provided to initialize."
            )

    def get_explanation(
        self,
        tx: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> MLExplanation:
        """Generates a structured, transparent explainability object for model inference."""
        if self.model is None:
            raise ValueError("Model must be loaded or trained before generating explanations.")

        feat_dict = self.extract_model_features(tx, history=history)
        X_single = pd.DataFrame([feat_dict])[self.feature_columns]

        proba = float(self.model.predict_proba(X_single)[0][1])
        prediction = int(self.model.predict(X_single)[0])

        if proba >= 0.70:
            risk_level = "HIGH"
            interpretation = (
                f"High statistical anomaly detected ({proba * 100:.1f}% probability). "
                "Feature patterns diverge significantly from typical account baseline behaviors."
            )
        elif proba >= 0.40:
            risk_level = "MEDIUM"
            interpretation = (
                f"Moderate anomaly score ({proba * 100:.1f}% probability). "
                "Certain transaction features show mild deviation from median retail patterns."
            )
        else:
            risk_level = "LOW"
            interpretation = (
                f"Low statistical anomaly score ({proba * 100:.1f}% probability). "
                "Observed transaction features align with normal consumer banking distributions."
            )

        # Feature importances for explainability
        importances = dict(zip(self.feature_columns, self.model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]

        top_signals_str = [f"{name} (importance: {round(val, 3)})" for name, val in top_features]
        signal_details = [
            {
                "feature_name": name,
                "observed_value": feat_dict.get(name),
                "feature_importance": round(val, 4),
            }
            for name, val in top_features
        ]

        return MLExplanation(
            anomaly_probability=round(proba, 4),
            prediction_label="Anomalous" if prediction == 1 else "Normal",
            risk_level=risk_level,
            is_anomalous=bool(proba >= 0.50),
            input_features=feat_dict,
            top_signals=top_signals_str,
            signal_details=signal_details,
            risk_interpretation=interpretation,
        )

    def predict_risk(self, tx: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Predicts the risk probability, risk level, and transparent explainability for a single transaction."""
        explanation = self.get_explanation(tx=tx, history=history)

        return {
            "ml_risk_probability": explanation.anomaly_probability,
            "ml_prediction": 1 if explanation.prediction_label == "Anomalous" else 0,
            "ml_risk_level": explanation.risk_level,
            "ml_is_anomalous": explanation.is_anomalous,
            "top_model_signals": explanation.top_signals,
            "ml_explanation": explanation.model_dump(),
        }
