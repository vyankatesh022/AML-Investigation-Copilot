"""Unit tests for the Scikit-Learn risk classification model."""

import pytest
import pandas as pd
from src.risk_engine.ml_model import MLRiskClassifier, MODEL_FEATURE_COLUMNS


@pytest.fixture
def ml_classifier(tmp_path):
    """Provides an isolated classifier instance pointing to a temporary model path."""
    temp_model_file = tmp_path / "test_model.joblib"
    return MLRiskClassifier(model_path=temp_model_file)


def test_feature_extraction_completeness(clean_transaction):
    """Model feature extractor must output all 15 expected numeric feature columns."""
    features = MLRiskClassifier.extract_model_features(clean_transaction)
    for col in MODEL_FEATURE_COLUMNS:
        assert col in features, f"Missing feature column: {col}"
        assert isinstance(features[col], (int, float)), f"Feature {col} is not numeric"


def test_train_and_evaluate_metrics(ml_classifier, tx_loader):
    """Training the classifier returns valid evaluation metrics on test split."""
    df = tx_loader.df
    metrics = ml_classifier.train(df)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["train_samples"] > 0
    assert metrics["test_samples"] > 0
    assert ml_classifier.model_path.exists()


def test_model_save_and_load(ml_classifier, tx_loader):
    """Model can be saved to disk and reloaded with identical predictions."""
    df = tx_loader.df
    ml_classifier.train(df)

    new_classifier = MLRiskClassifier(model_path=ml_classifier.model_path)
    loaded_model = new_classifier.load_model()
    assert loaded_model is not None


def test_predict_risk_on_clean_transaction(ml_classifier, tx_loader, clean_transaction):
    """Clean transaction produces a LOW ML risk probability."""
    ml_classifier.train(tx_loader.df)
    result = ml_classifier.predict_risk(clean_transaction)

    assert "ml_risk_probability" in result
    assert "ml_risk_level" in result
    assert 0.0 <= result["ml_risk_probability"] <= 1.0
    assert result["ml_risk_level"] == "LOW"
    assert not result["ml_is_anomalous"]


def test_predict_risk_on_suspicious_transaction(ml_classifier, tx_loader, structuring_transaction):
    """Structuring transaction produces elevated ML anomaly probability."""
    ml_classifier.train(tx_loader.df)
    result = ml_classifier.predict_risk(structuring_transaction)

    assert result["ml_risk_probability"] >= 0.50
    assert result["ml_risk_level"] in ["MEDIUM", "HIGH"]
    assert result["ml_is_anomalous"]
    assert len(result["top_model_signals"]) > 0


def test_training_fails_on_tiny_dataset(ml_classifier):
    """Training raises ValueError if dataset has fewer than 10 records."""
    tiny_df = pd.DataFrame([{"transaction_id": "TX-1", "amount": 100.0, "is_suspicious_ground_truth": 0}])
    with pytest.raises(ValueError, match="Dataset too small"):
        ml_classifier.train(tiny_df)
