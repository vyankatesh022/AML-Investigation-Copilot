"""Unit tests for the combined AML risk assessment engine."""

from src.risk_engine.assessor import TransactionRiskAssessor, CombinedRiskAssessment


def test_assessor_clean_transaction(clean_transaction, sample_customer):
    """Clean transaction results in LOW risk and is_flagged=False."""
    assessor = TransactionRiskAssessor()
    assessment: CombinedRiskAssessment = assessor.assess_transaction(
        tx=clean_transaction, customer=sample_customer
    )

    assert assessment.final_risk_level == "LOW"
    assert assessment.final_risk_score < 0.40
    assert not assessment.is_flagged
    assert len(assessment.triggered_rules) == 0
    assert "LOW risk" in assessment.explanation


def test_assessor_structuring_transaction(structuring_transaction):
    """Structuring transaction results in HIGH risk and is_flagged=True."""
    assessor = TransactionRiskAssessor()
    assessment: CombinedRiskAssessment = assessor.assess_transaction(tx=structuring_transaction)

    assert assessment.final_risk_level == "HIGH"
    assert assessment.final_risk_score >= 0.70
    assert assessment.is_flagged
    assert any("Structuring" in r for r in assessment.triggered_rules)
    assert assessment.rule_score >= 0.80
    assert "flagged as HIGH risk" in assessment.explanation


def test_assessor_high_amount_transaction(high_amount_transaction):
    """High-value wire transaction receives HIGH risk and critical severity."""
    assessor = TransactionRiskAssessor()
    assessment: CombinedRiskAssessment = assessor.assess_transaction(tx=high_amount_transaction)

    assert assessment.final_risk_level == "HIGH"
    assert assessment.is_flagged
    assert assessment.rule_severity in ["HIGH", "CRITICAL"]
    assert any("High Value" in r for r in assessment.triggered_rules)


def test_assessor_sanctioned_jurisdiction_transaction(sanctioned_country_transaction):
    """Transaction to high-risk country triggers critical severity and high risk."""
    assessor = TransactionRiskAssessor()
    assessment: CombinedRiskAssessment = assessor.assess_transaction(tx=sanctioned_country_transaction)

    assert assessment.final_risk_level == "HIGH"
    assert assessment.final_risk_score >= 0.90
    assert assessment.is_flagged
    assert any("Sanctioned Jurisdiction" in r for r in assessment.triggered_rules)


def test_assessor_output_schema(clean_transaction):
    """Assessment output must strictly conform to Pydantic schema types."""
    assessor = TransactionRiskAssessor()
    assessment = assessor.assess_transaction(clean_transaction)

    dump = assessment.model_dump()
    assert isinstance(dump["transaction_id"], str)
    assert isinstance(dump["final_risk_score"], float)
    assert isinstance(dump["is_flagged"], bool)
    assert isinstance(dump["triggered_rules"], list)
    assert isinstance(dump["ml_probability"], float)
    assert isinstance(dump["explanation"], str)
