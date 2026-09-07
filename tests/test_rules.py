"""Unit tests for AML heuristic compliance rules and rule engine."""

import pytest
from src.risk_engine.rules import (
    RuleEngine,
    RuleHighAmount,
    RuleStructuring,
    RuleRapidDrain,
    RuleHighRiskJurisdiction,
    RuleVelocityAnomaly,
)


def test_clean_transaction_does_not_trigger(rule_engine, clean_transaction, sample_customer):
    """Clean payment below thresholds must not trigger any compliance rules."""
    summary = rule_engine.evaluate(tx=clean_transaction, customer=sample_customer)
    assert not summary.is_flagged
    assert summary.overall_rule_score == 0.0
    assert summary.overall_severity == "LOW"
    assert summary.triggered_rules_count == 0
    assert len(summary.triggered_rules) == 0


def test_high_amount_rule_triggers(high_amount_transaction):
    """Transactions meeting or exceeding $10,000 must trigger RuleHighAmount."""
    rule = RuleHighAmount()
    result = rule.evaluate(tx=high_amount_transaction)
    assert result.triggered
    assert result.severity == "CRITICAL"  # >= $100,000 is CRITICAL
    assert result.score >= 0.75
    assert "statutory threshold" in result.description


def test_high_amount_rule_does_not_trigger_on_lower_amount(clean_transaction):
    """Transactions below $10,000 must not trigger RuleHighAmount."""
    rule = RuleHighAmount()
    result = rule.evaluate(tx=clean_transaction)
    assert not result.triggered
    assert result.score == 0.0


def test_structuring_rule_triggers_in_corridor(structuring_transaction):
    """Amounts in $9,000 - $9,999.99 range must trigger RuleStructuring."""
    rule = RuleStructuring()
    result = rule.evaluate(tx=structuring_transaction)
    assert result.triggered
    assert result.severity == "HIGH"
    assert result.score == 0.85
    assert "structuring" in result.description.lower()


@pytest.mark.parametrize("test_amount,expected_trigger", [
    (8999.99, False),
    (9000.00, True),
    (9500.00, True),
    (9999.00, True),
    (10000.00, False),
    (250.00, False),
])
def test_structuring_corridor_boundaries(test_amount, expected_trigger, clean_transaction):
    """Structuring corridor boundaries must be strictly enforced."""
    rule = RuleStructuring()
    tx = dict(clean_transaction)
    tx["amount"] = test_amount
    result = rule.evaluate(tx=tx)
    assert result.triggered == expected_trigger


def test_rapid_drain_triggers_on_depletion(rapid_drain_transaction):
    """Accounts depleted of > 90% balance down to <= $1,000 must trigger RuleRapidDrain."""
    rule = RuleRapidDrain()
    result = rule.evaluate(tx=rapid_drain_transaction)
    assert result.triggered
    assert result.severity == "HIGH"
    assert result.score == 0.80
    assert result.details["drain_ratio"] >= 0.90


def test_high_risk_jurisdiction_triggers(sanctioned_country_transaction):
    """Transactions to FATF/OFAC high-risk countries must trigger RuleHighRiskJurisdiction."""
    rule = RuleHighRiskJurisdiction()
    result = rule.evaluate(tx=sanctioned_country_transaction)
    assert result.triggered
    assert result.severity == "CRITICAL"
    assert result.score == 0.95
    assert result.details["country"] == "IRN"


def test_velocity_anomaly_triggers_on_frequency(clean_transaction):
    """Customers with >= 3 recent transactions trigger RuleVelocityAnomaly."""
    rule = RuleVelocityAnomaly()
    mock_history = [
        {"transaction_id": "TX-H1", "amount": 100.0},
        {"transaction_id": "TX-H2", "amount": 250.0},
        {"transaction_id": "TX-H3", "amount": 180.0},
    ]
    result = rule.evaluate(tx=clean_transaction, history=mock_history)
    assert result.triggered
    assert result.severity == "MEDIUM"
    assert result.score == 0.65


def test_combined_rule_engine_multiple_triggers(rule_engine, structuring_transaction):
    """When a transaction trips structuring and high-risk country, scores compound."""
    tx = dict(structuring_transaction)
    tx["counterparty_country"] = "PRK"  # High risk jurisdiction + Structuring

    summary = rule_engine.evaluate(tx=tx)
    assert summary.is_flagged
    assert summary.triggered_rules_count >= 2
    assert summary.overall_severity == "CRITICAL"
    # Score should be at least max rule (0.95) + boost
    assert summary.overall_rule_score >= 0.95
