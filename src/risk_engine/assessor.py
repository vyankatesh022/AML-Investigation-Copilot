"""Combined AML risk assessment synthesizing deterministic rules and machine learning predictions."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.risk_engine.rules import RuleEngine, RuleEvaluationSummary
from src.risk_engine.ml_model import MLRiskClassifier


class CombinedRiskAssessment(BaseModel):
    """Unified risk evaluation combining deterministic compliance rules and ML prediction."""

    transaction_id: str
    final_risk_level: str = Field(..., description="Final risk rating: LOW, MEDIUM, or HIGH")
    final_risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized risk score from 0.0 to 1.0")
    is_flagged: bool = Field(..., description="Whether the transaction requires compliance investigation")
    rule_score: float
    rule_severity: str
    triggered_rules: List[str]
    ml_probability: float
    ml_risk_level: str
    top_ml_signals: List[str]
    explanation: str


class TransactionRiskAssessor:
    """Orchestrates dual-screening: evaluates heuristic compliance rules and ML model."""

    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        ml_classifier: Optional[MLRiskClassifier] = None,
    ):
        self.rule_engine = rule_engine or RuleEngine()
        self.ml_classifier = ml_classifier or MLRiskClassifier()

    def assess_transaction(
        self,
        tx: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> CombinedRiskAssessment:
        """Runs rule evaluation and ML inference, producing a consolidated risk assessment."""
        tx_id = str(tx.get("transaction_id", "UNKNOWN"))

        # 1. Rule Evaluation
        rule_summary: RuleEvaluationSummary = self.rule_engine.evaluate(
            tx=tx, customer=customer, history=history
        )
        rule_score = rule_summary.overall_rule_score
        rule_severity = rule_summary.overall_severity
        triggered_rule_names = [r.rule_name for r in rule_summary.triggered_rules]

        # 2. ML Prediction (gracefully handles if model not yet trained)
        try:
            self.ml_classifier.ensure_model_loaded()
            ml_result = self.ml_classifier.predict_risk(tx=tx, history=history)
            ml_proba = ml_result["ml_risk_probability"]
            ml_level = ml_result["ml_risk_level"]
            top_signals = ml_result["top_model_signals"]
        except Exception:
            # Fallback if model artifact is unavailable
            ml_proba = 0.0
            ml_level = "LOW"
            top_signals = ["ML model not loaded; defaulting to rule score"]

        # 3. Transparent Combined Scoring Logic
        if rule_severity == "CRITICAL":
            final_score = round(max(rule_score, ml_proba), 3)
            final_level = "HIGH"
        elif rule_summary.is_flagged:
            # Both rules and ML contribute, with a floor set by the rule score
            weighted_score = (0.55 * rule_score) + (0.45 * ml_proba)
            final_score = round(max(rule_score * 0.85, weighted_score), 3)
            if final_score >= 0.70 or rule_severity == "HIGH":
                final_level = "HIGH"
            elif final_score >= 0.40 or rule_severity == "MEDIUM":
                final_level = "MEDIUM"
            else:
                final_level = "LOW"
        else:
            # No rules triggered; driven by ML anomaly score
            final_score = round(ml_proba * 0.75, 3)
            if final_score >= 0.60:
                final_level = "HIGH"
            elif final_score >= 0.35:
                final_level = "MEDIUM"
            else:
                final_level = "LOW"

        is_flagged = bool(final_level in ["MEDIUM", "HIGH"] or rule_summary.is_flagged)

        # 4. Formulate understandable explanation
        if not is_flagged:
            explanation = (
                f"Transaction {tx_id} is evaluated as LOW risk. No deterministic compliance rules were "
                f"triggered, and ML anomaly probability is low ({ml_proba * 100:.1f}%)."
            )
        else:
            rules_text = (
                f"{len(triggered_rule_names)} rules triggered ({', '.join(triggered_rule_names)})"
                if triggered_rule_names
                else "No explicit rules triggered"
            )
            explanation = (
                f"Transaction {tx_id} is flagged as {final_level} risk (score: {final_score:.2f}). "
                f"Triage basis: {rules_text}; ML anomaly probability is {ml_proba * 100:.1f}%."
            )

        return CombinedRiskAssessment(
            transaction_id=tx_id,
            final_risk_level=final_level,
            final_risk_score=final_score,
            is_flagged=is_flagged,
            rule_score=rule_score,
            rule_severity=rule_severity,
            triggered_rules=triggered_rule_names,
            ml_probability=ml_proba,
            ml_risk_level=ml_level,
            top_ml_signals=top_signals,
            explanation=explanation,
        )
