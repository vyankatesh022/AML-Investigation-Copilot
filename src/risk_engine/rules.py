"""Deterministic heuristic compliance rules for AML transaction monitoring."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.config import settings


class RuleResult(BaseModel):
    """Result of evaluating an individual compliance rule."""

    rule_id: str
    rule_name: str
    triggered: bool
    severity: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    score: float = 0.0  # Normalized 0.0 to 1.0
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)


class RuleEvaluationSummary(BaseModel):
    """Consolidated summary of all rule evaluations for a transaction."""

    transaction_id: str
    is_flagged: bool
    overall_rule_score: float  # Normalized 0.0 to 1.0
    overall_severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    triggered_rules_count: int
    triggered_rules: List[RuleResult]
    summary_narrative: str


class BaseRule(ABC):
    """Abstract base class for compliance rules."""

    rule_id: str
    rule_name: str

    @abstractmethod
    def evaluate(
        self,
        tx: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleResult:
        """Evaluates rule logic against transaction context."""
        pass


class RuleHighAmount(BaseRule):
    """Flags transactions that meet or exceed statutory reporting thresholds."""

    rule_id = "RULE_HIGH_AMOUNT"
    rule_name = "High Value Currency Reporting Threshold"

    def evaluate(
        self,
        tx: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleResult:
        amount = float(tx.get("amount", 0.0))
        threshold = settings.HIGH_VALUE_THRESHOLD

        if amount >= threshold:
            severity = "CRITICAL" if amount >= 100000.0 else "HIGH"
            score = 0.95 if amount >= 100000.0 else 0.75
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                severity=severity,
                score=score,
                description=f"Transaction amount of ${amount:,.2f} meets or exceeds statutory threshold (${threshold:,.2f}).",
                details={"amount": amount, "threshold": threshold},
            )

        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=False,
            severity="LOW",
            score=0.0,
            description=f"Transaction amount ${amount:,.2f} is below reporting threshold (${threshold:,.2f}).",
            details={"amount": amount, "threshold": threshold},
        )


class RuleStructuring(BaseRule):
    """Detects transactions positioned just below reporting threshold ($9,000 - $9,999.99)."""

    rule_id = "RULE_STRUCTURING"
    rule_name = "Potential Currency Structuring (Smurfing)"

    def evaluate(
        self,
        tx: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleResult:
        amount = float(tx.get("amount", 0.0))
        lower = settings.STRUCTURING_LOWER_BOUND
        upper = settings.STRUCTURING_UPPER_BOUND

        if lower <= amount <= upper:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                severity="HIGH",
                score=0.85,
                description=(
                    f"Transaction amount of ${amount:,.2f} falls into structuring range "
                    f"(${lower:,.2f} - ${upper:,.2f}), a classic indicator of evasion of the $10,000 threshold."
                ),
                details={"amount": amount, "range_min": lower, "range_max": upper},
            )

        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=False,
            severity="LOW",
            score=0.0,
            description=f"Amount ${amount:,.2f} outside structuring corridor.",
            details={"amount": amount},
        )


class RuleRapidDrain(BaseRule):
    """Detects sudden, near-complete depletion of account balance."""

    rule_id = "RULE_RAPID_DRAIN"
    rule_name = "Rapid Account Balance Depletion"

    def evaluate(
        self,
        tx: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleResult:
        amount = float(tx.get("amount", 0.0))
        old_orig = float(tx.get("oldbalanceOrg", 0.0))
        new_orig = float(tx.get("newbalanceOrig", 0.0))

        if old_orig > 3000.0:
            drain_ratio = (old_orig - new_orig) / old_orig
            if drain_ratio >= settings.RAPID_DRAIN_RATIO and new_orig <= 1000.0:
                return RuleResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    triggered=True,
                    severity="HIGH",
                    score=0.80,
                    description=(
                        f"Transaction emptied {drain_ratio * 100:.1f}% of starting balance "
                        f"(${old_orig:,.2f} down to ${new_orig:,.2f}), characteristic of mule/pass-through activity."
                    ),
                    details={
                        "old_balance": old_orig,
                        "new_balance": new_orig,
                        "drain_ratio": round(drain_ratio, 4),
                    },
                )

        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=False,
            severity="LOW",
            score=0.0,
            description="Balance drain within normal limits.",
            details={"old_balance": old_orig, "new_balance": new_orig},
        )


class RuleHighRiskJurisdiction(BaseRule):
    """Identifies transactions involving high-risk, OFAC, or FATF grey/black-listed jurisdictions."""

    rule_id = "RULE_HIGH_RISK_JURISDICTION"
    rule_name = "High-Risk Sanctioned Jurisdiction"

    def evaluate(
        self,
        tx: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleResult:
        country = str(tx.get("counterparty_country", "USA")).upper()

        if country in settings.HIGH_RISK_JURISDICTIONS:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                severity="CRITICAL",
                score=0.95,
                description=(
                    f"Counterparty jurisdiction '{country}' is designated as a high-risk or sanctioned "
                    f"territory under FATF / international AML standards."
                ),
                details={"country": country, "sanctioned_list": list(settings.HIGH_RISK_JURISDICTIONS)},
            )

        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=False,
            severity="LOW",
            score=0.0,
            description=f"Jurisdiction '{country}' is standard risk.",
            details={"country": country},
        )


class RuleVelocityAnomaly(BaseRule):
    """Detects rapid succession of transactions exceeding customer normal baseline."""

    rule_id = "RULE_VELOCITY_ANOMALY"
    rule_name = "High Transaction Frequency Velocity"

    def evaluate(
        self,
        tx: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleResult:
        amount = float(tx.get("amount", 0.0))

        # Check if customer historical profile shows high deviation
        if history and len(history) >= settings.RAPID_VELOCITY_TX_LIMIT_24H:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                severity="MEDIUM",
                score=0.65,
                description=(
                    f"Customer executed {len(history)} transactions within monitoring window, "
                    f"exceeding rapid velocity threshold ({settings.RAPID_VELOCITY_TX_LIMIT_24H})."
                ),
                details={"tx_count": len(history), "limit": settings.RAPID_VELOCITY_TX_LIMIT_24H},
            )

        # Check customer monthly expected turnover deviation if customer profile is present
        if customer and customer.get("expected_monthly_turnover"):
            turnover = float(customer["expected_monthly_turnover"])
            if amount > turnover * 1.5 and amount > 5000:
                return RuleResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    triggered=True,
                    severity="HIGH",
                    score=0.75,
                    description=(
                        f"Transaction amount of ${amount:,.2f} exceeds 150% of customer's declared "
                        f"monthly turnover (${turnover:,.2f})."
                    ),
                    details={"amount": amount, "expected_turnover": turnover},
                )

        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=False,
            severity="LOW",
            score=0.0,
            description="Transaction frequency and volume conform with customer baseline.",
            details={"history_count": len(history) if history else 0},
        )


class RuleEngine:
    """Orchestrates and aggregates multiple compliance rules."""

    def __init__(self, rules: Optional[List[BaseRule]] = None):
        self.rules: List[BaseRule] = rules or [
            RuleHighAmount(),
            RuleStructuring(),
            RuleRapidDrain(),
            RuleHighRiskJurisdiction(),
            RuleVelocityAnomaly(),
        ]

    def evaluate(
        self,
        tx: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleEvaluationSummary:
        """Executes all rules and returns an aggregated compliance summary."""
        results: List[RuleResult] = []
        for rule in self.rules:
            result = rule.evaluate(tx=tx, customer=customer, history=history)
            results.append(result)

        triggered = [r for r in results if r.triggered]
        tx_id = str(tx.get("transaction_id", "UNKNOWN"))

        if not triggered:
            return RuleEvaluationSummary(
                transaction_id=tx_id,
                is_flagged=False,
                overall_rule_score=0.0,
                overall_severity="LOW",
                triggered_rules_count=0,
                triggered_rules=[],
                summary_narrative="No heuristic compliance red flags detected. Transaction appears normal.",
            )

        # Calculate overall score: take highest severity score with additive boost for multiple hits
        max_score = max(r.score for r in triggered)
        additive_boost = min(0.15, 0.05 * (len(triggered) - 1))
        combined_score = round(min(1.0, max_score + additive_boost), 3)

        # Severity resolution
        severities = [r.severity for r in triggered]
        if "CRITICAL" in severities:
            overall_severity = "CRITICAL"
        elif "HIGH" in severities or combined_score >= 0.75:
            overall_severity = "HIGH"
        elif "MEDIUM" in severities or combined_score >= 0.50:
            overall_severity = "MEDIUM"
        else:
            overall_severity = "LOW"

        names = ", ".join(r.rule_name for r in triggered)
        narrative = (
            f"Alert triggered with {len(triggered)} red flags ({names}). "
            f"Calculated rule risk score: {combined_score:.2f} ({overall_severity})."
        )

        return RuleEvaluationSummary(
            transaction_id=tx_id,
            is_flagged=True,
            overall_rule_score=combined_score,
            overall_severity=overall_severity,
            triggered_rules_count=len(triggered),
            triggered_rules=triggered,
            summary_narrative=narrative,
        )
