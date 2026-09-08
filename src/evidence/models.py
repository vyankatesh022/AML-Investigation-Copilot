"""Structured data models for AML investigation evidence traceability."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """Enumeration of recognized investigation evidence types."""

    TRANSACTION_DETAIL = "TRANSACTION_DETAIL"
    RULE_INDICATOR = "RULE_INDICATOR"
    ML_RISK_SIGNAL = "ML_RISK_SIGNAL"
    CUSTOMER_PROFILE = "CUSTOMER_PROFILE"
    TRANSACTION_HISTORY = "TRANSACTION_HISTORY"
    AML_POLICY_CLAUSE = "AML_POLICY_CLAUSE"


class EvidenceItem(BaseModel):
    """Structured unit of investigation evidence maintaining full lineage and factual grounding."""

    evidence_id: str = Field(..., description="Unique deterministic identifier (e.g. 'EV-TX-001')")
    evidence_type: EvidenceType = Field(..., description="Category of evidence")
    source: str = Field(..., description="Originating system, model, table, or policy document")
    description: str = Field(..., description="Plain-language human-readable description of what this evidence shows")
    related_transaction_id: str = Field(..., description="Target transaction ID under investigation")
    related_customer_id: Optional[str] = Field(default=None, description="Related customer ID if applicable")
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Exact factual key-value pairs supporting the evidence")
    retrieval_timestamp: str = Field(..., description="ISO 8601 timestamp when this evidence was captured")
    relevance_explanation: str = Field(default="", description="Why this evidence is pertinent to the AML risk assessment")
    availability: str = Field(default="AVAILABLE", description="'AVAILABLE' if retrieved or 'UNAVAILABLE' if missing/unrecorded")
    confidence_or_score: Optional[float] = Field(default=None, description="Measured score, probability, or relevance metric (if genuinely supported)")
    confidence_type: Optional[str] = Field(default=None, description="Type of score (e.g. 'Cosine Similarity', 'Anomaly Probability', 'Rule Severity Score')")
