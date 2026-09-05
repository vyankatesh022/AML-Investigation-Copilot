"""State schema for the AML transaction investigation workflow."""

from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class InvestigationWorkflowState(TypedDict, total=False):
    """Type-hinted state dictionary passed through the investigation workflow."""

    transaction_id: str
    transaction_details: Optional[Dict[str, Any]]
    risk_analysis: Optional[Dict[str, Any]]
    customer_profile: Optional[Dict[str, Any]]
    transaction_history: Optional[List[Dict[str, Any]]]
    history_summary: Optional[Dict[str, Any]]
    triggered_risk_indicators: List[str]
    retrieved_policies: List[Dict[str, Any]]
    formatted_citations: str
    investigation_summary: Optional[str]
    recommended_action: str
    workflow_status: str  # "IN_PROGRESS", "COMPLETED", "FAILED"
    error_message: Optional[str]


class InvestigationResult(BaseModel):
    """Validated output model representing the final result of an investigation."""

    transaction_id: str
    workflow_status: str = Field(..., description="Status: 'COMPLETED' or 'FAILED'")
    final_risk_level: str = Field(default="UNKNOWN", description="LOW, MEDIUM, or HIGH")
    final_risk_score: float = Field(default=0.0)
    is_flagged: bool = Field(default=False)
    triggered_rules: List[str] = Field(default_factory=list)
    customer_id: Optional[str] = None
    investigation_summary: Optional[str] = None
    recommended_action: str = Field(default="CLOSE_AS_FALSE_POSITIVE")
    error_message: Optional[str] = None
