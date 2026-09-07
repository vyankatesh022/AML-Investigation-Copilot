"""Unit and integration tests for the LangGraph-inspired investigation workflow."""

import pytest
from src.workflow.graph import InvestigationWorkflow
from src.workflow.prompt import build_investigation_prompt
from src.workflow.llm import InvestigationLLMService


@pytest.fixture
def workflow():
    return InvestigationWorkflow()


def test_workflow_execution_high_risk_transaction(workflow):
    """Workflow successfully investigates structuring transaction and generates 8-part report."""
    result = workflow.investigate("TX-1003")

    assert result.workflow_status == "COMPLETED"
    assert result.final_risk_level == "HIGH"
    assert result.final_risk_score >= 0.70
    assert result.is_flagged is True
    assert any("Structuring" in r for r in result.triggered_rules)
    assert result.recommended_action == "ESCALATE_TO_SAR_COMMITTEE"
    assert result.investigation_summary is not None

    summary = result.investigation_summary
    # Verify all 8 required numbered sections are present
    assert "## 1. Transaction Details" in summary
    assert "## 2. Customer Information" in summary
    assert "## 3. Transaction Risk Assessment" in summary
    assert "## 4. Historical Transaction Observations" in summary
    assert "## 5. Potential Risk Indicators" in summary
    assert "## 6. Relevant AML Policy Information" in summary
    assert "## 7. AI-Generated Initial Assessment" in summary
    assert "## 8. Recommended Next Investigation Steps" in summary


def test_workflow_execution_clean_transaction(workflow):
    """Workflow evaluates a normal transaction as LOW risk with CLOSE disposition."""
    result = workflow.investigate("TX-1001")

    assert result.workflow_status == "COMPLETED"
    assert result.final_risk_level == "LOW"
    assert not result.is_flagged
    assert result.recommended_action == "CLOSE_AS_FALSE_POSITIVE"
    assert result.customer_id == "CUST-101"
    assert "## 1. Transaction Details" in result.investigation_summary


def test_workflow_missing_transaction_fails_safely(workflow):
    """Workflow halts safely with descriptive error when transaction does not exist."""
    result = workflow.investigate("TX-NON-EXISTENT-999")

    assert result.workflow_status == "FAILED"
    assert "not found" in result.error_message
    assert result.investigation_summary is None


def test_workflow_state_accumulation(workflow):
    """Each workflow node accurately populates its corresponding state dictionary."""
    state = workflow.run("TX-1003")

    assert state["workflow_status"] == "COMPLETED"
    assert state["transaction_details"] is not None
    assert state["transaction_details"]["amount"] == 9500.00

    assert state["risk_analysis"] is not None
    assert len(state["triggered_risk_indicators"]) > 0

    assert state["customer_profile"] is not None
    assert state["customer_profile"]["customer_id"] == "CUST-103"

    assert isinstance(state["transaction_history"], list)
    assert isinstance(state["retrieved_policies"], list)
    assert len(state["retrieved_policies"]) > 0
    assert len(state["formatted_citations"]) > 10
    assert state["investigation_summary"] is not None


def test_prompt_builder_includes_all_evidence(clean_transaction, sample_customer):
    """Prompt builder injects all metadata, KYC, risk scores, and citations."""
    mock_risk = {
        "final_risk_level": "LOW",
        "final_risk_score": 0.12,
        "rule_score": 0.0,
        "rule_severity": "LOW",
        "triggered_rules": [],
        "ml_probability": 0.05,
        "explanation": "Normal payment",
    }
    prompt = build_investigation_prompt(
        tx=clean_transaction,
        risk_analysis=mock_risk,
        customer=sample_customer,
        history=[],
        history_summary={},
        citations="[1] FinCEN SAR Advisory",
    )

    assert "TX-TEST-CLEAN" in prompt
    assert "Alice Johnson" in prompt
    assert "LOW" in prompt
    assert "[1] FinCEN SAR Advisory" in prompt


def test_offline_llm_synthesizer_grounding(clean_transaction, sample_customer):
    """Offline synthesizer formats structured Markdown matching exact 8 sections."""
    llm = InvestigationLLMService()
    summary = llm._synthesize_offline_summary(
        tx=clean_transaction,
        risk_analysis={"final_risk_level": "LOW", "final_risk_score": 0.1, "triggered_rules": []},
        customer=sample_customer,
        history=[],
        history_summary={},
        citations="Citations text",
    )

    assert "## 1. Transaction Details" in summary
    assert "## 8. Recommended Next Investigation Steps" in summary
    assert "Alice Johnson" in summary
