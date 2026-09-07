"""Comprehensive evaluation suite for AI investigation summaries, RAG retrieval, and end-to-end safety."""

import pytest
from src.workflow.graph import InvestigationWorkflow
from src.rag.retriever import AMLPolicyRetriever
from src.data.loader import TransactionDataLoader


@pytest.fixture
def workflow():
    return InvestigationWorkflow()


@pytest.fixture
def retriever():
    return AMLPolicyRetriever()


@pytest.fixture
def data_loader():
    return TransactionDataLoader()


# =====================================================================
# 1. AI Investigation Summary Evaluation (Criteria-Based Scoring)
# =====================================================================

def evaluate_summary_criteria(
    summary: str,
    tx_id: str,
    expected_customer_id: str,
    expected_amount: float,
    expected_rules: list,
    expected_policy_keyword: str,
) -> dict:
    """Evaluates an AI-generated investigation summary against 6 formal criteria:

    1. Relevance: Mentions target transaction ID and identified risk indicators.
    2. Evidence Usage: Uses transaction details, customer ID, and amounts.
    3. Factual Grounding: Contains accurate monetary values and entity identifiers.
    4. Policy Relevance: Cites applicable regulatory framework (FinCEN, FATF, Bank Policy).
    5. Clarity: Structured in standard 8 numbered sections with clear headings.
    6. Safety & Limitations: Contains non-autonomous disclaimer and preliminary scope.
    """
    assert summary is not None, "Summary must not be None"

    # 1. Relevance
    has_tx_id = tx_id in summary
    has_rules = any(rule.lower() in summary.lower() for rule in expected_rules) if expected_rules else True
    relevance = has_tx_id and has_rules

    # 2. Evidence Usage
    has_customer = expected_customer_id in summary
    has_amount_mention = f"{expected_amount:,.2f}" in summary or str(int(expected_amount)) in summary
    evidence_usage = has_customer and has_amount_mention

    # 3. Factual Grounding (No halluncinated amounts)
    factual_grounding = has_amount_mention and has_customer

    # 4. Policy Relevance
    policy_relevance = expected_policy_keyword.lower() in summary.lower()

    # 5. Clarity (Standard 8 numbered sections present)
    required_sections = [
        "## 1. Transaction Details",
        "## 2. Customer Information",
        "## 3. Transaction Risk Assessment",
        "## 4. Historical Transaction Observations",
        "## 5. Potential Risk Indicators",
        "## 6. Relevant AML Policy Information",
        "## 7. AI-Generated Initial Assessment",
        "## 8. Recommended Next Investigation Steps",
    ]
    sections_present = sum(1 for sec in required_sections if sec in summary)
    clarity = sections_present == 8

    # 6. Safety & Limitations (Contains advisory / human oversight language)
    safety_keywords = ["preliminary", "investigator", "compliance", "not a final legal decision", "advisory"]
    safety = any(kw in summary.lower() for kw in safety_keywords)

    score = (
        (1.0 if relevance else 0.0)
        + (1.0 if evidence_usage else 0.0)
        + (1.0 if factual_grounding else 0.0)
        + (1.0 if policy_relevance else 0.0)
        + (1.0 if clarity else 0.0)
        + (1.0 if safety else 0.0)
    ) / 6.0

    return {
        "overall_score": round(score, 2),
        "relevance": relevance,
        "evidence_usage": evidence_usage,
        "factual_grounding": factual_grounding,
        "policy_relevance": policy_relevance,
        "clarity": clarity,
        "safety": safety,
        "sections_present": sections_present,
    }


def test_ai_summary_evaluation_structuring(workflow):
    """Evaluate AI summary quality on a structuring transaction (TX-1003)."""
    result = workflow.investigate("TX-1003")
    assert result.workflow_status == "COMPLETED"

    metrics = evaluate_summary_criteria(
        summary=result.investigation_summary,
        tx_id="TX-1003",
        expected_customer_id="CUST-103",
        expected_amount=9500.00,
        expected_rules=["Structuring"],
        expected_policy_keyword="FinCEN",
    )

    assert metrics["overall_score"] == 1.0
    assert metrics["relevance"] is True
    assert metrics["evidence_usage"] is True
    assert metrics["factual_grounding"] is True
    assert metrics["policy_relevance"] is True
    assert metrics["clarity"] is True
    assert metrics["safety"] is True


def test_ai_summary_evaluation_sanctioned_jurisdiction(workflow):
    """Evaluate AI summary quality on a sanctioned country wire transfer (TX-1008)."""
    result = workflow.investigate("TX-1008")
    assert result.workflow_status == "COMPLETED"

    metrics = evaluate_summary_criteria(
        summary=result.investigation_summary,
        tx_id="TX-1008",
        expected_customer_id="CUST-107",
        expected_amount=15000.00,
        expected_rules=["Sanctioned Jurisdiction"],
        expected_policy_keyword="FATF",
    )

    assert metrics["overall_score"] == 1.0
    assert metrics["relevance"] is True
    assert metrics["evidence_usage"] is True
    assert metrics["factual_grounding"] is True
    assert metrics["policy_relevance"] is True
    assert metrics["clarity"] is True
    assert metrics["safety"] is True


def test_ai_summary_evaluation_clean_transaction(workflow):
    """Evaluate AI summary quality on a normal low-risk transaction (TX-1001)."""
    result = workflow.investigate("TX-1001")
    assert result.workflow_status == "COMPLETED"

    metrics = evaluate_summary_criteria(
        summary=result.investigation_summary,
        tx_id="TX-1001",
        expected_customer_id="CUST-101",
        expected_amount=45.50,
        expected_rules=[],
        expected_policy_keyword="AML",
    )

    assert metrics["overall_score"] >= 0.83  # At least 5/6 criteria
    assert metrics["relevance"] is True
    assert metrics["factual_grounding"] is True
    assert metrics["clarity"] is True
    assert metrics["safety"] is True


# =====================================================================
# 2. RAG Retrieval Quality Evaluation (4 Scenarios)
# =====================================================================

def test_rag_retrieval_scenario_structuring(retriever):
    """Scenario 1: Structuring corridor queries must retrieve FinCEN Guidance."""
    results = retriever.retrieve_for_indicators(
        risk_indicators=["Potential Currency Structuring (Smurfing)"],
        transaction_type="TRANSFER",
        top_k=3,
    )
    assert len(results) > 0
    top = results[0]
    assert "FinCEN" in top["source"] or "FinCEN" in top["doc_title"]
    assert "section_title" in top
    assert top["similarity_score"] > 0.0


def test_rag_retrieval_scenario_sanctioned_country(retriever):
    """Scenario 2: Sanctioned jurisdiction queries must retrieve FATF Wire Guidance."""
    results = retriever.retrieve_for_indicators(
        risk_indicators=["High-Risk Sanctioned Jurisdiction"],
        transaction_type="WIRE",
        counterparty_country="IRN",
        top_k=3,
    )
    assert len(results) > 0
    top = results[0]
    assert "FATF" in top["source"] or "FATF" in top["doc_title"]
    assert "section_title" in top


def test_rag_retrieval_scenario_high_value(retriever):
    """Scenario 3: High value currency queries must retrieve Bank AML / FinCEN CTR rules."""
    results = retriever.retrieve_for_indicators(
        risk_indicators=["High Value Currency Reporting Threshold"],
        transaction_type="WIRE",
        top_k=3,
    )
    assert len(results) > 0
    sources = [r["source"] for r in results]
    assert any("Bank" in s or "FinCEN" in s for s in sources)


def test_rag_retrieval_scenario_rapid_drain(retriever):
    """Scenario 4: Rapid balance drain queries must retrieve account turnover/velocity guidance."""
    results = retriever.retrieve_for_indicators(
        risk_indicators=["Rapid Account Balance Depletion"],
        transaction_type="CASH_OUT",
        top_k=3,
    )
    assert len(results) > 0
    assert all("doc_title" in r and "source" in r for r in results)


# =====================================================================
# 3. End-to-End Safety & Edge Cases
# =====================================================================

def test_investigation_with_unknown_customer(workflow):
    """Transaction with an un-enrolled customer completes safely without crashing."""
    state = workflow.nodes.get_customer_profile({
        "transaction_id": "TX-MOCK",
        "transaction_details": {"customer_id": "CUST-NONEXISTENT-999"},
    })
    assert state["customer_profile"] is None


def test_investigation_empty_history_handling(workflow):
    """Customer with zero prior transaction history is handled safely."""
    state = workflow.nodes.get_transaction_history({
        "transaction_id": "TX-MOCK",
        "transaction_details": {"customer_id": "CUST-NEW-CUSTOMER-001"},
    })
    assert state["transaction_history"] == []
    assert state["history_summary"].get("total_volume") == 0.0
    assert state["history_summary"].get("unique_counterparties") == 0
