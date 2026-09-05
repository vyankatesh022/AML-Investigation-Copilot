"""Prompt engineering templates for grounded AML transaction investigation summaries."""

from typing import Any, Dict, List


SYSTEM_PROMPT = """You are FinGuard AI, an expert AML (Anti-Money Laundering) Compliance Investigation Assistant.
Your mission is to perform an initial, objective investigation of potentially suspicious financial transactions.

IMPORTANT INVESTIGATION RULES:
1. Evidence-Based Grounding: Rely exclusively on the provided transaction details, customer profile, historical baseline, risk signals, and retrieved AML policies.
2. Fact vs Observation: Clearly distinguish between documented facts (e.g., exact transaction amounts, account balances) and analytical observations (e.g., behavioral deviations).
3. Regulatory Citation: Explicitly cite relevant retrieved AML policies, guidance notes, or statutory thresholds (e.g., FinCEN, FATF, Bank Policy).
4. No Hallucinations: If any piece of information is missing or unavailable (e.g. unknown customer or missing history), clearly state that it is unavailable. Never invent facts.
5. Preliminary Scope: This summary is an initial investigative copilot analysis to assist human compliance officers. Do NOT make definitive legal or criminal determinations.

OUTPUT STRUCTURE:
You must strictly format your investigation summary using these 8 numbered sections:
1. Transaction Details
2. Customer Information
3. Transaction Risk Assessment
4. Historical Transaction Observations
5. Potential Risk Indicators
6. Relevant AML Policy Information
7. AI-Generated Initial Assessment
8. Recommended Next Investigation Steps
"""


def build_investigation_prompt(
    tx: Dict[str, Any],
    risk_analysis: Dict[str, Any],
    customer: Dict[str, Any],
    history: List[Dict[str, Any]],
    history_summary: Dict[str, Any],
    citations: str,
) -> str:
    """Builds the comprehensive, grounded prompt provided to the LLM."""
    # Format transaction details
    tx_text = (
        f"- Transaction ID: {tx.get('transaction_id', 'N/A')}\n"
        f"- Timestamp: {tx.get('timestamp', 'N/A')}\n"
        f"- Amount: ${float(tx.get('amount', 0.0)):,.2f}\n"
        f"- Transaction Type: {tx.get('transaction_type', 'N/A')}\n"
        f"- Originator / Customer ID: {tx.get('customer_id', 'N/A')}\n"
        f"- Beneficiary / Counterparty ID: {tx.get('counterparty_id', 'N/A')}\n"
        f"- Originator Balance: ${float(tx.get('oldbalanceOrg', 0.0)):,.2f} -> ${float(tx.get('newbalanceOrig', 0.0)):,.2f}\n"
        f"- Beneficiary Balance: ${float(tx.get('oldbalanceDest', 0.0)):,.2f} -> ${float(tx.get('newbalanceDest', 0.0)):,.2f}\n"
        f"- Destination Country: {tx.get('counterparty_country', 'N/A')}\n"
        f"- Execution Channel: {tx.get('channel', 'N/A')}"
    )

    # Format customer details
    if customer:
        cust_text = (
            f"- Customer ID: {customer.get('customer_id', 'N/A')}\n"
            f"- Legal Name: {customer.get('full_name', 'N/A')}\n"
            f"- Occupation: {customer.get('occupation', 'N/A')}\n"
            f"- Declared Monthly Income: ${float(customer.get('monthly_income', 0.0)):,.2f}\n"
            f"- Expected Monthly Turnover: ${float(customer.get('expected_monthly_turnover', 0.0)):,.2f}\n"
            f"- KYC Risk Tier: {customer.get('kyc_risk_tier', 'N/A')}\n"
            f"- Politically Exposed Person (PEP): {'Yes' if customer.get('pep_status') else 'No'}\n"
            f"- Country of Residence: {customer.get('country_of_residence', 'N/A')}"
        )
    else:
        cust_text = "Customer KYC record is not available in the database."

    # Format history details
    if history_summary and history_summary.get("total_volume", 0) > 0:
        hist_text = (
            f"- Recorded Past Transactions: {len(history)}\n"
            f"- Historical Total Volume: ${float(history_summary.get('total_volume', 0.0)):,.2f}\n"
            f"- Historical Average Amount: ${float(history_summary.get('average_amount', 0.0)):,.2f}\n"
            f"- Unique Counterparties: {history_summary.get('unique_counterparties', 1)}"
        )
    else:
        hist_text = "No prior transaction history found for this account."

    # Format risk analysis
    rules_hit = risk_analysis.get("triggered_rules", [])
    rules_formatted = ", ".join(rules_hit) if rules_hit else "None"
    risk_text = (
        f"- Final Combined Risk Level: {risk_analysis.get('final_risk_level', 'LOW')}\n"
        f"- Final Risk Score: {risk_analysis.get('final_risk_score', 0.0):.2f}\n"
        f"- Rule Score: {risk_analysis.get('rule_score', 0.0):.2f} ({risk_analysis.get('rule_severity', 'LOW')})\n"
        f"- Triggered Compliance Rules: {rules_formatted}\n"
        f"- Machine Learning Anomaly Probability: {float(risk_analysis.get('ml_probability', 0.0)) * 100:.1f}%\n"
        f"- Triage Summary: {risk_analysis.get('explanation', 'N/A')}"
    )

    prompt = f"""EVIDENCE DOSSIER FOR TRANSACTION INVESTIGATION:

=== 1. TRANSACTION METADATA ===
{tx_text}

=== 2. CUSTOMER KYC & PROFILE ===
{cust_text}

=== 3. HISTORICAL BEHAVIORAL BASELINE ===
{hist_text}

=== 4. TRIAGE RISK ENGINE FINDINGS ===
{risk_text}

=== 5. RETRIEVED AML POLICIES & STATUTORY GUIDELINES ===
{citations}

INSTRUCTIONS:
Generate a comprehensive, structured AML investigation summary following the 8 required numbered sections.
Ensure that all statements reference the evidence above.
"""
    return prompt
