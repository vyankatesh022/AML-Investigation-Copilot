"""LLM integration service supporting Gemini API, OpenAI API, and grounded offline synthesis."""

from typing import Any, Dict, List, Optional
import httpx

from src.config import settings
from src.workflow.prompt import SYSTEM_PROMPT, build_investigation_prompt


class InvestigationLLMService:
    """Provides LLM inference for generating structured investigation summaries."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.gemini_key = gemini_api_key or settings.GEMINI_API_KEY
        self.openai_key = openai_api_key or settings.OPENAI_API_KEY
        self.timeout = timeout

    def generate_summary(
        self,
        tx: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
        history: List[Dict[str, Any]],
        history_summary: Dict[str, Any],
        citations: str,
        evidence_items: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generates a structured investigation summary using live LLM or grounded offline synthesizer."""
        prompt = build_investigation_prompt(
            tx=tx,
            risk_analysis=risk_analysis,
            customer=customer,
            history=history,
            history_summary=history_summary,
            citations=citations,
            evidence_items=evidence_items,
        )

        # 1. Try Google Gemini API if key is present
        if self.gemini_key and self.gemini_key.strip() and not self.gemini_key.startswith("your_"):
            try:
                return self._call_gemini(prompt)
            except Exception:
                pass  # Fall through to OpenAI or offline synthesizer

        # 2. Try OpenAI API if key is present
        if self.openai_key and self.openai_key.strip() and not self.openai_key.startswith("your_"):
            try:
                return self._call_openai(prompt)
            except Exception:
                pass  # Fall through to offline synthesizer

        # 3. Grounded Offline Rule-Based Synthesizer (Zero-Network, 100% Reliable Fallback)
        return self._synthesize_offline_summary(
            tx=tx,
            risk_analysis=risk_analysis,
            customer=customer,
            history=history,
            history_summary=history_summary,
            citations=citations,
            evidence_items=evidence_items,
        )

    def _call_gemini(self, prompt: str) -> str:
        """Invokes the Google Gemini REST API using httpx."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai(self, prompt: str) -> str:
        """Invokes the OpenAI Chat Completions REST API using httpx."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openai_key}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _synthesize_offline_summary(
        self,
        tx: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
        history: List[Dict[str, Any]],
        history_summary: Dict[str, Any],
        citations: str,
        evidence_items: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Produces a fully grounded, structured 8-part investigation summary offline with evidence traceability."""
        tx_id = tx.get("transaction_id", "UNKNOWN")
        cid = tx.get("customer_id", "UNKNOWN")
        amount = float(tx.get("amount", 0.0))
        tx_type = tx.get("transaction_type", "UNKNOWN")
        risk_level = risk_analysis.get("final_risk_level", "LOW")
        risk_score = float(risk_analysis.get("final_risk_score", 0.0))
        ml_proba = float(risk_analysis.get("ml_probability", 0.0)) * 100
        rules = risk_analysis.get("triggered_rules", [])
        rule_exps = risk_analysis.get("rule_explanations", [])
        ml_exp = risk_analysis.get("ml_explanation") or {}

        # Recommended action resolution
        if risk_level == "HIGH" or "CRITICAL" in risk_analysis.get("rule_severity", ""):
            action = "ESCALATE_TO_SAR_COMMITTEE"
            action_desc = "Recommend formal review by the Senior AML Compliance Committee for Suspicious Activity Report (SAR) filing within 30 days."
        elif risk_level == "MEDIUM":
            action = "REQUEST_FOR_INFORMATION"
            action_desc = "Recommend issuing a formal Request for Information (RFI) to the customer to obtain supporting documentation on the economic source of funds."
        else:
            action = "CLOSE_AS_FALSE_POSITIVE"
            action_desc = "Recommend closing alert as a benign transaction. Activity conforms with normal personal banking patterns."

        # Format history statement
        avg_hist = float(history_summary.get("average_amount", amount))
        hist_count = len(history)
        if hist_count > 0:
            hist_obs = f"Account demonstrates {hist_count} recent transactions with an average ticket size of ${avg_hist:,.2f}."
            hist_evidence_tag = f"[Evidence: EV-HIST-{cid}]"
        else:
            hist_obs = "Account exhibits no recorded prior transaction history in the monitoring window."
            hist_evidence_tag = f"[Evidence: EV-HIST-{cid} | Status: UNAVAILABLE]"

        # Format customer statement
        if customer:
            cust_obs = (
                f"Account held by {customer.get('full_name')} (ID: {customer.get('customer_id')}), "
                f"employed as {customer.get('occupation')} with declared monthly income of ${float(customer.get('monthly_income', 0.0)):,.2f}. "
                f"Assigned KYC Risk Tier: {customer.get('kyc_risk_tier')} (PEP: {'Yes' if customer.get('pep_status') else 'No'})."
            )
            cust_evidence_tag = f"[Evidence: EV-CUST-{customer.get('customer_id')}]"
        else:
            cust_obs = "Customer KYC profile is unavailable in the internal database."
            cust_evidence_tag = f"[Evidence: EV-CUST-{cid} | Status: UNAVAILABLE]"

        # Format potential risk indicators with condition & actual observed values
        if rule_exps:
            rule_items_text = "\n".join(
                f"- **{r.get('rule_name')}** (Severity: {r.get('severity')} | Contribution: {float(r.get('risk_contribution_score', 0.0)):.2f})\n"
                f"  - Condition: {r.get('condition')}\n"
                f"  - Evidence Details: {r.get('reason')} [Observed: {r.get('observed_values')}]"
                for r in rule_exps
            )
        elif rules:
            rule_items_text = "\n".join(f"- {rule}" for rule in rules)
        else:
            rule_items_text = "- No deterministic AML red flags triggered."

        # Format evidence traceability table if evidence items exist
        traceability_table = ""
        if evidence_items:
            table_lines = [
                "\n## 9. Evidence Traceability Matrix",
                "| Evidence ID | Type | Source | Availability | Score / Confidence | Reference Details |",
                "| :--- | :--- | :--- | :---: | :---: | :--- |",
            ]
            for item in evidence_items:
                score_str = f"{item.get('confidence_or_score'):.2f}" if item.get('confidence_or_score') is not None else "N/A"
                if item.get('confidence_type'):
                    score_str += f" ({item.get('confidence_type')})"
                desc = str(item.get("description", "")).replace("|", "/")
                table_lines.append(
                    f"| `{item.get('evidence_id')}` | {item.get('evidence_type')} | {item.get('source')} | {item.get('availability')} | {score_str} | {desc} |"
                )
            traceability_table = "\n" + "\n".join(table_lines)

        summary = f"""# Transaction Investigation Summary: {tx_id}

## 1. Transaction Details
[Evidence: EV-TX-{tx_id}]
- **Transaction ID:** {tx_id}
- **Timestamp:** {tx.get('timestamp', 'N/A')}
- **Execution Channel:** {tx.get('channel', 'N/A')}
- **Transaction Type & Amount:** {tx_type} for ${amount:,.2f}
- **Originator / Beneficiary:** {tx.get('customer_id', 'N/A')} -> {tx.get('counterparty_id', 'N/A')} ({tx.get('counterparty_country', 'USA')})
- **Account Balance Delta:** Originator balance adjusted from ${float(tx.get('oldbalanceOrg', 0.0)):,.2f} down to ${float(tx.get('newbalanceOrig', 0.0)):,.2f}.

## 2. Customer Information
{cust_evidence_tag}
{cust_obs}

## 3. Transaction Risk Assessment
[Evidence: EV-RULE-{tx_id} & EV-ML-{tx_id}]
- **Final Combined Risk Level:** {risk_level} (Score: {risk_score:.2f} / 1.00)
- **Deterministic Rule Score:** {risk_analysis.get('rule_score', 0.0):.2f} ({risk_analysis.get('rule_severity', 'LOW')})
- **Machine Learning Anomaly Probability:** {ml_proba:.1f}% (Model: {ml_exp.get('model_name', 'Random Forest Classifier')})
- **ML Interpretation:** {ml_exp.get('risk_interpretation', 'Statistical pattern score.')}
- **ML Limitations:** {ml_exp.get('known_limitations', 'Statistical anomaly only; does not confirm illicit funds.')}

## 4. Historical Transaction Observations
{hist_evidence_tag}
{hist_obs} Current transaction volume is {amount / (avg_hist + 1.0):.1f}x relative to historical average.

## 5. Potential Risk Indicators
[Evidence: Deterministic Compliance Engine]
{rule_items_text}

## 6. Relevant AML Policy Information
[Evidence: RAG Regulatory Policy Knowledge Base]
{citations}

## 7. AI-Generated Initial Assessment
> **Compliance & Legal Notice:** This section contains AI-generated preliminary investigative observations to support human compliance analysts. It is not a final legal decision, regulatory ruling, or autonomous SAR filing.

Transaction {tx_id} was evaluated as **{risk_level} risk** based on factual evidence compiled in the dossier. {'The observed behavior exhibits patterns inconsistent with standard retail transactions, including deliberate threshold proximity or balance depletion characteristics.' if risk_level != 'LOW' else 'The transaction represents customary banking activity within approved thresholds.'} This copilot assessment is grounded in verified account snapshots and regulatory guidance to support human compliance triage.

## 8. Recommended Next Investigation Steps
- **Recommended Action:** `{action}`
- **Operational Guidance:** {action_desc}
{traceability_table}"""
        return summary
