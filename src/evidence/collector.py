"""Evidence collection manager for compiling and organizing facts throughout an AML investigation."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from src.evidence.models import EvidenceItem, EvidenceType


class EvidenceCollector:
    """Collects, standardizes, and indexes factual evidence gathered during investigation workflow stages."""

    def __init__(self, items: Optional[List[EvidenceItem]] = None):
        self.items: List[EvidenceItem] = items or []

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_item(self, item: EvidenceItem) -> None:
        """Appends an evidence item to the collection."""
        self.items.append(item)

    def add_transaction_evidence(self, tx: Dict[str, Any]) -> EvidenceItem:
        """Captures core transaction record details as primary factual evidence."""
        tx_id = str(tx.get("transaction_id", "UNKNOWN"))
        cid = str(tx.get("customer_id", ""))
        amount = float(tx.get("amount", 0.0))
        channel = str(tx.get("channel", "N/A"))
        tx_type = str(tx.get("transaction_type", "N/A"))
        country = str(tx.get("counterparty_country", "USA"))

        item = EvidenceItem(
            evidence_id=f"EV-TX-{tx_id}",
            evidence_type=EvidenceType.TRANSACTION_DETAIL,
            source="Core Banking Ingestion Dataset",
            description=(
                f"Transaction {tx_id}: {tx_type} of ${amount:,.2f} via {channel} to destination {country}."
            ),
            related_transaction_id=tx_id,
            related_customer_id=cid if cid else None,
            supporting_data={
                "transaction_id": tx_id,
                "amount": amount,
                "timestamp": tx.get("timestamp", "N/A"),
                "transaction_type": tx_type,
                "channel": channel,
                "customer_id": cid,
                "counterparty_id": tx.get("counterparty_id", "N/A"),
                "counterparty_country": country,
                "oldbalanceOrg": float(tx.get("oldbalanceOrg", 0.0)),
                "newbalanceOrig": float(tx.get("newbalanceOrig", 0.0)),
                "oldbalanceDest": float(tx.get("oldbalanceDest", 0.0)),
                "newbalanceDest": float(tx.get("newbalanceDest", 0.0)),
            },
            retrieval_timestamp=self._now_iso(),
            relevance_explanation="Primary subject transaction initiating the compliance monitoring alert.",
            availability="AVAILABLE",
            confidence_or_score=1.0,
            confidence_type="Database Record Match",
        )
        self.add_item(item)
        return item

    def add_rule_evidence(
        self,
        tx_id: str,
        rule_exp: Dict[str, Any],
        index: int = 1,
        customer_id: Optional[str] = None,
    ) -> EvidenceItem:
        """Captures an individual deterministic compliance rule evaluation as evidence."""
        rule_id = str(rule_exp.get("rule_id", f"RULE_{index}"))
        rule_name = str(rule_exp.get("rule_name", "Unknown Compliance Rule"))
        severity = str(rule_exp.get("severity", "LOW"))
        score = float(rule_exp.get("risk_contribution_score", rule_exp.get("score", 0.0)))
        reason = str(rule_exp.get("reason", rule_exp.get("description", "")))
        condition = str(rule_exp.get("condition", "N/A"))

        item = EvidenceItem(
            evidence_id=f"EV-RULE-{rule_id}",
            evidence_type=EvidenceType.RULE_INDICATOR,
            source="Deterministic Compliance Rule Engine",
            description=f"Rule '{rule_name}' ({severity} severity): {reason}",
            related_transaction_id=tx_id,
            related_customer_id=customer_id,
            supporting_data={
                "rule_id": rule_id,
                "rule_name": rule_name,
                "condition": condition,
                "observed_values": rule_exp.get("observed_values", rule_exp.get("details", {})),
                "configured_thresholds": rule_exp.get("configured_thresholds", {}),
                "severity": severity,
                "score": score,
            },
            retrieval_timestamp=self._now_iso(),
            relevance_explanation=f"Deterministic heuristic evaluating statutory and policy compliance ({condition}).",
            availability="AVAILABLE",
            confidence_or_score=score,
            confidence_type="Rule Severity Score",
        )
        self.add_item(item)
        return item

    def add_ml_evidence(
        self,
        tx_id: str,
        ml_exp: Dict[str, Any],
        customer_id: Optional[str] = None,
    ) -> EvidenceItem:
        """Captures ML anomaly detection outputs and top feature signals as evidence."""
        proba = float(ml_exp.get("anomaly_probability", ml_exp.get("ml_risk_probability", 0.0)))
        is_anom = bool(ml_exp.get("is_anomalous", ml_exp.get("ml_is_anomalous", False)))
        model_name = str(ml_exp.get("model_name", "Random Forest Risk Classifier"))
        top_signals = ml_exp.get("top_signals", ml_exp.get("top_model_signals", []))
        interpretation = str(ml_exp.get("risk_interpretation", "Statistical pattern analysis on tabular features."))
        limitations = str(ml_exp.get("known_limitations", "Statistical anomaly detection only; does not prove financial crime."))

        item = EvidenceItem(
            evidence_id=f"EV-ML-{tx_id}",
            evidence_type=EvidenceType.ML_RISK_SIGNAL,
            source=model_name,
            description=(
                f"Statistical anomaly score: {proba * 100:.1f}%. "
                f"Status: {'Anomalous' if is_anom else 'Normal'}. "
                f"Interpretation: {interpretation}"
            ),
            related_transaction_id=tx_id,
            related_customer_id=customer_id,
            supporting_data={
                "model_name": model_name,
                "model_version": ml_exp.get("model_version", "1.0.0"),
                "anomaly_probability": proba,
                "is_anomalous": is_anom,
                "top_signals": top_signals,
                "input_features": ml_exp.get("input_features", {}),
                "known_limitations": limitations,
            },
            retrieval_timestamp=self._now_iso(),
            relevance_explanation="Quantifies statistical behavioral deviation from normal retail transaction distributions.",
            availability="AVAILABLE",
            confidence_or_score=proba,
            confidence_type="RandomForest Anomaly Probability",
        )
        self.add_item(item)
        return item

    def add_customer_evidence(
        self,
        tx_id: str,
        customer: Optional[Dict[str, Any]],
        customer_id: Optional[str],
    ) -> EvidenceItem:
        """Captures customer KYC demographics or notes lack of available KYC."""
        cid = customer_id or (customer.get("customer_id") if customer else "UNKNOWN")

        if customer:
            name = str(customer.get("full_name", "Unknown"))
            occ = str(customer.get("occupation", "N/A"))
            income = float(customer.get("monthly_income", 0.0))
            turnover = float(customer.get("expected_monthly_turnover", 0.0))
            tier = str(customer.get("kyc_risk_tier", "LOW"))
            pep = bool(customer.get("pep_status", False))

            item = EvidenceItem(
                evidence_id=f"EV-CUST-{cid}",
                evidence_type=EvidenceType.CUSTOMER_PROFILE,
                source="Customer KYC Profile Store",
                description=(
                    f"Customer {name} (ID: {cid}), occupation '{occ}', monthly income ${income:,.2f}, "
                    f"expected turnover ${turnover:,.2f}, KYC tier: {tier}, PEP: {'Yes' if pep else 'No'}."
                ),
                related_transaction_id=tx_id,
                related_customer_id=cid,
                supporting_data={
                    "customer_id": cid,
                    "full_name": name,
                    "occupation": occ,
                    "monthly_income": income,
                    "expected_monthly_turnover": turnover,
                    "kyc_risk_tier": tier,
                    "pep_status": pep,
                    "country_of_residence": customer.get("country_of_residence", "USA"),
                    "account_open_date": customer.get("account_open_date", "N/A"),
                },
                retrieval_timestamp=self._now_iso(),
                relevance_explanation="Establishes the customer KYC risk profile and baseline legitimate economic activity.",
                availability="AVAILABLE",
                confidence_or_score=1.0,
                confidence_type="Database Record Match",
            )
        else:
            item = EvidenceItem(
                evidence_id=f"EV-CUST-{cid}",
                evidence_type=EvidenceType.CUSTOMER_PROFILE,
                source="Customer KYC Profile Store",
                description=f"KYC profile for customer ID '{cid}' is unavailable in the internal database.",
                related_transaction_id=tx_id,
                related_customer_id=cid,
                supporting_data={"customer_id": cid},
                retrieval_timestamp=self._now_iso(),
                relevance_explanation="Absence of verified customer records warrants enhanced scrutiny (RFI).",
                availability="UNAVAILABLE",
                confidence_or_score=None,
                confidence_type=None,
            )

        self.add_item(item)
        return item

    def add_history_evidence(
        self,
        tx_id: str,
        customer_id: Optional[str],
        history: List[Dict[str, Any]],
        history_summary: Dict[str, Any],
    ) -> EvidenceItem:
        """Captures historical account observations and volume metrics as evidence."""
        cid = customer_id or "UNKNOWN"
        hist_count = len(history)
        tot_vol = float(history_summary.get("total_volume", 0.0))
        avg_amt = float(history_summary.get("average_amount", 0.0))
        counterparties = int(history_summary.get("unique_counterparties", 0))

        if hist_count > 0:
            desc = (
                f"Historical record demonstrates {hist_count} transactions (total volume: ${tot_vol:,.2f}, "
                f"average: ${avg_amt:,.2f}, unique counterparties: {counterparties})."
            )
            avail = "AVAILABLE"
        else:
            desc = "No prior transaction history found for this account in the monitoring window."
            avail = "UNAVAILABLE"

        item = EvidenceItem(
            evidence_id=f"EV-HIST-{cid}",
            evidence_type=EvidenceType.TRANSACTION_HISTORY,
            source="Historical Transaction Archive",
            description=desc,
            related_transaction_id=tx_id,
            related_customer_id=cid,
            supporting_data={
                "customer_id": cid,
                "transaction_count": hist_count,
                "total_volume": tot_vol,
                "average_amount": avg_amt,
                "unique_counterparties": counterparties,
                "recent_transaction_ids": [h.get("transaction_id") for h in history[:5]],
            },
            retrieval_timestamp=self._now_iso(),
            relevance_explanation="Provides the customer historical baseline to identify velocity and volume surges.",
            availability=avail,
            confidence_or_score=1.0 if hist_count > 0 else 0.0,
            confidence_type="Archive Query Match",
        )
        self.add_item(item)
        return item

    def add_policy_evidence(
        self,
        tx_id: str,
        chunk: Dict[str, Any],
        index: int,
        customer_id: Optional[str] = None,
    ) -> EvidenceItem:
        """Captures a retrieved AML policy clause with document lineage and similarity score."""
        chunk_id = str(chunk.get("chunk_id", f"POLICY-CHK-{index:02d}"))
        doc_title = str(chunk.get("doc_title", "AML Policy Document"))
        source = str(chunk.get("source", "Regulatory Standard"))
        section = str(chunk.get("section_title", "General Guidance"))
        score = float(chunk.get("similarity_score", 0.0))
        content = str(chunk.get("content", "")).strip()

        item = EvidenceItem(
            evidence_id=f"EV-RAG-{chunk_id}",
            evidence_type=EvidenceType.AML_POLICY_CLAUSE,
            source=f"{doc_title} ({source})",
            description=f"Section '{section}': \"{content[:180]}...\"",
            related_transaction_id=tx_id,
            related_customer_id=customer_id,
            supporting_data={
                "chunk_id": chunk_id,
                "doc_title": doc_title,
                "source": source,
                "section_title": section,
                "similarity_score": round(score, 4),
                "excerpt": content[:300],
            },
            retrieval_timestamp=self._now_iso(),
            relevance_explanation=f"Applicable regulatory mandate or bank guidance retrieved for detected risk indicators.",
            availability="AVAILABLE",
            confidence_or_score=round(score, 4),
            confidence_type="TF-IDF Cosine Similarity",
        )
        self.add_item(item)
        return item

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Serializes all collected evidence items to a list of dicts."""
        return [item.model_dump() for item in self.items]

    def format_traceability_markdown(self) -> str:
        """Generates a clean Markdown traceability matrix table."""
        if not self.items:
            return "_No formal evidence items were registered for this investigation._"

        lines = [
            "| Evidence ID | Evidence Type | Source | Availability | Score / Confidence | Description |",
            "| :--- | :--- | :--- | :---: | :---: | :--- |",
        ]
        for item in self.items:
            score_str = f"{item.confidence_or_score:.2f} ({item.confidence_type})" if item.confidence_or_score is not None else "N/A"
            clean_desc = item.description.replace("|", "/")
            lines.append(
                f"| `{item.evidence_id}` | {item.evidence_type.value} | {item.source} | {item.availability} | {score_str} | {clean_desc} |"
            )
        return "\n".join(lines)
