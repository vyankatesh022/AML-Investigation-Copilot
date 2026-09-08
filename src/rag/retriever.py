"""High-level AML policy retriever connecting transaction risk indicators to relevant regulatory clauses."""

from typing import Any, Dict, List, Optional
from src.rag.document_loader import PolicyDocumentLoader
from src.rag.vector_store import SimpleVectorStore

# Mapping of known risk indicators to semantic search query keywords
RISK_INDICATOR_QUERY_MAP: Dict[str, str] = {
    "Potential Currency Structuring (Smurfing)": "structuring smurfing 10000 dollar threshold evasion CTR currency transaction report",
    "High Value Currency Reporting Threshold": "statutory reporting threshold CTR currency transaction report 10000 high value wire",
    "Rapid Account Balance Depletion": "rapid movement funds balance depletion pass-through account mule cybercrime 90 percent",
    "High-Risk Sanctioned Jurisdiction": "FATF recommendation 16 wire transfer high risk sanctioned jurisdiction North Korea Iran enhanced due diligence",
    "High Transaction Frequency Velocity": "rapid velocity frequency bursts customer turnover deviation request for information RFI baseline",
}


class AMLPolicyRetriever:
    """Orchestrates AML policy retrieval based on transaction risk flags, types, and free-text queries."""

    def __init__(
        self,
        loader: Optional[PolicyDocumentLoader] = None,
        vector_store: Optional[SimpleVectorStore] = None,
    ):
        self.loader = loader or PolicyDocumentLoader()
        self.vector_store = vector_store or SimpleVectorStore()

    def initialize_knowledge_base(self, force_reindex: bool = False) -> int:
        """Loads policy documents from disk and indexes them into the vector store."""
        if not force_reindex and self.vector_store.load_index():
            return self.vector_store.count()

        chunks = self.loader.load_and_chunk_all()
        count = self.vector_store.add_documents(chunks)
        return count

    def build_query_from_indicators(
        self,
        risk_indicators: List[str],
        transaction_type: str = "",
        counterparty_country: str = "",
    ) -> str:
        """Translates risk signals into an enriched search query for the vector store."""
        query_terms: List[str] = []

        if transaction_type:
            query_terms.append(f"transaction type {transaction_type.lower()}")

        if counterparty_country and counterparty_country.upper() not in ["USA", "CAN", "GBR"]:
            query_terms.append(f"cross-border jurisdiction {counterparty_country} high risk sanctions")

        for indicator in risk_indicators:
            mapped_keywords = RISK_INDICATOR_QUERY_MAP.get(indicator)
            if mapped_keywords:
                query_terms.append(mapped_keywords)
            else:
                query_terms.append(indicator)

        final_query = " ".join(query_terms).strip()
        return final_query if final_query else "AML transaction monitoring compliance policy guidelines"

    def retrieve_for_indicators(
        self,
        risk_indicators: List[str],
        transaction_type: str = "",
        counterparty_country: str = "",
        top_k: int = 3,
        min_similarity: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Retrieves matching AML policy chunks based on detected risk indicators."""
        self.initialize_knowledge_base()
        query = self.build_query_from_indicators(
            risk_indicators=risk_indicators,
            transaction_type=transaction_type,
            counterparty_country=counterparty_country,
        )
        return self.vector_store.search(query=query, top_k=top_k, min_similarity=min_similarity)

    def retrieve_for_transaction(
        self,
        tx: Dict[str, Any],
        risk_indicators: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieves applicable regulatory policies for a specific transaction dictionary."""
        indicators = risk_indicators or []
        tx_type = str(tx.get("transaction_type", ""))
        country = str(tx.get("counterparty_country", "USA"))

        return self.retrieve_for_indicators(
            risk_indicators=indicators,
            transaction_type=tx_type,
            counterparty_country=country,
            top_k=top_k,
        )

    def search_policies(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Direct free-text policy search function."""
        self.initialize_knowledge_base()
        return self.vector_store.search(query=query, top_k=top_k)

    @staticmethod
    def format_citations(retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into clean, human-readable compliance citations with source traceability."""
        if not retrieved_chunks:
            return "No specific regulatory policy clauses directly matched the query."

        citations: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "AML Standard")
            doc_title = chunk.get("doc_title", "AML Policy")
            section = chunk.get("section_title", "General")
            score = chunk.get("similarity_score", 0.0)
            chunk_id = chunk.get("chunk_id", f"CHK-{idx:02d}")
            content = chunk.get("content", "").strip()

            citation = (
                f"[{idx}] {doc_title} ({source}) [Chunk ID: {chunk_id}]\n"
                f"    Section: {section} | Match Confidence: {score:.2f} (TF-IDF Cosine Similarity)\n"
                f"    Direct Policy Excerpt: \"{content[:250]}...\""
            )
            citations.append(citation)

        return "\n\n".join(citations)

    @staticmethod
    def get_traceable_sources(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Returns structured metadata records for each retrieved policy chunk to support evidence traceability."""
        sources = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            sources.append({
                "citation_index": idx,
                "chunk_id": chunk.get("chunk_id", ""),
                "doc_id": chunk.get("doc_id", ""),
                "doc_title": chunk.get("doc_title", ""),
                "source": chunk.get("source", ""),
                "section_title": chunk.get("section_title", ""),
                "similarity_score": round(float(chunk.get("similarity_score", 0.0)), 4),
                "score_type": "TF-IDF Cosine Similarity",
                "excerpt": chunk.get("content", "").strip()[:300],
            })
        return sources

    def evaluate_retrieval(self, test_suite: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Evaluates policy retrieval against benchmark test queries with known expected documents."""
        self.initialize_knowledge_base()

        benchmarks = test_suite or [
            {
                "query": "structuring cash deposits under $10,000 threshold",
                "expected_doc": "FINCEN_SAR_ADVISORY",
            },
            {
                "query": "cross border wire transfer high risk country sanctions North Korea Iran",
                "expected_doc": "FATF_RECOMMENDATION",
            },
            {
                "query": "customer turnover deviation request for information RFI escalation",
                "expected_doc": "BANK_INTERNAL_AML_POLICY",
            },
        ]

        total_queries = len(benchmarks)
        hits = 0
        details = []

        for b in benchmarks:
            results = self.vector_store.search(query=b["query"], top_k=3)
            retrieved_doc_ids = [r["chunk_id"].split("-CHK-")[0] for r in results]
            is_hit = b["expected_doc"] in retrieved_doc_ids
            if is_hit:
                hits += 1

            details.append({
                "query": b["query"],
                "expected_doc": b["expected_doc"],
                "retrieved_docs": retrieved_doc_ids,
                "hit": is_hit,
                "top_score": results[0]["similarity_score"] if results else 0.0,
            })

        hit_rate = round(hits / total_queries, 4) if total_queries > 0 else 0.0
        return {
            "total_test_queries": total_queries,
            "successful_hits": hits,
            "hit_rate": hit_rate,
            "evaluation_details": details,
        }
