"""Unit tests for the AML Policy RAG pipeline: loader, chunker, vector store, and retriever."""

import pytest
from src.rag.document_loader import PolicyDocumentLoader, DocumentChunk
from src.rag.vector_store import SimpleVectorStore
from src.rag.retriever import AMLPolicyRetriever


@pytest.fixture
def doc_loader():
    return PolicyDocumentLoader()


@pytest.fixture
def vector_store(tmp_path):
    temp_index = tmp_path / "temp_policy_vectors.joblib"
    return SimpleVectorStore(persist_path=temp_index)


@pytest.fixture
def retriever(doc_loader, vector_store):
    r = AMLPolicyRetriever(loader=doc_loader, vector_store=vector_store)
    r.initialize_knowledge_base(force_reindex=True)
    return r


def test_load_all_documents(doc_loader):
    """Document loader must successfully find and parse all 3 curated AML policies."""
    docs = doc_loader.load_all_documents()
    assert len(docs) == 3
    doc_ids = [d.doc_id for d in docs]
    assert "FINCEN_SAR_ADVISORY" in doc_ids
    assert "FATF_RECOMMENDATION" in doc_ids
    assert "BANK_INTERNAL_AML_POLICY" in doc_ids


def test_missing_document_raises_error(doc_loader, tmp_path):
    """Loading a non-existent document raises FileNotFoundError."""
    missing_path = tmp_path / "missing_policy.md"
    with pytest.raises(FileNotFoundError, match="AML policy document not found"):
        doc_loader.load_document(missing_path)


def test_empty_document_raises_error(doc_loader, tmp_path):
    """Loading an empty document raises ValueError."""
    empty_file = tmp_path / "empty_policy.md"
    empty_file.write_text("   \n   ", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        doc_loader.load_document(empty_file)


def test_document_chunking_metadata(doc_loader):
    """Chunking must preserve document title, source, section title, and metadata."""
    chunks = doc_loader.load_and_chunk_all(chunk_size=400, chunk_overlap=50)
    assert len(chunks) > 10

    first_chunk = chunks[0]
    assert isinstance(first_chunk, DocumentChunk)
    assert first_chunk.chunk_id.startswith(first_chunk.doc_id)
    assert len(first_chunk.doc_title) > 5
    assert len(first_chunk.source) > 3
    assert len(first_chunk.section_title) > 2
    assert len(first_chunk.content) >= 30
    assert "doc_title" in first_chunk.metadata


def test_vector_store_indexing_and_count(vector_store, doc_loader):
    """Vector store indexes all chunks and accurately tracks count."""
    chunks = doc_loader.load_and_chunk_all()
    count = vector_store.add_documents(chunks)

    assert count == len(chunks)
    assert vector_store.count() == count
    assert vector_store.is_indexed()


def test_vector_store_persistence(tmp_path, doc_loader):
    """Vector store can be persisted to disk and reloaded with search capability."""
    temp_index = tmp_path / "persisted_vectors.joblib"
    store1 = SimpleVectorStore(persist_path=temp_index)
    chunks = doc_loader.load_and_chunk_all()
    store1.add_documents(chunks)

    # Load in new instance
    store2 = SimpleVectorStore(persist_path=temp_index)
    loaded = store2.load_index()
    assert loaded
    assert store2.is_indexed()
    assert store2.count() == len(chunks)

    # Verify search on reloaded store
    results = store2.search("structuring $10,000 threshold", top_k=2)
    assert len(results) > 0


def test_retriever_for_structuring_indicator(retriever):
    """Querying for structuring returns FinCEN structuring advisory with high relevance."""
    results = retriever.retrieve_for_indicators(
        risk_indicators=["Potential Currency Structuring (Smurfing)"],
        transaction_type="TRANSFER",
        top_k=3,
    )
    assert len(results) > 0
    top_result = results[0]
    assert "FinCEN" in top_result["source"] or "FinCEN" in top_result["doc_title"]
    assert top_result["similarity_score"] > 0.10


def test_retriever_for_sanctioned_jurisdiction(retriever):
    """Querying for high-risk jurisdiction wire returns FATF wire transfer guidance."""
    results = retriever.retrieve_for_indicators(
        risk_indicators=["High-Risk Sanctioned Jurisdiction"],
        transaction_type="WIRE",
        counterparty_country="IRN",
        top_k=3,
    )
    assert len(results) > 0
    top_result = results[0]
    assert "FATF" in top_result["source"] or "FATF" in top_result["doc_title"]


def test_retriever_for_transaction_dict(retriever, structuring_transaction):
    """Retriever accepts transaction dictionary and maps to relevant policies."""
    results = retriever.retrieve_for_transaction(
        tx=structuring_transaction,
        risk_indicators=["Potential Currency Structuring (Smurfing)"],
        top_k=2,
    )
    assert len(results) == 2
    assert "FinCEN" in results[0]["doc_title"] or "FinCEN" in results[0]["source"]


def test_retriever_empty_query_handling(retriever):
    """Empty search query returns empty list without error."""
    results = retriever.search_policies(query="")
    assert results == []


def test_retriever_citation_formatting(retriever):
    """Retriever formats human-readable citations with title, section, and score."""
    results = retriever.search_policies("currency transaction reporting 10000", top_k=2)
    citations = retriever.format_citations(results)
    assert "[1]" in citations
    assert "Section:" in citations
    assert "Match Confidence:" in citations


def test_retriever_benchmark_evaluation(retriever):
    """RAG benchmark evaluation must successfully hit all test query targets."""
    eval_metrics = retriever.evaluate_retrieval()
    assert eval_metrics["total_test_queries"] == 3
    assert eval_metrics["successful_hits"] == 3
    assert eval_metrics["hit_rate"] == 1.0
