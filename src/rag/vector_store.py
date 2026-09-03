"""Vector database and embedding search engine for AML policy chunks."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings
from src.rag.document_loader import DocumentChunk


class SimpleVectorStore:
    """Stores document chunks, generates TF-IDF semantic embeddings, and performs cosine similarity search."""

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_dir = settings.DATA_DIR / "chroma_db"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_path = Path(persist_path or (self.persist_dir / "policy_vectors.joblib"))

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.chunks: List[DocumentChunk] = []
        self.tfidf_matrix: Optional[np.ndarray] = None

    def is_indexed(self) -> bool:
        """Returns True if documents have been vectorized and indexed in memory."""
        return bool(self.chunks and self.vectorizer is not None and self.tfidf_matrix is not None)

    def count(self) -> int:
        """Returns the number of indexed document chunks."""
        return len(self.chunks)

    def add_documents(self, chunks: List[DocumentChunk]) -> int:
        """Embeds and indexes document chunks into the vector store."""
        if not chunks:
            raise ValueError("Cannot index an empty list of document chunks.")

        self.chunks = chunks
        corpus = [f"{c.doc_title} {c.section_title} {c.content}" for c in chunks]

        # TF-IDF vectorizer configured with unigrams and bigrams
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            max_features=2500,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self.save_index()
        return len(self.chunks)

    def save_index(self) -> None:
        """Persists the vector index and chunks to disk."""
        if not self.is_indexed():
            raise ValueError("Cannot save empty vector store.")

        payload = {
            "chunks": [c.model_dump() for c in self.chunks],
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
        }
        joblib.dump(payload, self.persist_path)

    def load_index(self) -> bool:
        """Loads a previously persisted vector index from disk."""
        if not self.persist_path.exists():
            return False

        payload = joblib.load(self.persist_path)
        self.chunks = [DocumentChunk(**c) for c in payload["chunks"]]
        self.vectorizer = payload["vectorizer"]
        self.tfidf_matrix = payload["tfidf_matrix"]
        return True

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Performs vector similarity search given a text query and returns ranked chunks."""
        if not query or not query.strip():
            return []

        if not self.is_indexed():
            # Attempt auto-loading persisted index
            loaded = self.load_index()
            if not loaded or not self.is_indexed():
                raise ValueError("Vector store is empty. Documents must be indexed before searching.")

        # Vectorize incoming query
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Rank indices by descending similarity
        ranked_indices = np.argsort(similarities)[::-1]

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            sim_score = float(similarities[idx])
            if sim_score < min_similarity:
                continue

            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_title": chunk.doc_title,
                "source": chunk.source,
                "section_title": chunk.section_title,
                "content": chunk.content,
                "similarity_score": round(sim_score, 4),
                "metadata": chunk.metadata,
            })

            if len(results) >= top_k:
                break

        return results
