"""Document loader and text chunker for AML policies and regulatory guidelines."""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.config import settings


class DocumentChunk(BaseModel):
    """Schema representing an individual chunk of an AML policy document."""

    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. DOC-01-CHUNK-02")
    doc_id: str = Field(..., description="Parent document identifier")
    doc_title: str = Field(..., description="Title of the source document")
    source: str = Field(..., description="Issuing authority or organization")
    section_title: str = Field(..., description="Heading or topic of the section")
    content: str = Field(..., min_length=10, description="Cleaned text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for search and citations")


class PolicyDocument(BaseModel):
    """Schema representing a complete loaded AML policy document."""

    doc_id: str
    doc_title: str
    source: str
    file_path: str
    full_text: str


class PolicyDocumentLoader:
    """Loads, cleans, validates, and chunks AML policy documents."""

    def __init__(self, policies_dir: Optional[Path] = None):
        self.policies_dir = Path(policies_dir or settings.POLICIES_DATA_DIR)

    def load_document(self, file_path: Path) -> PolicyDocument:
        """Loads and parses a single markdown or text policy document."""
        if not file_path.exists():
            raise FileNotFoundError(f"AML policy document not found at: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            raise ValueError(f"Failed to read document at {file_path}: {exc}") from exc

        clean_text = content.strip()
        if not clean_text:
            raise ValueError(f"AML policy document at {file_path} is empty. Cannot extract text.")

        # Extract title from first markdown H1 heading or filename
        title_match = re.search(r"^#\s+(.+)$", clean_text, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else file_path.stem.replace("_", " ").title()

        # Extract source / issuing body if specified
        source_match = re.search(r"\*\*(?:Issuing Agency|Issuing Body|Owner):\*\*\s+(.+)$", clean_text, re.MULTILINE)
        source = source_match.group(1).strip() if source_match else "Regulatory Standard"

        doc_id = file_path.stem.upper()

        return PolicyDocument(
            doc_id=doc_id,
            doc_title=doc_title,
            source=source,
            file_path=str(file_path),
            full_text=clean_text,
        )

    def load_all_documents(self) -> List[PolicyDocument]:
        """Loads all supported markdown and text policy documents from the directory."""
        if not self.policies_dir.exists():
            raise FileNotFoundError(f"Policies directory not found at: {self.policies_dir}")

        doc_files = sorted(
            list(self.policies_dir.glob("*.md")) + list(self.policies_dir.glob("*.txt"))
        )
        if not doc_files:
            raise ValueError(f"No policy documents (.md or .txt) found in: {self.policies_dir}")

        documents = [self.load_document(f) for f in doc_files]
        return documents

    def chunk_document(self, doc: PolicyDocument, chunk_size: int = 500, chunk_overlap: int = 80) -> List[DocumentChunk]:
        """Splits a document into semantic section-based chunks with metadata."""
        chunks: List[DocumentChunk] = []

        # Split by markdown H2 headings (## Heading)
        sections = re.split(r"\n(?=##\s+)", doc.full_text)
        chunk_idx = 1

        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            # Extract section heading
            heading_match = re.match(r"^##\s+(.+)$", sec_clean, re.MULTILINE)
            section_title = heading_match.group(1).strip() if heading_match else "General Policy Guidance"

            # Remove header line from text content to keep clean prose
            body_text = re.sub(r"^##\s+.+$", "", sec_clean, flags=re.MULTILINE).strip()
            if not body_text:
                body_text = sec_clean

            # If section is small, keep as single chunk
            if len(body_text) <= chunk_size + chunk_overlap:
                chunk = DocumentChunk(
                    chunk_id=f"{doc.doc_id}-CHK-{chunk_idx:02d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.doc_title,
                    source=doc.source,
                    section_title=section_title,
                    content=body_text,
                    metadata={
                        "doc_title": doc.doc_title,
                        "source": doc.source,
                        "section": section_title,
                        "file_path": doc.file_path,
                    },
                )
                chunks.append(chunk)
                chunk_idx += 1
            else:
                # Split large section into overlapping windows
                start = 0
                while start < len(body_text):
                    end = min(start + chunk_size, len(body_text))
                    window = body_text[start:end].strip()

                    if len(window) >= 30:
                        chunk = DocumentChunk(
                            chunk_id=f"{doc.doc_id}-CHK-{chunk_idx:02d}",
                            doc_id=doc.doc_id,
                            doc_title=doc.doc_title,
                            source=doc.source,
                            section_title=section_title,
                            content=window,
                            metadata={
                                "doc_title": doc.doc_title,
                                "source": doc.source,
                                "section": section_title,
                                "file_path": doc.file_path,
                            },
                        )
                        chunks.append(chunk)
                        chunk_idx += 1

                    if end >= len(body_text):
                        break
                    start += chunk_size - chunk_overlap

        return chunks

    def load_and_chunk_all(self, chunk_size: int = 500, chunk_overlap: int = 80) -> List[DocumentChunk]:
        """Convenience method to load and chunk all policy documents in one call."""
        docs = self.load_all_documents()
        all_chunks: List[DocumentChunk] = []
        for doc in docs:
            chunks = self.chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)
        return all_chunks
