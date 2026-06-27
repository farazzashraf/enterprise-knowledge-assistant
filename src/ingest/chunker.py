"""
Split text into ~600-token chunks with metadata preservation.

Chunk length is measured in *tokens* (via tiktoken's cl100k_base encoder),
not characters. Gemini embedding-2 does not publish a local tokenizer, so we
use tiktoken as a model-agnostic proxy: it counts a stable, consistent token
unit that tracks the embedder's real limits far better than character counts,
especially for dense content like code and tables.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    text: str
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"Chunk(doc={self.metadata.get('doc')!r}, "
            f"page={self.metadata.get('page')}, "
            f"idx={self.metadata.get('chunk_idx')}, "
            f"chars={len(self.text)})"
        )


class DocumentChunker:
    """
    Splits documents into chunks with metadata preservation.

    Uses RecursiveCharacterTextSplitter for intelligent splitting
    that preserves semantic boundaries (paragraphs, sentences, words).
    Length is measured in tokens via tiktoken (cl100k_base).
    """

    # Tokenizer used to measure chunk length. cl100k_base is a proxy for
    # Gemini's tokenizer (which has no public local implementation).
    TOKENIZER_ENCODING = "cl100k_base"

    # Target chunk size in TOKENS. ~600 tokens leaves ample headroom under
    # Gemini embedding-2's input limit while keeping chunks retrieval-sized.
    DEFAULT_CHUNK_SIZE = 600
    # 15% overlap (in tokens) to preserve context across chunk boundaries
    DEFAULT_CHUNK_OVERLAP = 90

    # Ordered list of separators — tries the largest boundary first
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        " ",     # Words
        "",      # Characters (fallback)
    ]

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # from_tiktoken_encoder sets length_function to count tokens with the
        # given encoder, so chunk_size/overlap are interpreted as token counts.
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.TOKENIZER_ENCODING,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or self.DEFAULT_SEPARATORS,
            is_separator_regex=False,
            add_start_index=False,
        )

    def chunk_document(
        self,
        text: str,
        doc_name: str,
        page_number: int,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Split a single page of text into chunks with metadata.

        Args:
            text: Raw text content to chunk.
            doc_name: Source document name (e.g., "HR_Policy.pdf").
            page_number: Page number in the source document.
            base_metadata: Additional metadata merged into every chunk.

        Returns:
            List of Chunk objects. Empty list if text is empty/whitespace.
        """
        if not text or not text.strip():
            return []

        docs = self.splitter.create_documents([text])
        total = len(docs)

        chunks: List[Chunk] = []
        for idx, doc in enumerate(docs):
            metadata: Dict[str, Any] = {
                "doc": doc_name,
                "page": page_number,
                "chunk_idx": idx,
                "chunk_total": total,
                **(base_metadata or {}),
            }
            chunks.append(Chunk(text=doc.page_content, metadata=metadata))

        logger.debug(
            f"Chunked '{doc_name}' page {page_number}: {total} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def chunk_pages(
        self,
        pages: List[Tuple[str, int, str, Optional[Dict[str, Any]]]],
    ) -> List[Chunk]:
        """
        Chunk multiple pages from one or more documents.

        Args:
            pages: List of tuples:
                (doc_name, page_number, text, optional_base_metadata)

        Returns:
            Flat list of all chunks with metadata.
        """
        all_chunks: List[Chunk] = []
        for doc_name, page_number, text, extra_meta in pages:
            chunks = self.chunk_document(
                text=text,
                doc_name=doc_name,
                page_number=page_number,
                base_metadata=extra_meta,
            )
            all_chunks.extend(chunks)
        return all_chunks


# --- Convenience functions ---

def chunk_text(
    text: str,
    doc_name: str = "unknown",
    page_number: int = 1,
    chunk_size: int = 600,
    chunk_overlap: int = 90,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    One-shot chunking returning (text, metadata) tuples.

    Matches the plan's interface requirement:
    "Return list of (text, metadata) tuples"
    """
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [(c.text, c.metadata) for c in chunker.chunk_document(text, doc_name, page_number)]


def chunk_loaded_document(
    loaded_doc,
    chunk_size: int = 600,
    chunk_overlap: int = 90,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Chunk a LoadedDocument from src.ingest.loaders.

    Args:
        loaded_doc: A LoadedDocument instance with .pages attribute.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap in tokens.

    Returns:
        List of (text, metadata) tuples ready for embedding/indexing.
    """
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[Chunk] = []

    from pathlib import Path as _Path

    for page in loaded_doc.pages:
        # Use just the filename, not the full path, for cleaner citations
        doc_name = _Path(page.source).name
        chunks = chunker.chunk_document(
            text=page.content,
            doc_name=doc_name,
            page_number=page.page_number,
            base_metadata=page.metadata,
        )
        all_chunks.extend(chunks)

    logger.info(
        f"Chunked '{_Path(loaded_doc.source).name}': "
        f"{len(loaded_doc.pages)} pages -> {len(all_chunks)} chunks"
    )
    return [(c.text, c.metadata) for c in all_chunks]