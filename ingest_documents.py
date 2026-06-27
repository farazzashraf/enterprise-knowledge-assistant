"""
CLI entry point for the document ingestion pipeline.

Usage:
    python ingest_documents.py --data-dir data/sample_documents
    python ingest_documents.py --data-dir data/sample_documents --recreate
"""

import argparse
import sys
import time
import logging
from pathlib import Path

# ── Setup logging before any other imports ──────────────────────
from src.logger import setup_logging, get_logger

logger = get_logger(__name__)


def run_ingestion(data_dir: str, recreate: bool = False) -> None:
    """
    Full ingestion pipeline:
      1. Load documents (PDF, DOCX, TXT)
      2. Chunk into ~600-token pieces with metadata
      3. Generate Gemini embeddings
      4. Index into Qdrant (dense) + BM25 (sparse)
    """
    from src.config import get_settings
    from src.ingest.loaders import SmartLoader
    from src.ingest.chunker import chunk_loaded_document
    from src.ingest.embedder import GeminiEmbedder
    from src.ingest.indexer import HybridIndexer

    settings = get_settings()
    start_time = time.time()

    # ── Step 1: Load documents ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1/4: Loading documents")
    logger.info("=" * 60)

    dir_path = Path(data_dir)
    if not dir_path.exists():
        logger.error(f"Data directory not found: {dir_path}")
        sys.exit(1)

    loader = SmartLoader()
    documents = loader.load_directory(dir_path)

    if not documents:
        logger.error("No documents found. Check the data directory.")
        sys.exit(1)

    total_pages = sum(len(doc.pages) for doc in documents)
    logger.info(f"Loaded {len(documents)} documents, {total_pages} total pages")

    # ── Step 2: Chunk documents ─────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2/4: Chunking documents")
    logger.info("=" * 60)

    # chunk_size is in tokens; chunk_overlap is a fraction of it.
    overlap_tokens = int(settings.chunk_size * settings.chunk_overlap)
    logger.info(f"Chunk size: {settings.chunk_size} tokens, overlap: {overlap_tokens} tokens")

    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_loaded_document(
            loaded_doc=doc,
            chunk_size=settings.chunk_size,
            chunk_overlap=overlap_tokens,
        )
        all_chunks.extend(doc_chunks)

    if not all_chunks:
        logger.error("No chunks produced. Documents may be empty.")
        sys.exit(1)

    logger.info(f"Total chunks: {len(all_chunks)}")

    # Preview first chunk
    if all_chunks:
        text, meta = all_chunks[0]
        logger.info(f"Sample chunk: doc='{meta.get('doc')}', page={meta.get('page')}, "
                     f"idx={meta.get('chunk_idx')}, chars={len(text)}")

    # ── Step 3: Generate embeddings ─────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3/4: Generating embeddings via Gemini embedding-2")
    logger.info("=" * 60)

    if not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY is not set in .env")
        sys.exit(1)

    embedder = GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
    )

    chunk_texts = [text for text, _ in all_chunks]
    # Use the source document name as the title in the embedding's document
    # structure ("title: {doc} | text: ..."), which aids asymmetric retrieval.
    chunk_titles = [meta.get("doc") for _, meta in all_chunks]
    embeddings = embedder.embed_documents(chunk_texts, titles=chunk_titles)

    logger.info(f"Generated {len(embeddings)} embeddings "
                f"(dim={len(embeddings[0]) if embeddings else 0})")

    # ── Step 4: Index into Qdrant + BM25 ────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4/4: Indexing into Qdrant + BM25")
    logger.info("=" * 60)

    # Convert (text, metadata) tuples into dicts for the indexer
    chunk_dicts = [
        {"text": text, "metadata": meta}
        for text, meta in all_chunks
    ]

    indexer = HybridIndexer(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key or None,
        collection_name=settings.qdrant_collection_name,
        vector_dim=settings.gemini_embedding_dim,
        bm25_index_path=settings.bm25_index_path,
        bm25_corpus_path=settings.bm25_corpus_path,
    )

    indexer.index(
        chunks=chunk_dicts,
        embeddings=embeddings,
        recreate=recreate,
    )

    # ── Summary ─────────────────────────────────────────────────
    elapsed = time.time() - start_time
    stats = indexer.get_stats()

    logger.info("")
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Documents loaded:   {len(documents)}")
    logger.info(f"  Total pages:        {total_pages}")
    logger.info(f"  Chunks created:     {len(all_chunks)}")
    logger.info(f"  Embeddings:         {len(embeddings)}")
    logger.info(f"  Qdrant points:      {stats['qdrant'].get('points_count', 'N/A')}")
    logger.info(f"  BM25 documents:     {stats['bm25'].get('documents', 'N/A')}")
    logger.info(f"  Time elapsed:       {elapsed:.1f}s")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the AnthroSync knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_documents.py --data-dir data/sample_documents
  python ingest_documents.py --data-dir data/sample_documents --recreate
  python ingest_documents.py --data-dir data/sample_documents --log-level DEBUG
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample_documents",
        help="Path to directory containing documents (default: data/sample_documents)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Initialize logging
    setup_logging(level=args.log_level)

    logger.info("AnthroSync Document Ingestion Pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Recreate collection: {args.recreate}")
    logger.info("")

    try:
        run_ingestion(
            data_dir=args.data_dir,
            recreate=args.recreate,
        )
    except KeyboardInterrupt:
        logger.warning("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
