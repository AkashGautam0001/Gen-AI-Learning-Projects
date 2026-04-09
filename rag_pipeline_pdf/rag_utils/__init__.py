"""
RAG Pipeline Utilities Package

Organized modules for reusable RAG pipeline functions
"""

from .config import (
    PDF_PATH, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DIM,
    CHAT_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

from .logging_setup import setup_logging

from .clients import load_openai_client, load_qdrant_client

from .pdf_processor import read_pdf, get_chunk_text

from .embeddings import get_embeddings_batch, create_embeddings

from .qdrant_ops import (
    setup_collection, upload_vectors, check_collection_exist, search
)

from .llm import get_ai_response

from .text_utils import clean_llm_output, generate_queries, deduplicate_chunks

__all__ = [
    # Config
    "PDF_PATH", "COLLECTION_NAME", "EMBEDDING_MODEL", "EMBEDDING_DIM",
    "CHAT_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP",
    # Logging
    "setup_logging",
    # Clients
    "load_openai_client", "load_qdrant_client",
    # PDF
    "read_pdf", "get_chunk_text",
    # Embeddings
    "get_embeddings_batch", "create_embeddings",
    # Qdrant
    "setup_collection", "upload_vectors", "check_collection_exist", "search",
    # LLM
    "get_ai_response",
    # Text Utils
    "clean_llm_output", "generate_queries", "deduplicate_chunks",
]
