"""
Embedding utilities for text vectorization
"""
import logging
import time
from .config import EMBEDDING_MODEL


def get_embeddings_batch(openai_client, text_list):
    """Get embeddings for a batch of texts"""
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text_list
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return []


def create_embeddings(openai_client, chunks):
    """Create embeddings for all chunks"""
    logging.info("Creating embeddings...")
    start_time = time.time()
    embeddings = get_embeddings_batch(openai_client, chunks)
    logging.info(f"Embeddings created in {time.time() - start_time:.2f} seconds")
    return embeddings
