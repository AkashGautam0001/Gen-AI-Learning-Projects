"""
Qdrant vector database operations
"""
import logging
from qdrant_client.models import VectorParams, Distance, PointStruct
from .embeddings import get_embeddings_batch


def setup_collection(qdrant_client, collection_name, vector_size):
    """Create Qdrant collection if it doesn't exist"""
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logging.info("Collection created")
    except Exception:
        logging.warning("Collection may already exist")


def upload_vectors(qdrant_client, collection_name, chunks, embeddings, source):
    """Upload chunks and embeddings to Qdrant"""
    logging.info("Uploading vectors to Qdrant...")
    points = [
        PointStruct(
            id=i,
            vector=vector,
            payload={"text": chunks[i], "source": source, "chunk_id": i}
        )
        for i, vector in enumerate(embeddings)
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    logging.info("Upload complete!")


def check_collection_exist(qdrant_client, collection):
    """Check if collection exists in Qdrant"""
    response = qdrant_client.collection_exists(collection_name=collection)
    print(response)
    return response


def search(query, openai_client, qdrant_client, collection_name):
    """Search for similar chunks using semantic similarity"""
    logging.info(f"Searching for query: {query}")
    query_vector = get_embeddings_batch(openai_client, [query])[0]
    hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5
    )
    logging.info(f"Search returned {len(hits.points)} results")
    return [{
        "text": hit.payload["text"],
        "score": hit.score,
        "query": query
    } for hit in hits.points]
