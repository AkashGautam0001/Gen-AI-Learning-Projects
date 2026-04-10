"""
Cross-Encoder RAG Pipeline - RRF with semantic similarity scoring
"""
import logging
from operator import itemgetter
from rag_utils import (
    setup_logging, load_openai_client, load_qdrant_client, read_pdf,
    get_chunk_text, create_embeddings, setup_collection, upload_vectors,
    search, get_ai_response, check_collection_exist, generate_queries,
    PDF_PATH, COLLECTION_NAME, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP
)

from sentence_transformers import CrossEncoder


def rrf_fusion_with_semantic_score(result_per_query, k=60):
    """
    RRF with semantic similarity: Combine RRF scores with semantic similarity
    This hybrid approach balances ranking diversity with semantic relevance
    """
    logging.info("Starting RRF fusion with semantic scoring...")
    scores = {}
    
    for query_idx, results in enumerate(result_per_query):
        logging.info(f"Processing results for query {query_idx}: {len(results)} results")
        for rank, chunk in enumerate(results):
            key = chunk["text"]
            
            if key not in scores:
                scores[key] = {"text": key, "score": 0}
            
            # RRF score based on rank
            rrf_score = 1 / (rank + k)
            
            # Semantic similarity score from Qdrant
            semantic_sim_score = chunk.get("score", 0)
            
            # Combine both scores
            scores[key]["score"] += rrf_score + semantic_sim_score
            logging.debug(f"Rank {rank}: RRF={rrf_score:.4f}, Semantic={semantic_sim_score:.4f}")
    
    result = list(scores.values())
    logging.info(f"RRF fusion complete. Total unique chunks: {len(result)}")
    return result


def rerank_chunks(query, chunks):
    """Rerank retrieved chunks using a cross-encoder model"""
    logging.info("Reranking chunks with cross-encoder...")

    pairs = [(query, chunk["text"]) for chunk in chunks]

    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    
    return chunks

def get_top_chunks_reranked(query, chunks, top_k=5):
    reranked = rerank_chunks(query, chunks)

    sorted_chunks = sorted(
        reranked,
        key=lambda x: x["rerank_score"],
        reverse=True
    )
    return sorted_chunks[:top_k]

def chat_loop(openai_client, qdrant_client, collection_name):
    """Chat loop with RRF and semantic scoring"""
    print("\nRAG Chat Ready! Type 'exit' to stop.\n")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            print("Goodbye!")
            break

        # Generate multiple queries
        queries = generate_queries(openai_client, user_question)
        logging.info(f"Generated {len(queries)} queries from user question")

        # Search with each query
        all_chunks = []
        for query in queries:
            results = search(query, openai_client, qdrant_client, collection_name)
            logging.info(f"Retrieved {len(results)} chunks for query: {query}")
            all_chunks.append(results)

        # Apply RRF fusion with semantic scoring
        fused_chunks = rrf_fusion_with_semantic_score(all_chunks)
        
        # Sort and select top 5
        top_chunks = get_top_chunks_reranked(user_question, fused_chunks, top_k=5)

        combined_context = "\n\n".join([c["text"] for c in top_chunks])
        logging.info(f"Combined context length: {len(combined_context)} characters")
        answer = get_ai_response(openai_client, user_question, combined_context)

        print("\nAI:", answer, "\n")
        logging.info(f"User question: {user_question}")
        logging.info(f"AI answer: {answer}")


def main():
    setup_logging()
    logging.info("Starting RAG pipeline with cross-encoder ranking...")

    openai_client = load_openai_client()
    qdrant_client = load_qdrant_client()

    RUN_INGESTION = False
    logging.info(f"Mode: {'INGESTION' if RUN_INGESTION else 'CHAT'}")

    if RUN_INGESTION:
        text = read_pdf(PDF_PATH)
        chunks = get_chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        embeddings = create_embeddings(openai_client, chunks)

        if not check_collection_exist(qdrant_client, COLLECTION_NAME):
            setup_collection(qdrant_client, COLLECTION_NAME, EMBEDDING_DIM)

        upload_vectors(qdrant_client, COLLECTION_NAME, chunks, embeddings, PDF_PATH)
    else:
        if not check_collection_exist(qdrant_client, COLLECTION_NAME):
            raise Exception("Collection not found. Run ingestion first.")

    chat_loop(openai_client, qdrant_client, COLLECTION_NAME)


if __name__ == "__main__":
    main()