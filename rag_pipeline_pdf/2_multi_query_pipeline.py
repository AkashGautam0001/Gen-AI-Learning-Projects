"""
Multi Query RAG Pipeline - Multiple query generation with deduplication
"""
import logging
from rag_utils import (
    setup_logging, load_openai_client, load_qdrant_client, read_pdf,
    get_chunk_text, create_embeddings, setup_collection, upload_vectors,
    search, get_ai_response, check_collection_exist, generate_queries,
    deduplicate_chunks,
    PDF_PATH, COLLECTION_NAME, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP
)


def chat_loop(openai_client, qdrant_client, collection_name):
    """Chat loop with multi-query generation"""
    print("\nRAG Chat Ready! Type 'exit' to stop.\n")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            print("Goodbye!")
            break

        # Generate multiple queries
        queries = generate_queries(openai_client, user_question)
        logging.info(f"Generated {len(queries)} queries from user question")

        # Search for each query
        all_chunks = []
        for query in queries:
            results = search(query, openai_client, qdrant_client, collection_name)
            logging.info(f"Retrieved {len(results)} chunks for query: {query}")
            all_chunks.extend(results)

        # Deduplicate and combine
        unique_chunks = deduplicate_chunks(all_chunks)
        combined_context = "\n\n".join([c["text"] for c in unique_chunks])
        answer = get_ai_response(openai_client, user_question, combined_context)

        print("\nAI:", answer, "\n")
        logging.info(f"User question: {user_question}")
        logging.info(f"AI answer: {answer}")


def main():
    setup_logging()
    logging.info("Starting RAG pipeline...")

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