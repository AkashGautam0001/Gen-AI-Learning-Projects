"""
Simple RAG Pipeline - Single query semantic search
"""
import logging
from rag_utils import (
    setup_logging, load_openai_client, load_qdrant_client, read_pdf,
    get_chunk_text, create_embeddings, setup_collection, upload_vectors,
    search, get_ai_response,
    PDF_PATH, COLLECTION_NAME, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP
)


def chat_loop(openai_client, qdrant_client, collection_name):
    """Chat loop for simple semantic search"""
    print("\nRAG Chat Ready! Type 'exit' to stop.\n")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            print("Goodbye!")
            break

        # Search with single query
        results = search(user_question, openai_client, qdrant_client, collection_name)
        context = "\n\n".join([r["text"] for r in results])
        answer = get_ai_response(openai_client, user_question, context)

        print("\nAI:", answer, "\n")
        logging.info(f"User question: {user_question}")
        logging.info(f"AI answer: {answer}")


def main():
    setup_logging()
    
    openai_client = load_openai_client()
    qdrant_client = load_qdrant_client()

    # Extract and process PDF
    text = read_pdf(PDF_PATH)
    chunks = get_chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = create_embeddings(openai_client, chunks)

    # Setup and upload to Qdrant
    setup_collection(qdrant_client, COLLECTION_NAME, EMBEDDING_DIM)
    upload_vectors(qdrant_client, COLLECTION_NAME, chunks, embeddings, PDF_PATH)

    # Start chat
    chat_loop(openai_client, qdrant_client, COLLECTION_NAME)


if __name__ == "__main__":
    main()