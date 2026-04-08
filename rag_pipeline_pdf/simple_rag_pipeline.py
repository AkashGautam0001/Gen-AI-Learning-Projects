import os
import logging
import time
import re
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

PDF_PATH = "PDF-Guide-Node-Andrew-Mead-v3.pdf"
COLLECTION_NAME = "pdf_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ---------------- LOGGING SETUP ----------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


# ---------------- CLIENTS ----------------
def load_openai_client():
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")
    if not api_key:
        logging.error("API key not found!")
        raise ValueError("OPEN_API_KEY not set")
    return OpenAI(api_key=api_key)


def load_qdrant_client(host="localhost", port=6333):
    logging.info("Connecting to Qdrant...")
    return QdrantClient(host=host, port=port)


# ---------------- PDF ----------------
def read_pdf(pdf_path):
    logging.info(f"Reading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        try:
            text += page.extract_text()
            logging.info(f"Extracted page {i}")
        except Exception as e:
            logging.error(f"Error extracting page {i}: {e}")
    logging.info(f"Total text length: {len(text)}")
    return text


# ---------------- CHUNKING ----------------
def get_chunk_text(text, chunk_size=800, overlap=50):
    logging.info("Chunking text...")
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    logging.info(f"Total chunks created: {len(chunks)}")
    return chunks


# ---------------- EMBEDDINGS ----------------
def get_embeddings_batch(openai_client, text_list):
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
    logging.info("Creating embeddings...")
    start_time = time.time()
    embeddings = get_embeddings_batch(openai_client, chunks)
    logging.info(f"Embeddings created in {time.time() - start_time:.2f} seconds")
    return embeddings


# ---------------- QDRANT COLLECTION ----------------
def setup_collection(qdrant_client, collection_name, vector_size):
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logging.info("Collection created")
    except Exception:
        logging.warning("Collection may already exist")


# ---------------- UPLOAD ----------------
def upload_vectors(qdrant_client, collection_name, chunks, embeddings, source):
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


# ---------------- SEARCH ----------------
def search(query, openai_client, qdrant_client, collection_name):
    logging.info(f"Searching for query: {query}")
    query_vector = get_embeddings_batch(openai_client, [query])[0]
    hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5
    )
    logging.info(f"Search returned {len(hits.points)} results")
    return "\n\n".join([hit.payload["text"] for hit in hits.points])


# ---------------- GPT RESPONSE ----------------
def get_ai_response(openai_client, user_question, context):
    system_prompt = f"""
    You are a helpful AI assistant.
    Answer the question ONLY using the provided context.
    If the answer is not in the context, say: "Answer not found in document".

    Context:
    {context}

    Question: {user_question}
    Answer:
    """
    logging.info("Sending query to GPT...")
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers from provided PDF context only."},
            {"role": "user", "content": system_prompt}
        ]
    )
    return response.choices[0].message.content


# ---------------- CHAT LOOP ----------------
def chat_loop(openai_client, qdrant_client, collection_name):
    print("\nRAG Chat Ready! Type 'exit' to stop.\n")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            print("Goodbye!")
            break

        context = search(user_question, openai_client, qdrant_client, collection_name)
        answer = get_ai_response(openai_client, user_question, context)

        print("\nAI:", answer, "\n")
        logging.info(f"User question: {user_question}")
        logging.info(f"AI answer: {answer}")


# ---------------- MAIN ----------------
def main():
    setup_logging()
    logging.info("Starting RAG pipeline...")

    openai_client = load_openai_client()
    qdrant_client = load_qdrant_client()

    text = read_pdf(PDF_PATH)
    chunks = get_chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = create_embeddings(openai_client, chunks)

    setup_collection(qdrant_client, COLLECTION_NAME, EMBEDDING_DIM)
    upload_vectors(qdrant_client, COLLECTION_NAME, chunks, embeddings, PDF_PATH)

    chat_loop(openai_client, qdrant_client, COLLECTION_NAME)


if __name__ == "__main__":
    main()