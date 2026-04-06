import os
import logging
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import time
import re

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting RAG pipeline...")

# ---------------- LOAD ENV ----------------
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

if not api_key:
    logging.error("API key not found!")
    raise ValueError("OPEN_API_KEY not set")

client = OpenAI(api_key=api_key)

# ---------------- READ PDF ----------------
logging.info("Reading PDF...")
reader = PdfReader("PDF-Guide-Node-Andrew-Mead-v3.pdf")

text = ""
for i, page in enumerate(reader.pages):
    try:
        page_text = page.extract_text()
        text += page_text
        logging.info(f"Extracted page {i}")
    except Exception as e:
        logging.error(f"Error extracting page {i}: {e}")

logging.info(f"Total text length: {len(text)}")

# ---------------- CHUNKING ----------------
def get_chunk_text(text, chunk_size=800, overlap=50):
    logging.info("Chunking text...")

    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk)+len(sentence) < chunk_size:
            logging.info("Sentence if : "+sentence)
            current_chunk += sentence + " "
        else:
            logging.info("Sentence else : "+sentence)
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logging.info(f"Total chunks created: {len(chunks)}")

    return chunks
chunks = get_chunk_text(text, 800, 100)

# ---------------- EMBEDDINGS ----------------
def get_embeddings_batch(text_list):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text_list
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return []

logging.info("Creating embeddings...")
start_time = time.time()

embeddings = get_embeddings_batch(chunks)

logging.info(f"Embeddings created in {time.time() - start_time:.2f} seconds")

# ---------------- QDRANT ----------------
logging.info("Connecting to Qdrant...")
client_qdrant = QdrantClient(host="localhost", port=6333)

try:
    client_qdrant.create_collection(
        collection_name="pdf_docs",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    logging.info("Collection created")
except Exception as e:
    logging.warning("Collection may already exist")

# ---------------- UPLOAD ----------------
logging.info("Uploading vectors to Qdrant...")
points = []

for i, vector in enumerate(embeddings):
    points.append(
        PointStruct(
            id=i,
            vector=vector,
            payload={
                "text": chunks[i],
                "source": "PDF-Guide-Node-Andrew-Mead-v3.pdf",
                "chunk_id": i,
            }
        )
    )

client_qdrant.upsert(
    collection_name="pdf_docs",
    points=points
)

logging.info("Upload complete!")

# ---------------- SEARCH ----------------
def search(query):
    logging.info(f"Searching for query: {query}")
    query_vector = get_embeddings_batch([query])[0]

    hits = client_qdrant.query_points(
        collection_name="pdf_docs",
        query=query_vector,
        limit=5
    )

    logging.info(f"Search returned {len(hits.points)} results")
    return "\n\n".join([hit.payload["text"] for hit in hits.points])

# ---------------- CHAT LOOP ----------------
print("\nRAG Chat Ready! Type 'exit' to stop.\n")

while True:
    user_question = input("You: ")

    if user_question.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve context from Qdrant
    context = search(user_question)

    SYSTEM_PROMPT = f"""
    You are a helpful AI assistant.
    Answer the question ONLY using the provided context.
    If the answer is not in the context, say: "Answer not found in document".

    Context:
    {context}

    Question: {user_question}
    Answer:
    """

    logging.info("Sending query to GPT...")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers from provided PDF context only."},
            {"role": "user", "content": SYSTEM_PROMPT}
        ]
    )

    answer = response.choices[0].message.content

    print("\nAI:", answer, "\n")

    logging.info(f"User question: {user_question}")
    logging.info(f"AI answer: {answer}")

print(response.choices[0].message.content)