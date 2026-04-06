import os
import logging
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import time

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
def get_chunk_text(text, chunk_size=500, overlap=50):
    logging.info("Chunking text...")
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    logging.info(f"Total chunks created: {len(chunks)}")
    return chunks

chunks = get_chunk_text(text)

# ---------------- EMBEDDINGS ----------------
def get_embeddings(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return None

logging.info("Creating embeddings...")
start_time = time.time()

embeddings = []
for i, chunk in enumerate(chunks):
    logging.info(f"Creating embedding for chunk {i}")
    emb = get_embeddings(chunk)
    if emb:
        embeddings.append(emb)

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
            payload={"text": chunks[i]}
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
    query_vector = get_embeddings(query)

    hits = client_qdrant.query_points(
        collection_name="pdf_docs",
        query=query_vector,
        limit=3
    )

    logging.info(f"Search returned {len(hits.points)} results")
    return hits

# ---------------- RUN SEARCH ----------------
context = search("What is this PDF about?")
logging.info("Search finished")

print(context)