from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent/"Attention_That_You_Need.pdf"

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents=docs)

# chunks = []
# for i, text in enumerate(texts):
#     chunk = {
#         "id": f"chunk-{i}",
#         "text": text,
#         "metadata": {"source": f"{pdf_path.name}-page-{docs[0].metadata['page_number']}"}
#     }
#     chunks.append(chunk)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vectoreStore = QdrantVectorStore.from_documents(
    documents=texts,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="my_documents"
)

print(vectoreStore)