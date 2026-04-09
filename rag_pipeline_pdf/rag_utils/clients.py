"""
Client initialization for OpenAI and Qdrant
"""
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient


def load_openai_client():
    """Load and validate OpenAI client"""
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")
    if not api_key:
        logging.error("API key not found!")
        raise ValueError("OPEN_API_KEY not set")
    return OpenAI(api_key=api_key)


def load_qdrant_client(host="localhost", port=6333):
    """Connect to Qdrant database"""
    logging.info("Connecting to Qdrant...")
    return QdrantClient(host=host, port=port)
