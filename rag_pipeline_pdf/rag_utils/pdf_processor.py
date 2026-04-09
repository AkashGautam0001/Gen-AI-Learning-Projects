"""
PDF processing utilities
"""
import logging
import re
from pypdf import PdfReader


def read_pdf(pdf_path):
    """Extract text from PDF file"""
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


def get_chunk_text(text, chunk_size=800, overlap=50):
    """Split text into overlapping chunks"""
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
