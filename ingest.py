import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Config
DATA_DIR = os.path.join("data", "materijali")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "fidit_kolegij"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # free, small, good baseline

CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 200    # isto chars


# Helpers
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def read_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number_1based, page_text)
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = clean_text(txt)
        if txt:
            pages.append((i + 1, txt))
    return pages


def get_pdf_files(data_dir: str) -> List[str]:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Ne postoji folder: {data_dir}")
    pdfs = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".pdf"):
            pdfs.append(os.path.join(data_dir, fn))
    return sorted(pdfs)


def main():
    # 1) load embedding 
    print(f"[1/5] Ucitavam embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # 2) chromadb lokalno inicija.
    print(f"[2/5] Inicijaliziram ChromaDB u: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # 3) citamo pdf-ove
    pdf_files = get_pdf_files(DATA_DIR)
    if not pdf_files:
        print(f"Nema PDF-ova u {DATA_DIR}. Ubaci barem jedan PDF i probaj opet.")
        return

    print(f"[3/5] Pronadjeno PDF-ova: {len(pdf_files)}")
    total_chunks = 0


if __name__ == "__main__":
    main()