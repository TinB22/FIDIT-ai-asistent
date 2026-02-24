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
DATA_DIR = os.path.join("data", "materials")
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
    
    
    # 4) build-amo bazu znanja (file-ove)
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        source_name = os.path.basename(pdf_path)
        print(f"  - Citanje: {source_name}")

        pages = read_pdf_pages(pdf_path)
        for page_no, page_text in pages:
            chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for ci, chunk in enumerate(chunks):
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                documents.append(chunk)
                metadatas.append({
                    "source": source_name,
                    "page": page_no,
                    "chunk_index": ci,
                    "source_path": pdf_path
                })
                total_chunks += 1

    if total_chunks == 0:
        print("Nisam izvukao tekst (0 chunkova). Provjeri PDF (možda je sken bez teksta).")
        return

    print(f"[4/5] Ukupno chunkova: {total_chunks}. Radim embedding...")


    # 5) embeding i upsert (update + insert)
    embeddings = embedder.encode(documents, show_progress_bar=True, normalize_embeddings=True).tolist()

    print("[5/5] Spremanje u Chroma kolekciju...")
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("\n Gotovo!")
    print(f"Kolekcija: {COLLECTION_NAME}")
    print(f"Chroma path: {CHROMA_DIR}")
    print(f"Broj dokumenata u kolekciji: {collection.count()}")


if __name__ == "__main__":
    main()