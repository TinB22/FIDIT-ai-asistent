import json
import os
from typing import List, Dict, Any, Tuple, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "fidit_kolegij"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MIN_DOCS_FOR_ANSWER = 2
MAX_DISTANCE_FOR_CONFIDENCE = 0.55

ROUTING_PATH = os.path.join("data", "mappings", "routing.json")


def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def load_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def normalize_query(q: str) -> str:
    return " ".join(q.strip().lower().split())


def load_routing_rules() -> List[Dict[str, Any]]:
    if not os.path.exists(ROUTING_PATH):
        return []
    try:
        with open(ROUTING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("rules", [])
    except Exception:
        return []


def route_sources(query: str, rules: List[Dict[str, Any]]) -> Optional[List[str]]:
    """
    Returns list of sources to filter on, or None if no routing match.
    """
    q = query.lower()
    matched_sources = []
    for rule in rules:
        kws = [k.lower() for k in rule.get("keywords", [])]
        if any(kw in q for kw in kws):
            matched_sources.extend(rule.get("sources", []))
    # unique
    matched_sources = list(dict.fromkeys(matched_sources))
    return matched_sources if matched_sources else None


def retrieve_context(collection, embedder, query: str, top_k: int, routed_sources: Optional[List[str]] = None):
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()

    where = None
    if routed_sources:
        # Chroma "where" ne podržava uvijek $in ovisno o verziji, pa radimo jednostavno:
        # Ako ima više izvora, probamo $or; ako ne radi u vašoj verziji, fallback na single-source.
        if len(routed_sources) == 1:
            where = {"source": routed_sources[0]}
        else:
            where = {"$or": [{"source": s} for s in routed_sources]}

    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return docs, metas, dists, where


def format_citations(metas: List[Dict[str, Any]]) -> str:
    seen = set()
    lines = []
    for m in metas:
        key = (m.get("source"), m.get("page"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {m.get('source')} (str. {m.get('page')})")
    return "\n".join(lines) if lines else "- (nema citata)"


def should_clarify(docs: List[str], dists: List[float]) -> Tuple[bool, str]:
    if len(docs) < MIN_DOCS_FOR_ANSWER:
        return True, "Nemam dovoljno materijala iz baze za siguran odgovor."
    if not dists:
        return True, "Nemam dovoljno signala za odgovor."
    if dists[0] > MAX_DISTANCE_FOR_CONFIDENCE:
        return True, "Pitanje je preširoko ili nije jasno povezano s materijalima."
    return False, ""


def clarifying_questions() -> str:
    return (
        "Mogu pomoći, ali trebam malo preciznije:\n"
        "1) Na koje poglavlje / temu iz materijala misliš?\n"
        "2) Je li pitanje teorijsko ili želiš primjer/zadatak?\n"
        "3) Koji je kontekst (npr. startup, ePlaćanje, sigurnost, BMC...)?"
    )


def build_prompt(user_q: str, context_docs: List[str], context_metas: List[Dict[str, Any]]) -> str:
    citations = format_citations(context_metas)
    joined = "\n\n---\n\n".join(context_docs)
    context_block = joined[:5000] # Malo smo povećali limit za bogatiji kontekst

    return f"""
Ti si stručni akademski asistent na fakultetu (FIDIT). Tvoj zadatak je pomoći studentu da duboko razumije koncepte digitalne ekonomije i inovacija.

UPUTE ZA ODGOVARANJE:
1. DEFINICIJA: Počni s jasnom i stručnom definicijom pojma iz materijala.
2. OBJAŠNJENJE: Objasni "zašto" i "kako" (ne samo "što"). Poveži pojam sa širim kontekstom digitalne transformacije.
3. PRIMJER: Navedi konkretan primjer iz materijala ili realnog poslovanja koji ilustrira pojam.
4. DOSLJEDNOST: Koristi isključivo priloženi kontekst. Ako u kontekstu nema dovoljno informacija, navedi što nedostaje.

PITANJE STUDENTA:
"{user_q}"

KONTEKST IZ MATERIJALA:
{context_block}

IZVORI:
{citations}

Odgovori na hrvatskom jeziku, profesionalnim i poticajnim tonom.
"""