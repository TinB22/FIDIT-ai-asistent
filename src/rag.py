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
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


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
    q = query.lower()
    matched_sources: List[str] = []
    for rule in rules:
        kws = [k.lower() for k in rule.get("keywords", [])]
        if any(kw in q for kw in kws):
            matched_sources.extend(rule.get("sources", []))
    matched_sources = list(dict.fromkeys(matched_sources))
    return matched_sources if matched_sources else None


def retrieve_context(
    collection,
    embedder,
    query: str,
    top_k: int,
    routed_sources: Optional[List[str]] = None,
):
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()

    where = None
    if routed_sources:
        if len(routed_sources) == 1:
            where = {"source": routed_sources[0]}
        else:
            where = {"$or": [{"source": s} for s in routed_sources]}

    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return docs, metas, dists, where


def format_citations(metas: List[Dict[str, Any]]) -> str:
    seen = set()
    lines = []
    for m in metas:
        source = m.get("source", "Nepoznat izvor")
        clean_source = source.split(". ", 1)[-1] if ". " in source else source
        key = (clean_source, m.get("page"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {clean_source} (str. {m.get('page')})")
    return "\n".join(lines) if lines else "- (nema citata)"


def should_clarify(docs: List[str], dists: List[float]) -> Tuple[bool, str]:
    if len(docs) < MIN_DOCS_FOR_ANSWER:
        return True, "Nemam dovoljno materijala iz baze za siguran odgovor."
    if not dists:
        return True, "Nemam dovoljno signala za odgovor."
    if dists[0] > MAX_DISTANCE_FOR_CONFIDENCE:
        return True, "Pitanje nije jasno povezano s materijalima kolegija."
    return False, ""


def clarifying_questions() -> str:
    return (
        "Mogu pomoći, ali trebam malo preciznije:\n"
        "1) Na koje poglavlje / temu iz materijala misliš?\n"
        "2) Je li pitanje teorijsko ili želiš primjer/zadatak?\n"
        "3) Koji je kontekst (npr. startup, ePlaćanje, sigurnost, BMC...)?"
    )


def build_prompt(
    user_q: str,
    context_docs: List[str],
    context_metas: List[Dict[str, Any]],
    max_context_chars: int = 7000,
    verbosity: str = "medium",  # "short" | "medium" | "long"
) -> str:
    joined = "\n\n---\n\n".join(context_docs)
    context_block = joined[:max_context_chars]

    if verbosity == "short":
        length_rules = (
            "- Napiši KRATAK odgovor: ukupno 4–6 rečenica.\n"
            "- DEFINICIJA: 1–2 rečenice.\n"
            "- OBJAŠNJENJE: 2–3 rečenice.\n"
            "- PRIMJER: 1 rečenica.\n"
        )
    elif verbosity == "long":
        length_rules = (
            "- Napiši OPŠIRNIJI odgovor, ali bez nepotrebnog ponavljanja.\n"
            "- DEFINICIJA: 2–3 rečenice.\n"
            "- OBJAŠNJENJE: 5–8 rečenica (može u bulletima).\n"
            "- PRIMJER: 2–4 rečenice.\n"
        )
    else:
        length_rules = (
            "- Napiši SREDNJE DUG odgovor.\n"
            "- DEFINICIJA: 1–2 rečenice.\n"
            "- OBJAŠNJENJE: 3–5 rečenica.\n"
            "- PRIMJER: 1–2 rečenice.\n"
        )

    return f"""
Ti si stručni akademski asistent na fakultetu (FIDIT). Tvoj zadatak je pomoći studentu da razumije koncepte digitalne ekonomije.

STRUKTURA ODGOVORA (Obavezno prati ovaj redoslijed):
1. **DEFINICIJA**
2. **OBJAŠNJENJE**
3. **PRIMJER**

PRAVILA:
- Odgovaraj ISKLJUČIVO na temelju priloženog KONTEKSTA.
- Koristi se službenim hrvatskim jezikom.
- Profesionalan, ali pristupačan ton.
- Ako informacija nedostaje, reci: "Na temelju dostupnih materijala, ne mogu u potpunosti odgovoriti..."
- Ne započinji novu sekciju ako nema dovoljno mjesta; radije skrati zadnju rečenicu i završi odgovor potpunom rečenicom.
{length_rules}

PITANJE STUDENTA:
"{user_q}"

KONTEKST IZ MATERIJALA:
{context_block}

Odgovori na hrvatskom jeziku.
"""