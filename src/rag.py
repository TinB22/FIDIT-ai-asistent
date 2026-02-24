import json
import os
import re
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
    q = query.lower()
    matched_sources: List[str] = []
    for rule in rules:
        kws = [k.lower() for k in rule.get("keywords", [])]
        if any(kw in q for kw in kws):
            matched_sources.extend(rule.get("sources", []))
    matched_sources = list(dict.fromkeys(matched_sources))
    return matched_sources if matched_sources else None


def _split_query_terms(query: str) -> List[str]:
    q = normalize_query(query)
    tokens = re.findall(r"[a-zA-ZčćđšžČĆĐŠŽ0-9]+", q)
    stop = {
        "što", "sta", "je", "su", "se", "u", "na", "za", "od", "do", "i", "ili", "a",
        "kako", "koji", "koja", "koje", "kada", "gdje", "zasto", "zašto",
        "objasni", "navedi", "primjer", "primjeri", "definicija", "pojam",
        "molim", "možeš", "mozete", "možete",
    }
    terms = [t for t in tokens if len(t) >= 4 and t not in stop]
    return list(dict.fromkeys(terms))


def _rerank_by_term_overlap(
    docs: List[str],
    metas: List[Dict[str, Any]],
    dists: List[float],
    query: str,
) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    terms = _split_query_terms(query)
    if not terms:
        return docs, metas, dists

    scored = []
    for i, doc in enumerate(docs):
        text = (doc or "").lower()
        overlap = sum(1 for t in terms if t in text)

        penalty = 0.08 if overlap == 0 else 0.0
        bonus = -0.02 * min(overlap, 5)

        new_dist = float(dists[i]) + penalty + bonus
        scored.append((new_dist, i))

    scored.sort(key=lambda x: x[0])
    new_docs = [docs[i] for _, i in scored]
    new_metas = [metas[i] for _, i in scored]
    new_dists = [dists[i] for _, i in scored]
    return new_docs, new_metas, new_dists


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

    if docs and metas and dists:
        docs, metas, dists = _rerank_by_term_overlap(docs, metas, dists, query)

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
    joined = "\n\n---\n\n".join(context_docs)
    context_block = joined[:4000]

    return f"""
Ti si AI asistent za podršku učenju.
Odgovaraj prvenstveno na temelju priloženih službenih materijala kolegija.
Ako informacija nije u kontekstu ili je kontekst očito o drugoj temi, jasno reci da nisi siguran i postavi podpitanje.

PRAVILA:
- Ne izmišljaj činjenice.
- Drži se teme upita.
- Ne uvodi pojmove koji nisu u kontekstu.
- Budi jasan i pedagoški nastrojen.
- Ne traži niti pohranjuj osobne podatke.

PITANJE STUDENTA:
{user_q}

KONTEKST (iz vektorske baze):
{context_block}

Odgovori na hrvatskom.
"""


def extractive_fallback_answer(user_q: str, docs: List[str], max_chars: int = 900) -> str:
    """
    Fallback bez LLM-a (ako Ollama pukne).
    Vrati najrelevantnije rečenice iz dohvaćenih chunkova.
    """
    if not docs:
        return "Ne mogu generirati odgovor jer nema dovoljno relevantnih materijala u bazi."

    text = " ".join(docs)
    text = re.sub(r"\s+", " ", text).strip()

    # razbij na "rečenice" grubo
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    terms = _split_query_terms(user_q)

    def score(s: str) -> int:
        s_low = s.lower()
        return sum(1 for t in terms if t in s_low)

    ranked = sorted(sentences, key=score, reverse=True)
    picked = []
    total = 0
    for s in ranked:
        s = s.strip()
        if not s:
            continue
        # preskoči jako kratko
        if len(s) < 40:
            continue
        if s in picked:
            continue
        if total + len(s) + 1 > max_chars:
            break
        picked.append(s)
        total += len(s) + 1
        if len(picked) >= 5:
            break

    if not picked:
        # fallback: vrati početak konteksta
        return (docs[0][:max_chars]).strip()

    return " ".join(picked).strip()