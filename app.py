import os
from typing import List, Dict, Any, Tuple

import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import ollama

# Config i putanje
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "fidit_kolegij"

# model za embeddinge, ovaj je okej za performanse
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# mistral je default, al moze i phi3 ako nekome steka komp
DEFAULT_LLM_MODEL = "mistral"
TOP_K = 5

# Heuristika da ne lupa gluposti ako nema nista u bazi
MIN_DOCS_FOR_ANSWER = 2
MAX_DISTANCE_FOR_CONFIDENCE = 0.55  # prag za slicnost, iznad ovoga je vjv irelevantno


# Logika za bazu i model
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_chroma_collection():
    # gasim telemetriju da ne salje bezveze podatke van
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def retrieve_context(collection, embedder, query: str, top_k: int = TOP_K):
    # pretvori upit u vektor i trazi najslicnije u chromi
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return docs, metas, dists


def format_citations(metas: List[Dict[str, Any]]) -> str:
    # filtriraj duplate da ne ispisuje istu stranicu vise puta
    seen = set()
    lines = []
    for m in metas:
        key = (m.get("source"), m.get("page"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {m.get('source')} (str. {m.get('page')})")
    return "\n".join(lines) if lines else "- (nema citata)"


def should_ask_clarifying_question(docs: List[str], dists: List[float]) -> Tuple[bool, str]:
    # provjera jel uopce imamo pametne podatke za odgovor
    if len(docs) < MIN_DOCS_FOR_ANSWER:
        return True, "Nemam dovoljno materijala iz baze za siguran odgovor."
    if not dists:
        return True, "Nemam dovoljno signala za odgovor."
    
    best = dists[0]
    if best > MAX_DISTANCE_FOR_CONFIDENCE:
        # ako je distance prevelik, znaci da je fulao temu
        return True, "Pitanje je preširoko ili nije jasno povezano s materijalima."
    return False, ""


def generate_clarifying_questions(user_q: str) -> str:
    # fiksni odgovor ako sustav nije siguran, cisto da usmjeri korisnika
    return (
        "Mogu pomoći, ali trebam malo preciznije:\n"
        "1) Na koje poglavlje / temu iz materijala misliš?\n"
        "2) Je li pitanje teorijsko ili želiš primjer/zadatak?\n"
        "3) Imaš li konkretan pojam/definiciju koju treba objasniti?"
    )


def build_prompt(user_q: str, context_docs: List[str], context_metas: List[Dict[str, Any]]) -> str:
    # slazemo prompt za LLM s pravilima ponasanja
    citations = format_citations(context_metas)
    context_block = "\n\n---\n\n".join(context_docs)

    return f"""
Ti si AI asistent za podršku učenju.
Odgovaraj prvenstveno na temelju priloženih službenih materijala kolegija.
Ako informacija nije u kontekstu, jasno reci da nisi siguran i predloži što student može provjeriti.

PRAVILA:
- Ne izmišljaj činjenice.
- Budi jasan, pedagoški, sa primjerom ako je prikladno.
- Na kraju prikaži "IZVORI" koje si koristio.
- Ne traži niti pohranjuj osobne podatke.

PITANJE STUDENTA:
{user_q}

KONTEKST (iz vektorske baze):
{context_block}

IZVORI (iz metapodataka):
{citations}

Odgovori na hrvatskom.
"""


def call_llm(model_name: str, prompt: str) -> str:
    # spoji se na lokalni ollama server
    resp = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": 0.2, # niska temperatura da ne halucinira previse
            "num_predict": 220 # max tokena da model ne piše "cijeli roman"
        }
    )
    return resp["response"].strip()


# Streamlit UI dio
st.set_page_config(page_title="FIDIT AI asistent", layout="wide")

st.title("FIDIT - AI asistent za podršku učenju")

with st.expander("Važne informacije (transparentnost)", expanded=True):
    st.markdown(
        """
**Komuniciraš s AI sustavom.** Odgovori su namijenjeni isključivo kao pomoć u učenju i **ne zamjenjuju nastavu, službene materijale ni profesora**.

**Privatnost:** sustav je prototip i **ne prikuplja osobne podatke** niti sprema razgovore u bazu.  
Ako uneseš osobne podatke, preporuka je da ih odmah ukloniš iz upita.

**Izvori:** odgovori se temelje prvenstveno na službenim materijalima kolegija.  
Kada se koristi sadržaj, sustav prikazuje citate (izvor + stranica).
        """
    )

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Postavke")
    model_name = st.text_input("Ollama model", value=DEFAULT_LLM_MODEL, help="Npr. mistral... mora bit upaljen u ollami")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("Brzo, ali manje")
    with c2:
        # Desno poravnanje teksta
        st.markdown("<p style='text-align: right;'>Sporo, ali više</p>", unsafe_allow_html=True)

    top_k = st.slider("", 2, 10, TOP_K, label_visibility="collapsed")

    if st.button("Provjeri kolekciju"):
        try:
            collection = load_chroma_collection()
            st.success(f"Kolekcija '{COLLECTION_NAME}' ima {collection.count()} dokumenata.")
        except Exception as e:
            st.error(f"Greška kod baze: {e}")

with col1:
    st.subheader("Chat")
    if "history" not in st.session_state:
        st.session_state.history = []

    user_q = st.text_input("Upiši pitanje:", placeholder="Npr. Što je digitalna inovacija?")

    if st.button("Pošalji") and user_q.strip():
        # prvo dohvati podatke
        embedder = load_embedder()
        collection = load_chroma_collection()

        docs, metas, dists = retrieve_context(collection, embedder, user_q, top_k=top_k)

        # odluci jel imamo dovoljno za odgovor
        need_clarify, reason = should_ask_clarifying_question(docs, dists)

        if need_clarify:
            answer = f"Napomena: {reason}\n\n" + generate_clarifying_questions(user_q)
            citations = ""
            debug = {"distances": dists[:3], "top_sources": [(m.get("source"), m.get("page")) for m in metas[:3]]}
        else:
            prompt = build_prompt(user_q, docs, metas)
            try:
                answer = call_llm(model_name, prompt)
            except Exception as e:
                answer = f"Greška: Ne mogu doći do Ollama modela '{model_name}'. Provjeri jel pokrenut. Detalji: {e}"
            citations = format_citations(metas)
            debug = {"distances": dists[:3], "top_sources": [(m.get("source"), m.get("page")) for m in metas[:3]]}

        st.session_state.history.append((user_q, answer, citations, debug))

    # ispis chat-a unazad
    for q, a, c, debug in reversed(st.session_state.history):
        st.markdown(f"**Student:** {q}")
        st.markdown(f"**Asistent:**\n\n{a}")
        if c:
            st.markdown("**IZVORI:**")
            st.markdown(c)
        with st.expander("Debug info (interni podaci)", expanded=False):
            st.write(debug)
        st.divider()