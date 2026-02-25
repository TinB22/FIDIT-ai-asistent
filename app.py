import json
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import ollama

# Importi iz vlastitih skripti (lokalni moduli)
from src.rag import (
    load_collection,
    load_embedder,
    load_routing_rules,
    route_sources,
    retrieve_context,
    should_clarify,
    clarifying_questions,
    build_prompt,
    format_citations,
    normalize_query,
)
from src.privacy import redact_personal_data
from src.cache import stable_key, get_cached_answer, set_cached_answer


DEFAULT_LLM_MODEL = "mistral"
DEFAULT_TOP_K = 5

# Fiksna temperatura za stabilne odgovore, num_predict uklonjen za prirodnu duljinu
FIXED_TEMPERATURE = 0.2


@st.cache_resource
def get_embedder():
    return load_embedder()


@st.cache_resource
def get_collection():
    return load_collection()


@st.cache_resource
def get_routing_rules():
    return load_routing_rules()


def load_questions() -> List[Dict[str, Any]]:
    path = os.path.join("data", "questions", "questions.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("questions", [])
    except Exception:
        return []


def call_llm(model_name: str, prompt: str, temperature: float) -> str:
    # Poziv bez num_predict omogućuje modelu da prirodno završi misao
    resp = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": temperature,
        },
    )
    return resp["response"].strip()


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="FIDIT AI asistent", page_icon="", layout="wide")
st.title(" FIDIT – AI asistent za podršku učenju")

with st.expander(" Važne informacije (transparentnost)", expanded=True):
    st.markdown(
        """
**Komuniciraš s AI sustavom.** Odgovori su namijenjeni isključivo kao pomoć u učenju i **ne zamjenjuju nastavu, službene materijale ni profesora**.

**Privatnost:** sustav je prototip i **ne prikuplja osobne podatke** niti sprema razgovore u bazu.  
Ako uneseš osobne podatke, preporuka je da ih ukloniš iz upita.

**Izvori:** odgovori se temelje prvenstveno na službenim materijalima kolegija.  
Kada se koristi sadržaj, sustav prikazuje citate (izvor + stranica).
        """
    )

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Postavke")

    model_name = st.text_input("Ollama model", value=DEFAULT_LLM_MODEL)
    top_k = st.slider("Broj dohvaćenih odlomaka (top_k)", 2, 10, DEFAULT_TOP_K)

    # Temperature i num_predict maknuti iz sučelja radi čišćeg dizajna
    show_debug = st.checkbox("Prikaži debug", value=False)

    if st.button("Provjeri kolekciju"):
        try:
            collection = get_collection()
            st.success(f"Kolekcija ima {collection.count()} dokumenata.")
        except Exception as e:
            st.error(f"Greška: {e}")

    st.divider()
    st.subheader("Test pitanja")

    questions = load_questions()
    q_labels = ["(odaberi)"] + [f"{q['id']} – {q['question']}" for q in questions]
    selected = st.selectbox("Odaberi pitanje iz baze:", q_labels, index=0)

with col1:
    st.subheader("Chat")

    if "history" not in st.session_state:
        st.session_state.history = []

    prefill = ""
    selected_q = None
    if selected and selected != "(odaberi)":
        qid = selected.split(" – ", 1)[0]
        selected_q = next((q for q in questions if q["id"] == qid), None)
        if selected_q:
            prefill = selected_q["question"]

    user_q = st.text_input("Upiši pitanje:", value=prefill, placeholder="Npr. Što je digitalna inovacija?")

    if st.button("Pošalji") and user_q.strip():
        # 1) privacy redact
        redacted_q, redaction_report = redact_personal_data(user_q)

        # 2) setup
        embedder = get_embedder()
        collection = get_collection()
        rules = get_routing_rules()

        # 3) routing
        routed_sources = route_sources(redacted_q, rules)

        # 4) Retrieval cache
        norm_q = normalize_query(redacted_q)
        retrieval_key = f"retrieval::{norm_q}::{top_k}::{routed_sources}"
        if "retrieval_cache" not in st.session_state:
            st.session_state.retrieval_cache = {}

        if retrieval_key in st.session_state.retrieval_cache:
            docs, metas, dists, where = st.session_state.retrieval_cache[retrieval_key]
        else:
            docs, metas, dists, where = retrieve_context(
                collection, embedder, redacted_q, top_k=top_k, routed_sources=routed_sources
            )
            st.session_state.retrieval_cache[retrieval_key] = (docs, metas, dists, where)

        # 5) Clarify check
        need_clarify, reason = should_clarify(docs, dists)
        if need_clarify:
            answer = f" {reason}\n\n{clarifying_questions()}"
            citations = ""
        else:
            prompt = build_prompt(redacted_q, docs, metas)

            # 6) Disk cache za test pitanja
            cache_hit = None
            cache_key = None
            if selected_q:
                # num_predict maknut iz ključa jer se više ne koristi
                cache_key = stable_key(
                    selected_q["id"],
                    norm_q,
                    model_name,
                    str(top_k),
                    str(FIXED_TEMPERATURE),
                    str(collection.count()),
                )
                cache_hit = get_cached_answer(cache_key)

            if cache_hit:
                answer = cache_hit["answer"]
                citations = cache_hit.get("citations", format_citations(metas))
            else:
                try:
                    # Pozivamo s fiksnom temperaturom
                    answer = call_llm(model_name, prompt, temperature=FIXED_TEMPERATURE)
                except Exception as e:
                    answer = f" Ne mogu pozvati Ollama model '{model_name}'. Greška: {e}"
                citations = format_citations(metas)

                if cache_key and selected_q:
                    set_cached_answer(cache_key, {"answer": answer, "citations": citations})

        debug = {
            "redaction": redaction_report,
            "routed_sources": routed_sources,
            "where_filter": where,
            "distances_top3": dists[:3],
            "top_sources": [(m.get("source"), m.get("page")) for m in metas[:3]],
        }

        st.session_state.history.append((user_q, answer, citations, debug))

    # Render history -> prvo novije
    for q, a, c, debug in reversed(st.session_state.history):
        # pitanje korisnika (diskretno)
        st.markdown(f"<p style='color: #888; font-style: italic; margin-bottom: 5px;'>Pitanje: {q}</p>", unsafe_allow_html=True)
        
        # glavni odgovor (istaknut i stručan)
        st.markdown(f"<div style='font-size: 1.1rem; font-weight: 400; line-height: 1.6; color: #E0E0E0;'>{a}</div>", unsafe_allow_html=True)
        
        # izvori (mali, sivi i ukošeni)
        if c:
            st.markdown(
                f"""
                <div style='color: #666; font-size: 0.8rem; font-style: italic; border-top: 0.5px solid #333; padding-top: 10px; margin-top: 15px;'>
                Reference iz materijala:<br>{c}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        if show_debug:
            with st.expander("Debug info", expanded=False):
                st.write(debug)
        st.divider()