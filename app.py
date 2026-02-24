import json
import os
from typing import Any, Dict, List

import streamlit as st
import ollama

from src.rag import (
    extractive_fallback_answer,
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

# Fiksne postavke
DEFAULT_LLM_MODEL = "mistral"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2


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
    except Exception as e:
        answer = "Greška pri generiranju (Ollama). Prikazujem sažetak iz materijala:\n\n"
        answer += extractive_fallback_answer(redacted_q, docs)


def call_llm(model_name: str, prompt: str, temperature: float) -> str:
    resp = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": temperature,
        },
    )
    return resp["response"].strip()


# Streamlit UI
st.set_page_config(page_title="FIDIT AI asistent", layout="wide")
st.title("FIDIT - AI asistent za podršku učenju")

with st.expander("Važne informacije (transparentnost)", expanded=True):
    st.markdown(
        """
**Komuniciraš s AI sustavom.** Odgovori služe kao pomoć u učenju i ne zamjenjuju službenu nastavu.
**Privatnost:** Sustav ne sprema tvoje podatke.
        """
    )

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Postavke")
    model_name = st.text_input("Ollama model", value=DEFAULT_LLM_MODEL)

    st.write("Količina konteksta")
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #888; margin-bottom: -10px;">
            <span>Brzo, ali manje</span>
            <span>Sporo, ali više</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_k = st.slider("", 2, 10, DEFAULT_TOP_K, label_visibility="collapsed")

    show_debug = st.checkbox("Prikaži debug info", value=False)

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
        redacted_q, redaction_report = redact_personal_data(user_q)
        embedder = get_embedder()
        collection = get_collection()
        rules = get_routing_rules()

        routed_sources = route_sources(redacted_q, rules)
        norm_q = normalize_query(redacted_q)

        docs, metas, dists, where = retrieve_context(
            collection, embedder, redacted_q, top_k=top_k, routed_sources=routed_sources
        )

        need_clarify, reason = should_clarify(docs, dists)
        if need_clarify:
            answer = f"Napomena: {reason}\n\n{clarifying_questions()}"
            citations = ""
        else:
            prompt = build_prompt(redacted_q, docs, metas)

            cache_hit = None
            cache_key = None
            if selected_q:
                cache_key = stable_key(
                    selected_q["id"],
                    norm_q,
                    model_name,
                    str(top_k),
                    str(DEFAULT_TEMPERATURE),
                    str(collection.count()),
                )
                cache_hit = get_cached_answer(cache_key)

            if cache_hit:
                answer = cache_hit["answer"]
            else:
                try:
                    answer = call_llm(model_name, prompt, temperature=DEFAULT_TEMPERATURE)
                except Exception as e:
                    answer = f"Greška: {e}"

                if cache_key and selected_q:
                    set_cached_answer(cache_key, {"answer": answer})

            citations = format_citations(metas)

        debug = {"redaction": redaction_report, "distances_top3": dists[:3]}
        st.session_state.history.append((user_q, answer, citations, debug))

    for q, a, c, debug in reversed(st.session_state.history):
        st.markdown(
            f"<p style='color: #aaa; font-style: italic; margin-bottom: 0px;'>Pitanje: {q}</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size: 1.1rem; font-weight: 500; line-height: 1.6;'>{a}</div>",
            unsafe_allow_html=True,
        )

        if c:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style='color: #666; font-size: 0.85rem; font-style: italic; border-top: 0.5px solid #444; padding-top: 10px;'>
                Korišteni izvori:<br>{c}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if show_debug:
            with st.expander("Debug info", expanded=False):
                st.write(debug)

        st.divider()