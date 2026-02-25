import json
import os
from typing import Any, Dict, List

import streamlit as st
import ollama

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


def context_limit_for_topk(top_k: int) -> int:
    # 2 -> 2500 znakova, 10 -> 9000 znakova
    return int(2500 + (top_k - 2) * (6500 / 8))


def num_predict_for_topk(top_k: int) -> int:
    # malo podignuto da izbjegnemo rezanje sekcija
    # 2 -> 240 tokena, 10 -> 680 tokena
    return int(240 + (top_k - 2) * (440 / 8))


def verbosity_for_topk(top_k: int) -> str:
    if top_k <= 3:
        return "short"
    if top_k >= 8:
        return "long"
    return "medium"


def call_llm(model_name: str, prompt: str, temperature: float, num_predict: int) -> str:
    resp = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
    )
    return resp["response"].strip()


def looks_truncated(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    # ako završava uredno, vjerojatno nije cut
    if t.endswith((".", "!", "?", "…")):
        return False

    # tipični cut slučajevi (posebno kad krene nova sekcija)
    bad_endings = (
        "**PRIM", "**PRIMJ", "**PRIMJE", "**PRIMJER",
        "**OBJA", "**OBJAŠ", "**OBJAŠNJ", "**OBJAŠNJEN",
        "PRIM", "PRIMJ", "PRIMJE", "PRIMJER",
        "OBJA", "OBJAŠ", "OBJAŠNJ", "OBJAŠNJEN",
        ":", "-", "(",
    )
    if any(t.endswith(x) for x in bad_endings):
        return True

    # ako završava slovom/brojem bez interpunkcije, moguće je odrezano
    return t[-1].isalnum()


def continue_answer(model_name: str, prev_answer: str, temperature: float, num_predict: int) -> str:
    # kratki nastavak bez ponavljanja cijelog odgovora
    cont_prompt = f"""
Nastavi točno tamo gdje si stao. Nemoj ponavljati već napisano.
Završi odgovor potpunom rečenicom. Ako si krenuo sekciju PRIMJER, dovrši je u 1–2 rečenice.

TEKST DO SADA:
{prev_answer}

NASTAVAK:
"""
    resp = ollama.generate(
        model=model_name,
        prompt=cont_prompt,
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
    )
    return resp["response"].strip()


# UI SETUP
st.set_page_config(page_title="FIDIT AI asistent", page_icon="🎓", layout="wide")
st.title(" FIDIT – AI asistent za podršku učenju")

with st.expander("ℹ Važne informacije (transparentnost)", expanded=False):
    st.markdown("""
    **Komuniciraš s AI sustavom.** Odgovori su namijenjeni isključivo kao pomoć u učenju.
    **Izvori:** Sustav koristi službene materijale kolegija i prikazuje reference ispod svakog odgovora.
    """)

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader(" Postavke")
    model_name = st.text_input("Ollama model", value=DEFAULT_LLM_MODEL)
    top_k = st.slider("Broj dohvaćenih odlomaka (top_k)", 2, 10, DEFAULT_TOP_K)
    show_debug = st.checkbox("Prikaži debug info", value=False)

    if st.button(" Provjeri bazu"):
        try:
            count = get_collection().count()
            st.success(f"Baza sadrži {count} fragmenata dokumenata.")
        except Exception as e:
            st.error(f"Greška pri pristupu bazi: {e}")

    st.divider()
    st.subheader(" Test pitanja")
    questions = load_questions()
    q_labels = ["(odaberi)"] + [f"{q['id']} – {q['question']}" for q in questions]
    selected = st.selectbox("Brzi odabir:", q_labels, index=0)

with col1:
    st.subheader(" Chat")

    if "history" not in st.session_state:
        st.session_state.history = []

    if "retrieval_cache" not in st.session_state:
        st.session_state.retrieval_cache = {}

    prefill = ""
    selected_q = None
    if selected and selected != "(odaberi)":
        qid = selected.split(" – ", 1)[0]
        selected_q = next((q for q in questions if q["id"] == qid), None)
        if selected_q:
            prefill = selected_q["question"]

    user_q = st.text_input("Postavi pitanje:", value=prefill, placeholder="Npr. Što je SWOT analiza?")

    if st.button("Pošalji") and user_q.strip():
        redacted_q, redaction_report = redact_personal_data(user_q)

        embedder = get_embedder()
        collection = get_collection()
        rules = get_routing_rules()
        routed_sources = route_sources(redacted_q, rules)

        norm_q = normalize_query(redacted_q)
        retrieval_key = f"retr_{norm_q}_{top_k}_{routed_sources}"

        if retrieval_key in st.session_state.retrieval_cache:
            docs, metas, dists, where = st.session_state.retrieval_cache[retrieval_key]
        else:
            with st.spinner("Pretražujem bazu znanja..."):
                docs, metas, dists, where = retrieve_context(
                    collection, embedder, redacted_q, top_k=top_k, routed_sources=routed_sources
                )
                st.session_state.retrieval_cache[retrieval_key] = (docs, metas, dists, where)

        need_clarify, reason = should_clarify(docs, dists)

        if need_clarify:
            answer = f" {reason}\n\n{clarifying_questions()}"
            citations = ""
        else:
            max_ctx = context_limit_for_topk(top_k)
            max_out = num_predict_for_topk(top_k)
            verbosity = verbosity_for_topk(top_k)

            prompt = build_prompt(
                redacted_q, docs, metas,
                max_context_chars=max_ctx,
                verbosity=verbosity
            )

            cache_key = None
            cache_hit = None
            if selected_q:
                cache_key = stable_key(
                    selected_q["id"],
                    norm_q,
                    model_name,
                    str(top_k),
                    str(FIXED_TEMPERATURE),
                    str(max_ctx),
                    str(max_out),
                    str(verbosity),
                )
                cache_hit = get_cached_answer(cache_key)

            if cache_hit:
                answer = cache_hit["answer"]
                citations = cache_hit.get("citations", format_citations(metas))
            else:
                with st.spinner("Generiram odgovor..."):
                    try:
                        answer = call_llm(
                            model_name,
                            prompt,
                            temperature=FIXED_TEMPERATURE,
                            num_predict=max_out,
                        )

                        # Ako je ipak odrezano, dopuni jednim kratkim nastavkom
                        if looks_truncated(answer):
                            cont = continue_answer(
                                model_name,
                                answer,
                                temperature=FIXED_TEMPERATURE,
                                num_predict=min(180, max(80, int(max_out * 0.35))),
                            )
                            if cont:
                                if not answer.endswith("\n"):
                                    answer += "\n"
                                answer += cont

                        citations = format_citations(metas)
                        if cache_key:
                            set_cached_answer(cache_key, {"answer": answer, "citations": citations})

                    except Exception as e:
                        answer = f" Greška pri pozivu modela: {e}"
                        citations = ""

        debug_data = {
            "top_k": top_k,
            "max_context_chars": context_limit_for_topk(top_k),
            "num_predict": num_predict_for_topk(top_k),
            "verbosity": verbosity_for_topk(top_k),
            "distances": dists[:3] if dists else [],
            "sources": [(m.get("source"), m.get("page")) for m in metas[:3]] if metas else [],
        }

        st.session_state.history.append((user_q, answer, citations, debug_data))

    for q, a, c, d in reversed(st.session_state.history):
        st.markdown(
            f"<p style='color: #888; font-style: italic; margin-bottom: 5px;'>Pitanje: {q}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-bottom: 10px;'>{a}</div>",
            unsafe_allow_html=True,
        )

        if c:
            st.markdown(
                f"<div style='color: #555; font-size: 0.85rem; margin-bottom: 20px;'><b>Izvori:</b><br>{c}</div>",
                unsafe_allow_html=True,
            )

        if show_debug:
            with st.expander("Debug detalji"):
                st.json(d)

        st.divider()