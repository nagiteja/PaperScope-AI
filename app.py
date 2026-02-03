from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Dict, List

import streamlit as st

from services.pdf_parser import extract_pages_from_pdf, extract_text_from_pdf
from services.rag_indexer import index_document
from services.rag_qa import answer_question
from services.summarizer import summarize_whitepaper


def _init_session_state() -> None:
    if "doc_id" not in st.session_state:
        st.session_state["doc_id"] = None
    if "file_bytes" not in st.session_state:
        st.session_state["file_bytes"] = None
    if "indexed" not in st.session_state:
        st.session_state["indexed"] = False
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def _compute_doc_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def _get_recent_history(
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    recent: List[Dict[str, str]] = []
    counts = {"user": 0, "assistant": 0}
    for msg in reversed(history):
        role = msg.get("role")
        if role in counts and counts[role] < 2:
            recent.append(msg)
            counts[role] += 1
        if counts["user"] == 2 and counts["assistant"] == 2:
            break
    return list(reversed(recent))


st.set_page_config(page_title="Whitepaper Intelligence - MVP")
# Note: ChromaDB is pinned to v0.3.23 to avoid onnxruntime on Python 3.14.
# Note: google-generativeai is used because google-genai requires pydantic>=2,
# which conflicts with chromadb==0.3.23 (pydantic<2).
st.title("Whitepaper Intelligence - MVP")

_init_session_state()

st.header("File Upload")
uploaded_file = st.file_uploader(
    "Upload a whitepaper PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    doc_id = _compute_doc_id(file_bytes)
    if doc_id != st.session_state["doc_id"]:
        st.session_state["doc_id"] = doc_id
        st.session_state["file_bytes"] = file_bytes
        st.session_state["indexed"] = False
        st.session_state["chat_history"] = []

summary_tab, qa_tab = st.tabs(["Summary", "Q&A"])

with summary_tab:
    st.header("Generate Summary")
    generate_clicked = st.button("Generate Summary", type="primary")

    st.header("Summary Output Display")
    output_container = st.empty()

    if generate_clicked:
        if not st.session_state["file_bytes"]:
            st.error("Please upload a PDF file before generating a summary.")
        else:
            try:
                with st.spinner("Extracting text from PDF..."):
                    whitepaper_text = extract_text_from_pdf(
                        BytesIO(st.session_state["file_bytes"])
                    )
                if not whitepaper_text:
                    st.error("No text could be extracted from the PDF.")
                else:
                    with st.spinner("Generating summary with Gemini..."):
                        summary = summarize_whitepaper(whitepaper_text)
                    output_container.text(summary)
            except Exception as exc:
                st.error(f"An error occurred: {exc}")

with qa_tab:
    st.header("Build Q&A Index")
    build_clicked = st.button(
        "Build Q&A Index",
        type="secondary",
        disabled=not bool(st.session_state["file_bytes"]),
    )
    if build_clicked:
        try:
            with st.spinner("Indexing document for Q&A..."):
                pages = extract_pages_from_pdf(
                    BytesIO(st.session_state["file_bytes"])
                )
                index_document(st.session_state["doc_id"], pages)
                st.session_state["indexed"] = True
            st.success("Q&A index built successfully.")
        except Exception as exc:
            st.error(f"Failed to build index: {exc}")

    st.header("Q&A Chat")
    if not st.session_state["indexed"]:
        st.info("Build the Q&A index before asking questions.")

    for msg in st.session_state["chat_history"]:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask a question about the whitepaper")
    if question:
        if not st.session_state["indexed"]:
            st.error("Please build the Q&A index first.")
        else:
            st.session_state["chat_history"].append(
                {"role": "user", "content": question}
            )
            recent_history = _get_recent_history(st.session_state["chat_history"])
            with st.spinner("Searching the document..."):
                try:
                    response = answer_question(
                        st.session_state["doc_id"],
                        question,
                        recent_history,
                    )
                except Exception as exc:
                    response = f"Information not found in the document."
                    st.error(f"Q&A failed: {exc}")

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response}
            )
            st.chat_message("assistant").write(response)
