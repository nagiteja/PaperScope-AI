from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Dict, List

import streamlit as st

from services.pdf_parser import extract_pages_from_pdf, extract_text_from_pdf
from services.eval_service import run_qa_evaluation, run_summary_evaluation
from services.rag_indexer import index_document
from services.rag_qa import answer_question_with_debug
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
    if "summary_output" not in st.session_state:
        st.session_state["summary_output"] = None
    if "qa_debug" not in st.session_state:
        st.session_state["qa_debug"] = []
    if "eval_report" not in st.session_state:
        st.session_state["eval_report"] = {}


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


def reset_app_state() -> None:
    keys_to_clear = [
        "doc_id",
        "uploaded_file_name",
        "uploaded_file_bytes",
        "file_bytes",
        "whitepaper_text",
        "pages_text",
        "summary_output",
        "indexed",
        "chat_history",
        "qa_debug",
        "eval_report",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


st.set_page_config(page_title="Whitepaper Intelligence - MVP")
# Note: ChromaDB is pinned to v0.3.23 to avoid onnxruntime on Python 3.14.
# Note: google-generativeai is used because google-genai requires pydantic>=2,
# which conflicts with chromadb==0.3.23 (pydantic<2).
st.title("Whitepaper Intelligence - MVP")

_init_session_state()

st.header("File Upload")
reset_col, upload_col = st.columns([1, 4])
with reset_col:
    if st.button("Reset / New Whitepaper", type="secondary"):
        reset_app_state()
        st.success("Reset complete. Upload a new whitepaper.")
        st.rerun()
with upload_col:
    pass
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
        st.session_state["summary_output"] = None
        st.session_state["qa_debug"] = []
        st.session_state["eval_report"] = {}

summary_tab, qa_tab, eval_tab = st.tabs(["Summary", "Q&A", "Evaluation"])

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
                    st.session_state["summary_output"] = summary
                    output_container.text(summary)
            except Exception as exc:
                st.error(f"An error occurred: {exc}")
    elif st.session_state.get("summary_output"):
        output_container.text(st.session_state["summary_output"])

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
                    result = answer_question_with_debug(
                        st.session_state["doc_id"],
                        question,
                        recent_history,
                    )
                except Exception as exc:
                    result = {
                        "answer_text": "Information not found in the document.",
                        "retrieved_chunks": [],
                    }
                    st.error(f"Q&A failed: {exc}")

            response = str(result.get("answer_text", "Information not found in the document."))
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response}
            )
            st.session_state["qa_debug"].append(
                {
                    "question": question,
                    "answer": response,
                    "retrieved_chunks": result.get("retrieved_chunks", []),
                }
            )
            st.chat_message("assistant").write(response)

with eval_tab:
    st.header("Summary Evaluation")
    use_judge = st.toggle("Use Gemini Judge (slower but smarter)", value=True)
    eval_summary_clicked = st.button("Evaluate Summary", type="primary")
    if eval_summary_clicked:
        if not st.session_state["summary_output"]:
            st.info("Generate summary first.")
        elif not st.session_state["file_bytes"]:
            st.info("Upload a PDF first.")
        else:
            try:
                with st.spinner("Evaluating summary..."):
                    whitepaper_text = extract_text_from_pdf(
                        BytesIO(st.session_state["file_bytes"])
                    )
                    report = run_summary_evaluation(
                        st.session_state["summary_output"],
                        whitepaper_text,
                        use_judge,
                    )
                    st.session_state["eval_report"]["summary"] = report
                st.success("Summary evaluation complete.")
            except Exception as exc:
                st.error(f"Summary evaluation failed: {exc}")

    summary_report = st.session_state["eval_report"].get("summary")
    if summary_report:
        st.write({"passed": summary_report["passed"]})
        st.table(
            [{"metric": k, "value": v} for k, v in summary_report["metrics"].items()]
        )
        if summary_report.get("failures"):
            st.error("Failures:")
            st.write(summary_report["failures"])
        if summary_report.get("judge"):
            st.write("Judge:")
            st.json(summary_report["judge"])

    st.header("Q&A Evaluation")
    eval_qa_clicked = st.button("Evaluate Q&A", type="secondary")
    last_n = st.slider("Evaluate last N answers", min_value=1, max_value=10, value=5)
    if eval_qa_clicked:
        if not st.session_state["indexed"]:
            st.info("Build Q&A index first.")
        elif not st.session_state["qa_debug"]:
            st.info("Ask at least one question first.")
        else:
            try:
                with st.spinner("Evaluating Q&A..."):
                    qa_items = st.session_state["qa_debug"][-last_n:]
                    report = run_qa_evaluation(qa_items, use_judge)
                    st.session_state["eval_report"]["qa"] = report
                st.success("Q&A evaluation complete.")
            except Exception as exc:
                st.error(f"Q&A evaluation failed: {exc}")

    qa_report = st.session_state["eval_report"].get("qa")
    if qa_report:
        table_rows = []
        for item in qa_report["items"]:
            table_rows.append(
                {
                    "question": item["question"],
                    "passed": item["passed"],
                    "hallucination_risk": item["metrics"].get(
                        "hallucination_risk_numeric"
                    ),
                }
            )
        st.dataframe(table_rows, width="stretch")

        for idx, item in enumerate(qa_report["items"], start=1):
            with st.expander(f"Q&A Item {idx}"):
                st.write({"passed": item["passed"]})
                st.write("Question:", item["question"])
                st.write("Answer:", item["answer"])
                st.write("Metrics:", item["metrics"])
                if item.get("failures"):
                    st.error("Failures:")
                    st.write(item["failures"])
                st.write("Retrieved Chunks:")
                for chunk in item.get("retrieved_chunks", []):
                    st.write(
                        f"Page {chunk.get('page')} | Section: {chunk.get('section')}"
                    )
                    st.write(chunk.get("text", ""))
                if item.get("judge"):
                    st.write("Judge:")
                    st.json(item["judge"])
