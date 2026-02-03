from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from services.embedding_client import embed_text
from services.gemini_client import generate_text
from services.rag_indexer import build_or_load_index


PROMPT_PATH = Path("prompts/mvp2_qa_system_prompt.txt")


def _load_system_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _keyword_set(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) >= 4}


def _has_keyword_overlap(question: str, documents: List[str]) -> bool:
    q_words = _keyword_set(question)
    if not q_words:
        return True
    for doc in documents:
        if q_words & _keyword_set(doc):
            return True
    return False


def _format_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return "None"
    lines = []
    for msg in chat_history:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "").strip()
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"


def _format_context(documents: List[str], metadatas: List[Dict[str, object]]) -> str:
    blocks = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        page = meta.get("page", "Unknown")
        section = meta.get("section", "Unknown Section")
        header = f"[Chunk {idx}] Page {page} | Section: {section}"
        blocks.append(f"{header}\n{doc}")
    return "\n\n".join(blocks) if blocks else "None"


def _response_has_references(text: str) -> bool:
    return "REFERENCES" in text.upper() and "PAGE" in text.upper()


def _retrieve_chunks(doc_id: str, question: str) -> Dict[str, List[object]]:
    collection = build_or_load_index(doc_id)
    query_embedding = embed_text(question, task_type="retrieval_query")
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0] if result else []
    metadatas = result.get("metadatas", [[]])[0] if result else []
    distances = result.get("distances", [[]])[0] if result else []
    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
    }


def answer_question_with_debug(
    doc_id: str,
    question: str,
    chat_history: List[Dict[str, str]],
) -> Dict[str, object]:
    """Answer a question and return debug retrieval context."""
    if not question.strip():
        return {
            "answer_text": "Information not found in the document.",
            "retrieved_chunks": [],
        }

    retrieval = _retrieve_chunks(doc_id, question)
    documents = retrieval["documents"]
    metadatas = retrieval["metadatas"]
    distances = retrieval["distances"]

    if not documents:
        return {
            "answer_text": "Information not found in the document.",
            "retrieved_chunks": [],
        }

    min_distance = min(distances) if distances else None
    documents_short = all(len(doc.strip()) < 200 for doc in documents)
    keyword_overlap = _has_keyword_overlap(question, documents)

    if min_distance is None:
        return {
            "answer_text": "Information not found in the document.",
            "retrieved_chunks": [],
        }
    if documents_short and not keyword_overlap:
        return {
            "answer_text": "Information not found in the document.",
            "retrieved_chunks": [],
        }
    if min_distance > 0.35 and not keyword_overlap:
        return {
            "answer_text": "Information not found in the document.",
            "retrieved_chunks": [],
        }

    system_prompt = _load_system_prompt()
    context_block = _format_context(documents, metadatas)
    history_block = _format_history(chat_history)

    user_prompt = (
        "CONVERSATION MEMORY (LAST 2 TURNS EACH SIDE):\n"
        f"{history_block}\n\n"
        "CONTEXT CHUNKS:\n"
        f"{context_block}\n\n"
        "QUESTION:\n"
        f"{question}\n"
    )

    response = generate_text(f"{system_prompt}\n\n{user_prompt}")
    if not _response_has_references(response):
        response = "Information not found in the document."

    retrieved_chunks = []
    for doc, meta in zip(documents, metadatas):
        retrieved_chunks.append(
            {
                "page": meta.get("page", "Unknown"),
                "section": meta.get("section", "Unknown Section"),
                "text": doc,
            }
        )

    return {
        "answer_text": response,
        "retrieved_chunks": retrieved_chunks,
    }


def answer_question(
    doc_id: str,
    question: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """Answer a question using only retrieved document chunks."""
    result = answer_question_with_debug(doc_id, question, chat_history)
    return str(result.get("answer_text", "Information not found in the document."))
