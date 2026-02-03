from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")

import chromadb
from chromadb.config import Settings
import pydantic

from services.embedding_client import embed_text
from services.sectionizer import iter_lines_with_section


CHROMA_DIR = Path("data/chroma")


def _safe_collection_name(doc_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)
    return f"doc_{safe}"


def _get_client() -> chromadb.Client:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_version = getattr(chromadb, "__version__", "unknown")
    print(f"ChromaDB version: {chroma_version}")
    pyd_version = getattr(pydantic, "__version__", "unknown")
    print(f"Pydantic version: {pyd_version}")
    if pyd_version != "unknown":
        major = int(pyd_version.split(".", maxsplit=1)[0])
        if major >= 2:
            raise RuntimeError("Pydantic>=2 detected. Please install pydantic<2.")

    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(CHROMA_DIR),
            anonymized_telemetry=False,
        )
    )


def build_or_load_index(doc_id: str) -> chromadb.Collection:
    """Create or load a persistent Chroma collection for the doc."""
    client = _get_client()
    return client.get_or_create_collection(
        name=_safe_collection_name(doc_id),
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_page_text(
    page_text: str,
    *,
    target_size: int = 1600,
    overlap: int = 200,
) -> List[Dict[str, str]]:
    """Chunk a page's text while tracking current section."""
    lines = page_text.splitlines()
    buffer = ""
    chunks: List[Dict[str, str]] = []
    last_section = "Unknown Section"

    for section, line in iter_lines_with_section(lines):
        if line.strip():
            last_section = section
        line_with_break = line.strip() + "\n"
        buffer += line_with_break

        if len(buffer) >= target_size:
            chunks.append(
                {
                    "text": buffer.strip(),
                    "section": last_section,
                }
            )
            buffer = buffer[-overlap:].lstrip()

    if buffer.strip():
        chunks.append({"text": buffer.strip(), "section": last_section})
    return chunks


def index_document(doc_id: str, pages: List[Dict[str, object]]) -> None:
    """Index a document's pages into Chroma."""
    client = _get_client()
    collection = client.get_or_create_collection(
        name=_safe_collection_name(doc_id),
        metadata={"hnsw:space": "cosine"},
    )
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception:  # noqa: BLE001
        pass

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, object]] = []
    embeddings: List[List[float]] = []

    chunk_id = 0
    for page in pages:
        page_number = int(page.get("page_number", 0))
        page_text = str(page.get("text", "") or "")
        if not page_text.strip():
            continue

        chunks = _chunk_page_text(page_text)
        for chunk in chunks:
            chunk_text = chunk["text"]
            if not chunk_text.strip():
                continue
            section = chunk.get("section", "Unknown Section")
            chunk_id += 1
            chunk_key = f"{doc_id}_p{page_number}_c{chunk_id}"
            ids.append(chunk_key)
            documents.append(chunk_text)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "page": page_number,
                    "section": section,
                    "chunk_id": chunk_id,
                }
            )
            embeddings.append(embed_text(chunk_text, task_type="retrieval_document"))

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        client.persist()