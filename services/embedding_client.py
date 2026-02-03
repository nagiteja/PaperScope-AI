from __future__ import annotations

import os
import time
from typing import List

from dotenv import load_dotenv
import google.generativeai as genai


def _get_api_key() -> str:
    """Load the Gemini API key from environment."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key missing. Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY."
        )
    return api_key


def embed_text(
    text: str,
    *,
    task_type: str = "retrieval_document",
    retries: int = 3,
    delay_seconds: float = 1.0,
) -> List[float]:
    """Create an embedding for the given text using Gemini embeddings API."""
    if not text.strip():
        raise ValueError("Text for embedding is empty.")

    api_key = _get_api_key()
    genai.configure(api_key=api_key)

    model = "models/gemini-embedding-001"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = genai.embed_content(
                model=model,
                content=text,
                task_type=task_type,
            )
            embedding = response.get("embedding") if isinstance(response, dict) else None
            if embedding:
                return embedding
            if hasattr(response, "embedding") and response.embedding:
                return response.embedding
            raise RuntimeError("Empty embedding response.")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(delay_seconds)
            else:
                break

    raise RuntimeError(f"Embedding failed after {retries} attempts: {last_error}")


def sanity_test() -> bool:
    """Run a tiny embedding call to validate configuration."""
    try:
        vec = embed_text("OK")
        return bool(vec)
    except Exception:  # noqa: BLE001
        return False
