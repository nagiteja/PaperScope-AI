from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai


DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _get_api_key() -> str:
    """Load the Gemini API key from environment."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key missing. Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY."
        )
    return api_key


def _get_model(model: str) -> genai.GenerativeModel:
    """Create and return a Gemini model instance."""
    api_key = _get_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model)


def generate_text(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = 0.2,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate text with Gemini for the given prompt."""
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    model_client = _get_model(model)
    response = model_client.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
        },
    )
    if not response or not getattr(response, "text", None):
        raise RuntimeError("Empty response from Gemini.")
    return response.text.strip()


def sanity_test() -> bool:
    """Run a tiny generation call to validate configuration."""
    try:
        output = generate_text("Say OK.")
        return bool(output)
    except Exception:  # noqa: BLE001
        return False
