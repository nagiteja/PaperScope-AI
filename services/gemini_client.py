from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai


def _get_api_key() -> str:
    """Load the Gemini API key from environment."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment.")
    return api_key


def _get_model() -> genai.GenerativeModel:
    """Create and return a Gemini model instance."""
    api_key = _get_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash-lite")


def generate_text(prompt: str, *, temperature: Optional[float] = 0.2) -> str:
    """Generate text with Gemini for the given prompt."""
    model = _get_model()
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
        },
    )
    if not response or not getattr(response, "text", None):
        raise RuntimeError("Empty response from Gemini.")
    return response.text.strip()
