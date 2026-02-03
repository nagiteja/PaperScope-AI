from __future__ import annotations

from pathlib import Path

from services.gemini_client import generate_text


PROMPT_PATH = Path("prompts/summary_system_prompt.txt")


def _load_system_prompt() -> str:
    """Load the summary system prompt from disk."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def summarize_whitepaper(whitepaper_text: str) -> str:
    """Apply the summary prompt and return the structured summary."""
    if not whitepaper_text.strip():
        raise ValueError("Whitepaper text is empty.")

    system_prompt = _load_system_prompt()
    user_prompt = (
        f"{system_prompt}\n\n"
        "WHITEPAPER TEXT:\n"
        f"{whitepaper_text}"
    )
    return generate_text(user_prompt)
