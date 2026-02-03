from __future__ import annotations

from pathlib import Path

from services.gemini_client import generate_text


PROMPT_PATH = Path("prompts/summary_system_prompt.txt")


def _load_system_prompt() -> str:
    """Load the summary system prompt from disk."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def contains_investment_language(text: str) -> bool:
    """Check for banned investment or trading language."""
    banned_terms = [
        "buy",
        "sell",
        "hold",
        "bullish",
        "bearish",
        "price target",
        "moon",
        "guaranteed returns",
        "financial advice",
    ]
    text_lower = text.lower()
    return any(term in text_lower for term in banned_terms)


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
    summary = generate_text(user_prompt)

    if contains_investment_language(summary):
        retry_prompt = (
            f"{user_prompt}\n\n"
            "IMPORTANT: The previous output contained banned investment language. "
            "You must regenerate the summary and strictly avoid all investment or "
            "trading terms."
        )
        summary = generate_text(retry_prompt)

    return summary
