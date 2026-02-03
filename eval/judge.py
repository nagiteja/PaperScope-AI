from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from services.gemini_client import generate_text


SUMMARY_JUDGE_PROMPT = """
You are a strict evaluator. Use only the provided whitepaper sample text.
Evaluate the summary for faithfulness and coverage.
Return JSON only, with fields:
{"faithfulness": 0-5, "coverage": 0-5, "notes": "...", "major_issues": ["..."]}
If unsupported claims exist, lower faithfulness.
""".strip()


QA_JUDGE_PROMPT = """
You are a strict evaluator. Use only the provided retrieved chunks.
Evaluate answer groundedness, whether it answers the question, and citation quality.
Return JSON only, with fields:
{"grounded": 0-5, "answers_question": 0-5, "citation_quality": 0-5,
 "notes": "...", "hallucination_flags": ["..."]}
If claims are not supported by chunks, grounded must be low.
If "Information not found in the document." is correct, grounded can be high.
""".strip()


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:  # noqa: BLE001
            return None
    return None


def judge_summary(whitepaper_sample: str, summary: str) -> Optional[Dict[str, Any]]:
    prompt = (
        f"{SUMMARY_JUDGE_PROMPT}\n\n"
        "WHITEPAPER SAMPLE:\n"
        f"{whitepaper_sample}\n\n"
        "SUMMARY:\n"
        f"{summary}\n"
    )
    response = generate_text(prompt)
    return _parse_json(response)


def judge_qa(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    chunk_lines = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        chunk_lines.append(
            f"[Chunk {idx}] Page {chunk.get('page')} | Section: {chunk.get('section')}\n"
            f"{chunk.get('text')}"
        )
    prompt = (
        f"{QA_JUDGE_PROMPT}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:\n"
        f"{answer}\n\n"
        "RETRIEVED CHUNKS:\n"
        f"{'\n\n'.join(chunk_lines)}\n"
    )
    response = generate_text(prompt)
    return _parse_json(response)
