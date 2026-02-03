from __future__ import annotations

import re
from typing import Dict, List, Tuple


REQUIRED_HEADINGS = [
    "EXECUTIVE SUMMARY",
    "KEY PROJECT GOAL",
    "CORE TECHNOLOGY / MECHANISM",
    "TOKEN ROLE / UTILITY",
    "TOKENOMICS HIGHLIGHTS",
    "SECURITY / TRUST SIGNALS",
    "RISKS OR UNCERTAINTIES",
]




def check_summary_required_sections(summary: str) -> Tuple[bool, str]:
    missing = [
        heading for heading in REQUIRED_HEADINGS if heading.lower() not in summary.lower()
    ]
    if missing:
        return False, f"Missing headings: {', '.join(missing)}"
    return True, "All required headings present"


def check_summary_word_limit(summary: str, max_words: int = 2000) -> Tuple[bool, str]:
    word_count = len(summary.split())
    if word_count > max_words:
        return False, f"Word count {word_count} exceeds {max_words}"
    return True, f"Word count {word_count} within limit"



def check_summary_missing_info_phrase(summary: str) -> Tuple[bool, str]:
    exact_phrase = "Information not found in the document."
    lowered = summary.lower()
    if exact_phrase.lower() in lowered:
        return True, "Uses required missing-info phrase"
    bad_phrases = [
        "not specified",
        "not mentioned",
        "not provided",
        "not described",
        "not stated",
    ]
    if any(p in lowered for p in bad_phrases):
        return False, "Missing-info phrasing not using required phrase"
    return True, "No missing-info phrasing issues found"


def parse_references(answer: str) -> List[Tuple[str, str]]:
    refs = []
    for line in answer.splitlines():
        match = re.search(r"Page\s*([0-9]+)\s*\|\s*Section:\s*(.+)", line)
        if match:
            refs.append((match.group(1).strip(), match.group(2).strip()))
    return refs


def extract_section(answer: str, start_label: str, end_label: str) -> str:
    pattern = re.compile(
        rf"{start_label}\s*(.*?){end_label}", flags=re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(answer)
    if match:
        return match.group(1).strip()
    return ""


def check_qa_structure(answer: str) -> Tuple[bool, str]:
    required = ["ANSWER", "EVIDENCE", "REFERENCES"]
    missing = [r for r in required if r.lower() not in answer.lower()]
    if missing:
        return False, f"Missing sections: {', '.join(missing)}"
    return True, "Structure valid"


def check_qa_reference_format(answer: str) -> Tuple[bool, str]:
    refs = parse_references(answer)
    if not refs:
        return False, "No Page/Section references found"
    return True, "References include Page and Section"


def check_not_found_format(answer: str) -> Tuple[bool, str]:
    if answer.strip() == "Information not found in the document.":
        return True, "Exact not-found response used"
    return True, "Not-found response not used"


def check_reference_validity(
    answer: str,
    retrieved_chunks: List[Dict[str, object]],
) -> Tuple[bool, str]:
    if answer.strip() == "Information not found in the document.":
        return True, "Not-found response; skipping reference validation"

    refs = parse_references(answer)
    if not refs:
        return False, "No references to validate"

    available = {
        (str(chunk.get("page")), str(chunk.get("section")))
        for chunk in retrieved_chunks
    }
    missing = [ref for ref in refs if (ref[0], ref[1]) not in available]
    if missing:
        return False, f"References not found in retrieved chunks: {missing}"
    return True, "All references match retrieved chunks"


def check_numeric_hallucination(
    answer: str,
    retrieved_chunks: List[Dict[str, object]],
) -> Tuple[bool, str]:
    if answer.strip() == "Information not found in the document.":
        return True, "Not-found response; skipping numeric check"

    answer_text = extract_section(answer, "ANSWER", "EVIDENCE") or answer
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", answer_text)
    if not numbers:
        return True, "No numeric claims detected"

    combined = "\n".join(chunk.get("text", "") for chunk in retrieved_chunks).lower()
    missing = [num for num in numbers if num.lower() not in combined]
    if missing:
        return False, f"Numeric claims not found in context: {missing}"
    return True, "All numeric claims found in context"
