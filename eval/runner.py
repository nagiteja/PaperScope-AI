from __future__ import annotations

from typing import Any, Dict, List

from eval import judge
from eval.schemas import QAEvalItem, QAEvalReport, SummaryEvalResult
from eval.scorers import (
    check_not_found_format,
    check_numeric_hallucination,
    check_qa_reference_format,
    check_qa_structure,
    check_reference_validity,
    check_summary_missing_info_phrase,
    check_summary_required_sections,
    check_summary_word_limit,
)


def _sample_whitepaper(text: str, chunk_size: int = 12000) -> str:
    if len(text) <= chunk_size * 2:
        return text
    return f"{text[:chunk_size]}\n\n...\n\n{text[-chunk_size:]}"


def evaluate_summary(
    summary_text: str,
    whitepaper_text: str,
    use_judge: bool,
) -> SummaryEvalResult:
    metrics: Dict[str, Any] = {}
    failures: List[str] = []

    checks = [
        ("has_required_sections", check_summary_required_sections(summary_text)),
        ("within_word_limit", check_summary_word_limit(summary_text)),
        ("missing_info_phrase_consistency", check_summary_missing_info_phrase(summary_text)),
    ]

    for name, (passed, message) in checks:
        metrics[name] = message
        if not passed:
            failures.append(f"{name}: {message}")

    judge_result = None
    if use_judge:
        whitepaper_sample = _sample_whitepaper(whitepaper_text)
        judge_result = judge.judge_summary(whitepaper_sample, summary_text)

    return SummaryEvalResult(
        metrics=metrics,
        passed=len(failures) == 0,
        failures=failures,
        judge=judge_result,
    )


def evaluate_qa(
    qa_items: List[Dict[str, Any]],
    use_judge: bool,
) -> QAEvalReport:
    items: List[QAEvalItem] = []
    for item in qa_items:
        question = item.get("question", "")
        answer = item.get("answer", "")
        retrieved_chunks = item.get("retrieved_chunks", [])

        metrics: Dict[str, Any] = {}
        failures: List[str] = []

        checks = [
            ("qa_structure_valid", check_qa_structure(answer)),
            ("citation_presence_format", check_qa_reference_format(answer)),
            ("not_found_correctness", check_not_found_format(answer)),
            ("citation_validity_vs_retrieval", check_reference_validity(answer, retrieved_chunks)),
            ("hallucination_risk_numeric", check_numeric_hallucination(answer, retrieved_chunks)),
        ]

        for name, (passed, message) in checks:
            metrics[name] = message
            if not passed:
                failures.append(f"{name}: {message}")

        judge_result = None
        if use_judge:
            judge_result = judge.judge_qa(question, answer, retrieved_chunks)

        items.append(
            QAEvalItem(
                question=question,
                answer=answer,
                metrics=metrics,
                passed=len(failures) == 0,
                failures=failures,
                retrieved_chunks=retrieved_chunks,
                judge=judge_result,
            )
        )

    return QAEvalReport(items=items)
