from __future__ import annotations

from typing import Any, Dict, List

from eval.runner import evaluate_qa, evaluate_summary


def run_summary_evaluation(
    summary_text: str,
    whitepaper_text: str,
    use_judge: bool,
) -> Dict[str, Any]:
    result = evaluate_summary(summary_text, whitepaper_text, use_judge)
    return {
        "metrics": result.metrics,
        "passed": result.passed,
        "failures": result.failures,
        "judge": result.judge,
    }


def run_qa_evaluation(
    qa_items: List[Dict[str, Any]],
    use_judge: bool,
) -> Dict[str, Any]:
    result = evaluate_qa(qa_items, use_judge)
    return {
        "items": [
            {
                "question": item.question,
                "answer": item.answer,
                "metrics": item.metrics,
                "passed": item.passed,
                "failures": item.failures,
                "retrieved_chunks": item.retrieved_chunks,
                "judge": item.judge,
            }
            for item in result.items
        ]
    }
