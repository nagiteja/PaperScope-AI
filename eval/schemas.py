from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SummaryEvalResult:
    metrics: Dict[str, Any]
    passed: bool
    failures: List[str] = field(default_factory=list)
    judge: Optional[Dict[str, Any]] = None


@dataclass
class QAEvalItem:
    question: str
    answer: str
    metrics: Dict[str, Any]
    passed: bool
    failures: List[str] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    judge: Optional[Dict[str, Any]] = None


@dataclass
class QAEvalReport:
    items: List[QAEvalItem]
