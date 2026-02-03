from __future__ import annotations

from typing import Iterable, Iterator, Tuple


def is_heading(line: str) -> bool:
    """Heuristically determine whether a line is a section heading."""
    cleaned = line.strip()
    if len(cleaned) < 3:
        return False
    if cleaned.startswith("##"):
        return True
    if cleaned.endswith(":"):
        return True
    if len(cleaned) <= 60 and cleaned.isupper():
        return True

    words = [w for w in cleaned.split() if w.isalpha()]
    if 0 < len(words) <= 6:
        title_like = all(w[0].isupper() for w in words)
        if title_like:
            return True
    return False


def iter_lines_with_section(lines: Iterable[str]) -> Iterator[Tuple[str, str]]:
    """Yield (section_name, line) pairs using heading heuristics."""
    current_section = "Unknown Section"
    for line in lines:
        cleaned = line.strip()
        if cleaned and is_heading(cleaned):
            heading = cleaned.lstrip("#").strip().rstrip(":").strip()
            current_section = heading or "Unknown Section"
        yield current_section, line
