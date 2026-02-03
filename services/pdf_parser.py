from __future__ import annotations

from typing import BinaryIO, Dict, List

import fitz  # PyMuPDF


def extract_text_from_pdf(file: BinaryIO) -> str:
    """Extract and clean text from all pages of a PDF file."""
    pdf_bytes = file.read()
    if not pdf_bytes:
        return ""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for page in doc:
        text = page.get_text("text")
        if text:
            pages_text.append(text.strip())

    doc.close()
    return "\n\n".join(pages_text).strip()


def extract_pages_from_pdf(file: BinaryIO) -> List[Dict[str, object]]:
    """Extract text per page with page numbers."""
    pdf_bytes = file.read()
    if not pdf_bytes:
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append(
            {
                "page_number": idx,
                "text": text.strip(),
            }
        )

    doc.close()
    return pages
