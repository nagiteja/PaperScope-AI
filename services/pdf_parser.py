from __future__ import annotations

from typing import BinaryIO

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
