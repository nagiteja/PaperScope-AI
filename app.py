from __future__ import annotations

import streamlit as st

from services.pdf_parser import extract_text_from_pdf
from services.summarizer import summarize_whitepaper


st.set_page_config(page_title="Whitepaper Intelligence - MVP")
st.title("Whitepaper Intelligence - MVP")

st.header("File Upload")
uploaded_file = st.file_uploader(
    "Upload a whitepaper PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

st.header("Generate Summary")
generate_clicked = st.button("Generate Summary", type="primary")

st.header("Summary Output Display")
output_container = st.empty()

if generate_clicked:
    if not uploaded_file:
        st.error("Please upload a PDF file before generating a summary.")
    else:
        try:
            with st.spinner("Extracting text from PDF..."):
                whitepaper_text = extract_text_from_pdf(uploaded_file)
            if not whitepaper_text:
                st.error("No text could be extracted from the PDF.")
            else:
                with st.spinner("Generating summary with Gemini..."):
                    summary = summarize_whitepaper(whitepaper_text)
                output_container.text(summary)
        except Exception as exc:
            st.error(f"An error occurred: {exc}")
