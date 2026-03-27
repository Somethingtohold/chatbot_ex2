"""
WEF Report Ingestion Script
============================
Downloads the World Economic Forum Future of Jobs Report 2025, extracts its text,
cleans it, and saves it to knowledge_base/wef_future_of_jobs_2025.txt.

Usage:
    python ingest_wef.py
"""

import io
import re
from pathlib import Path

import requests
from pypdf import PdfReader

URL = "https://reports.weforum.org/docs/WEF_Future_of_Jobs_Report_2025.pdf"
OUTPUT = Path(__file__).parent / "knowledge_base" / "wef_future_of_jobs_2025.txt"
MAX_PAGES = 200


def download_pdf(url: str) -> PdfReader:
    print(f"Downloading PDF from: {url}")
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    print(f"Downloaded {len(response.content) / 1024:.0f} KB")
    return PdfReader(io.BytesIO(response.content))


def extract_text(reader: PdfReader) -> str:
    pages = reader.pages[:MAX_PAGES]
    print(f"Extracting text from {len(pages)} pages...")
    return "\n".join(page.extract_text() or "" for page in pages)


def clean_text(raw: str) -> str:
    skip_patterns = [
        r"^\s*\d{1,3}\s*$",               # standalone page numbers
        r"^World Economic Forum\s*$",      # repeated header
        r"^weforum\.org\s*$",              # URL footer
        r"^©\s*\d{4}",                     # copyright lines
        r"^\s*www\.",                       # website footers
        r"^\s*[A-Z ]{2,60}\s*\|\s*\d+\s*$",  # "TITLE | page" footers
    ]
    skip_re = [re.compile(p, re.IGNORECASE) for p in skip_patterns]

    cleaned = []
    blank_count = 0
    for line in raw.splitlines():
        stripped = line.strip()
        if any(p.match(stripped) for p in skip_re):
            continue
        if stripped == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(stripped)

    text = "\n".join(cleaned)
    return text.strip()


if __name__ == "__main__":
    reader = download_pdf(URL)
    raw = extract_text(reader)
    cleaned = clean_text(raw)
    OUTPUT.write_text(cleaned, encoding="utf-8")
    print(f"Saved {len(cleaned):,} characters to {OUTPUT}")
