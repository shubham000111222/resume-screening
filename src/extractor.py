"""
extractor.py — PDF & plain-text resume extraction
"""
from __future__ import annotations

import io
import re
import logging
from pathlib import Path
from typing import Union

import pdfplumber
import docx2txt

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_from_pdf_bytes(content: bytes) -> str:
    """Extract raw text from PDF bytes using pdfplumber."""
    text_parts: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
    return "\n".join(text_parts)


def extract_from_docx_bytes(content: bytes) -> str:
    """Extract raw text from DOCX bytes."""
    try:
        return docx2txt.process(io.BytesIO(content))
    except Exception as exc:
        logger.error("DOCX extraction failed: %s", exc)
        return ""


def extract_from_txt_bytes(content: bytes) -> str:
    """Decode plain-text bytes."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def extract_text(file_name: str, content: bytes) -> str:
    """
    Dispatch extraction by file extension.

    Parameters
    ----------
    file_name : str
        Original filename (used to detect format).
    content   : bytes
        Raw file bytes.

    Returns
    -------
    str
        Extracted plain text.
    """
    ext = Path(file_name).suffix.lower()
    if ext == ".pdf":
        return extract_from_pdf_bytes(content)
    elif ext in (".docx", ".doc"):
        return extract_from_docx_bytes(content)
    elif ext in (".txt", ".text"):
        return extract_from_txt_bytes(content)
    else:
        logger.warning("Unsupported format '%s', attempting plain-text decode.", ext)
        return extract_from_txt_bytes(content)


# ──────────────────────────────────────────────────────────────────────────────
# Batch extraction (for local directories)
# ──────────────────────────────────────────────────────────────────────────────

def extract_directory(directory: Union[str, Path]) -> dict[str, str]:
    """
    Extract text from all supported files in a directory.

    Returns
    -------
    dict[str, str]
        Mapping of filename → extracted text.
    """
    directory = Path(directory)
    results: dict[str, str] = {}
    for fpath in directory.iterdir():
        if fpath.suffix.lower() in (".pdf", ".docx", ".doc", ".txt"):
            try:
                results[fpath.name] = extract_text(fpath.name, fpath.read_bytes())
            except Exception as exc:
                logger.error("Failed to extract '%s': %s", fpath.name, exc)
    return results
