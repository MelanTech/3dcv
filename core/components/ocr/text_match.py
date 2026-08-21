"""Text normalization helpers for OCR template matching."""
from __future__ import annotations


def normalize_ocr_match_text(text: object) -> str:
    """Normalize OCR text/templates before fuzzy matching."""
    return "".join(str(text).split()).casefold()
