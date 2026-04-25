"""
src/utils.py — Shared Utility Functions
RAG-Based Customer Support Assistant
"""

import os
import re
import logging
import logging.handlers
from datetime import datetime

# ─── Log Directory ───────────────────────────────────────────────────────────
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


# ─── 1. Logger Setup ──────────────────────────────────────────────────────────
def setup_logger(name: str = "rag_assistant", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a named logger that writes to both:
      - Console (StreamHandler)
      - Rotating file: logs/app.log — max 2 MB, keeps 3 backups
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "app.log")

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Formatter ──────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    # ── Rotating File Handler ──────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=2 * 1024 * 1024,   # 2 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ─── 2. Text Cleaner ──────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Sanitise raw text extracted from PDFs before embedding or display.
    Steps:
      1. Strip leading/trailing whitespace
      2. Collapse multiple blank lines → single newline
      3. Remove non-printable / control characters
      4. Replace consecutive spaces → single space
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove non-printable control characters (except newline and tab)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ─── 3. Timestamp Helper ──────────────────────────────────────────────────────
def timestamp(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Return the current local datetime as a formatted string.

    Parameters
    ----------
    fmt : strftime format string (default: '2024-10-25 14:30:00')

    Returns
    -------
    str — formatted timestamp
    """
    return datetime.now().strftime(fmt)


def timestamp_iso() -> str:
    """Return timestamp in ISO-8601 format (suitable for JSON payloads)."""
    return datetime.now().isoformat()


# ─── 4. Keyword Masking (Privacy Helper) ─────────────────────────────────────
def mask_sensitive(text: str) -> str:
    """
    Mask common sensitive patterns in text before logging.
    Patterns masked:
      - Email addresses
      - Phone numbers (10-digit Indian format)
      - Credit/debit card numbers
    """
    # Mask email addresses
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]", text)

    # Mask 10-digit phone numbers
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)

    # Mask 16-digit card numbers (with or without spaces/dashes)
    text = re.sub(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "[CARD]", text)

    return text


# ─── 5. Truncate Text ─────────────────────────────────────────────────────────
def truncate(text: str, max_chars: int = 300) -> str:
    """
    Truncate text to max_chars and append ellipsis if needed.
    Used for log preview of long strings.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


# ─── Module-Level Default Logger ──────────────────────────────────────────────
# Automatically available when other modules import utils
log = setup_logger()
