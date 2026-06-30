"""
utils.py — Shared utility functions: text normalization, date math,
cosine similarity, clamping, logging setup.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from datetime import date, datetime
from typing import Any

import numpy as np

from config import LOG_FORMAT, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
    return logging.getLogger(name)


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, value))


def safe_float(val: Any, default: float = 0.0) -> float:
    """Convert val to float safely."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Convert val to int safely."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def parse_date(val: Any) -> date | None:
    """Try to parse a date from various string formats."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    s = str(val).strip()
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y-%m",
        "%Y/%m",
        "%b %Y",
        "%B %Y",
        "%Y",
    ):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def years_between(start: date | None, end: date | None) -> float:
    """Return fractional years between two dates. Returns 0 for invalid inputs."""
    if start is None or end is None:
        return 0.0
    delta = (end - start).days
    return max(0.0, delta / 365.25)


def today() -> date:
    """Return today's date."""
    return date.today()


def cosine_sim_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_sim_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Batch cosine similarity between each query row and every corpus row.

    Args:
        queries: shape (Q, D)
        corpus:  shape (N, D)

    Returns:
        shape (Q, N) — each row is similarities of one query to all corpus docs
    """
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    c_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    return q_norm @ c_norm.T


def keyword_hit_rate(text: str, keywords: list[str]) -> float:
    """Fraction of keywords that appear in text (case-insensitive)."""
    if not keywords or not text:
        return 0.0
    t = text.lower()
    return sum(1 for kw in keywords if kw.lower() in t) / len(keywords)


def count_token_repetitions(tokens: list[str], threshold: int = 4) -> int:
    """Return number of distinct tokens that appear >= threshold times."""
    return sum(1 for cnt in Counter(tokens).values() if cnt >= threshold)


def truncate(text: str, max_chars: int = 512) -> str:
    """Truncate string to max_chars."""
    return text[:max_chars] if len(text) > max_chars else text


def flatten_list(lst: list[Any]) -> list[Any]:
    """Flatten one level of nesting."""
    out: list[Any] = []
    for item in lst:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def days_since(d: date | None) -> int:
    """Days from date d to today. Returns large number if d is None."""
    if d is None:
        return 9999
    return (today() - d).days