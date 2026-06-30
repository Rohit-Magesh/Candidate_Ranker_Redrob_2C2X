"""
loader.py — Efficient streaming loader for gzipped JSONL candidate files.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Generator

import orjson
from tqdm import tqdm

from utils import get_logger

logger = get_logger(__name__)


def iter_candidates(
    path: Path | str,
    show_progress: bool = True,
) -> Generator[dict, None, None]:
    """
    Stream candidates one-by-one from a .jsonl.gz (or .jsonl) file.

    Args:
        path: Path to the .jsonl.gz file.
        show_progress: Show tqdm progress bar.

    Yields:
        One dict per candidate record.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    logger.info("Streaming candidates from %s", path)
    opener = gzip.open if path.suffix == ".gz" else open
    errors = 0

    with opener(path, "rb") as fh:
        bar = tqdm(unit=" candidates", desc="Loading", disable=not show_progress)
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = orjson.loads(line)
                yield record
                bar.update(1)
            except orjson.JSONDecodeError as exc:
                errors += 1
                if errors <= 5:
                    logger.warning("Skipping malformed JSON line: %s", exc)
        bar.close()

    if errors:
        logger.warning("Total malformed lines skipped: %d", errors)


def load_all_candidates(
    path: Path | str,
    show_progress: bool = True,
) -> list[dict]:
    """
    Load all candidates into a list. Suitable for ≤16 GB RAM environments.

    Args:
        path: Path to the .jsonl.gz file.
        show_progress: Show progress bar.

    Returns:
        List of candidate dicts.
    """
    candidates = list(iter_candidates(path, show_progress=show_progress))
    logger.info("Loaded %d candidates total", len(candidates))
    return candidates


def load_job_description(path: Path | str) -> str:
    """
    Load the job description from a plain-text / Markdown file.

    Args:
        path: Path to job_description.md.

    Returns:
        Raw text content.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Job description not found: {path}")
    text = path.read_text(encoding="utf-8")
    logger.info("Loaded job description (%d chars)", len(text))
    return text