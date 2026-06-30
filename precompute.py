from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import CACHE_DIR, CANDIDATES_PATH, EMBEDDING_CACHE_FILE, JD_PATH
from jd_parser import parse_jd
from loader import load_all_candidates, load_job_description
from parser import parse_candidates_batch
from semantic_matcher import compute_candidate_embeddings, embed_jd
from utils import get_logger

logger = get_logger(__name__)


def run_precompute(
    candidates_path: Path,
    force_recompute: bool = False,
) -> None:
    """
    Precompute and cache candidate embeddings.

    Args:
        candidates_path: Path to candidates.jsonl.gz.
        force_recompute: If True, ignore existing cache.
    """
    t0 = time.time()

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    logger.info("Loading candidates from %s", candidates_path)
    raws = load_all_candidates(candidates_path)

    # Parse
    logger.info("Parsing %d candidates", len(raws))
    candidates = parse_candidates_batch(raws)
    del raws  # free memory

    # Extract texts and IDs
    texts = [c.full_text for c in candidates]
    ids = [c.candidate_id for c in candidates]
    del candidates  # free memory

    # Compute embeddings with caching
    logger.info("Computing/caching embeddings for %d candidates", len(texts))
    embs = compute_candidate_embeddings(
        texts=texts,
        candidate_ids=ids,
        cache_path=EMBEDDING_CACHE_FILE,
        force_recompute=force_recompute,
    )

    elapsed = time.time() - t0
    logger.info(
        "Precompute complete — %d embeddings (shape %s) in %.1f s",
        len(ids),
        embs.shape,
        elapsed,
    )
    logger.info("Cache written to %s", EMBEDDING_CACHE_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute candidate embeddings for the Redrob ranking challenge."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=CANDIDATES_PATH,
        help="Path to candidates.jsonl.gz",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cache exists",
    )
    args = parser.parse_args()
    run_precompute(args.candidates, force_recompute=args.force)


if __name__ == "__main__":
    main()