from __future__ import annotations
print("rank.py started")
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from behavioral_scorer import compute_behavioral_multiplier
from config import (
    CACHE_DIR,
    CANDIDATES_PATH,
    EMBEDDING_CACHE_FILE,
    JD_PATH,
    MAX_TOTAL_PENALTY,
    OUTPUT_CSV,
    OUTPUT_DIR,
    PENALTY_WEIGHTS,
    RANDOM_SEED,
    TOP_K,
)
from feature_engineering import FeatureVector, build_features
from honeypot_detector import compute_honeypot_penalty
from jd_parser import parse_jd
from loader import load_all_candidates, load_job_description
from parser import ParsedCandidate, parse_candidates_batch
from reasoning import generate_reasoning

from utils import clamp, get_logger

logger = get_logger(__name__)

np.random.seed(RANDOM_SEED)


def _compute_penalty(
    candidate: ParsedCandidate,
    honeypot_penalties: dict[str, float],
    fv: FeatureVector,
) -> float:
    """
    Aggregate all penalty signals into a single penalty score in [0, MAX_TOTAL_PENALTY].

    Higher penalty → score multiplied by (1 - penalty).
    """
    hp_total = honeypot_penalties.get("total", 0.0)
    kw_stuff = honeypot_penalties.get("keyword_stuffing", 0.0)

    # Inactivity penalty
    from utils import days_since
    from config import LAST_ACTIVE_STALE_DAYS
    inactive = 0.0
    if candidate.signals:
        d = days_since(candidate.signals.last_active_date)
        if d > LAST_ACTIVE_STALE_DAYS:
            inactive = 0.30
        elif d > 90:
            inactive = 0.15

    # Poor recruiter response
    poor_response = 0.0
    if candidate.signals and candidate.signals.recruiter_response_rate < 0.15:
        poor_response = 0.20

    # Service-company-only penalty (when JD explicitly discourages it)
    service_penalty = 0.0
    if candidate.service_company_fraction > 0.85:
        service_penalty = 0.20
    elif candidate.service_company_fraction > 0.70:
        service_penalty = 0.10

    # Research-only penalty
    research_penalty = 0.25 if candidate.is_research_only else 0.0

    # No production deployment evidence
    no_prod = 0.0
    if not candidate.has_production_experience and fv.production_ml_relevance < 0.25:
        no_prod = 0.15

    # Weighted aggregate
    raw = (
        PENALTY_WEIGHTS.get("impossible_history", 0.30) * hp_total
        + PENALTY_WEIGHTS.get("honeypot_score", 0.25) * hp_total
        + PENALTY_WEIGHTS.get("keyword_stuffing", 0.15) * kw_stuff
        + PENALTY_WEIGHTS.get("inactive_profile", 0.10) * inactive
        + PENALTY_WEIGHTS.get("poor_recruiter_response", 0.10) * poor_response
        + PENALTY_WEIGHTS.get("service_company_only", 0.05) * service_penalty
        + PENALTY_WEIGHTS.get("research_only", 0.05) * research_penalty
        + 0.10 * no_prod
    )
    return clamp(raw, 0.0, MAX_TOTAL_PENALTY)


def rank_candidates(
    candidates_path: Path,
    jd_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Run the full ranking pipeline.

    Returns:
        DataFrame with columns [candidate_id, rank, score, reasoning].
    """
    wall_start = time.time()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    logger.info("=== Step 1: Loading candidates ===")
    raws = load_all_candidates(candidates_path)
    logger.info("Loaded %d raw candidates", len(raws))

    # ── 2. Parse ──────────────────────────────────────────────────────────────
    logger.info("=== Step 2: Parsing candidates ===")
    candidates: list[ParsedCandidate] = parse_candidates_batch(raws)
    del raws  # free memory
    logger.info("Parsed %d candidates", len(candidates))

    # ── 3. Parse JD ──────────────────────────────────────────────────────────
    logger.info("=== Step 3: Parsing JD ===")
    jd_text = load_job_description(jd_path)
    jd = parse_jd(jd_text)

    logger.info("=== Step 4: Skipping embeddings ===")

# Give everyone a neutral semantic score.
# Feature engineering will still rank candidates using
# experience, skills, projects, behavioral signals, etc.
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [jd.full_summary] + [c.full_text for c in candidates]
    vec = TfidfVectorizer(stop_words="english", max_features=10000)
    X = vec.fit_transform(texts)
    jd_vec = X[0]
    cand_vecs = X[1:]
    semantic_scores = (cand_vecs @ jd_vec.T).toarray().flatten()

    elapsed = time.time() - wall_start
    logger.info("Embedding done in %.1f s", elapsed)

    # ── 5. Feature engineering + behavioral + penalties ──────────────────────
    logger.info("=== Step 5: Feature engineering ===")

    scored: list[tuple[str, float, FeatureVector, ParsedCandidate]] = []

    for i, candidate in enumerate(tqdm(candidates, desc="Scoring", unit=" cands")):
        sem_score = float(semantic_scores[i])

        # Feature vector
        fv = build_features(candidate, jd, sem_score)
        base_score = fv.to_weighted_score()

        # Behavioral multiplier
        beh_mult = compute_behavioral_multiplier(candidate)
        score_after_beh = base_score * beh_mult

        # Honeypot / penalty
        hp = compute_honeypot_penalty(candidate)
        penalty = _compute_penalty(candidate, hp, fv)
        final_score = score_after_beh * (1.0 - penalty)
        final_score = clamp(final_score, 0.0, 1.0)

        scored.append((candidate.candidate_id, final_score, fv, candidate))

    # ── 6. Sort and select top-K ──────────────────────────────────────────────
    logger.info("=== Step 6: Sorting ===")
    # Stable sort: primary = final_score desc, secondary = candidate_id asc
    scored.sort(key=lambda x: (-x[1], x[0]))
    top = scored[:TOP_K]

    # ── 7. Reasoning ─────────────────────────────────────────────────────────
    logger.info("=== Step 7: Generating reasoning ===")

    # ------------------------------------------------------------------
    # Convert raw ranking scores into recruiter-friendly confidence scores.
    # This does NOT affect ranking order—only the displayed score.
    # Top candidate ≈ 0.99
    # Lowest Top-100 candidate ≈ 0.80
    # ------------------------------------------------------------------

    raw_scores = np.array([score for _, score, _, _ in top], dtype=np.float32)

    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if max_score > min_score:
        normalized = (raw_scores - min_score) / (max_score - min_score)

        # Square-root stretch:
        # spreads the strongest candidates out while keeping everyone
        # in the intuitive 0.80–0.99 confidence range.
        display_scores = 0.80 + 0.19 * np.sqrt(normalized)
    else:
        display_scores = np.full_like(raw_scores, 0.90)

    rows = []

    for rank_idx, ((cid, score, fv, cand), display_score) in enumerate(
        zip(top, display_scores),
        start=1,
    ):
        reasoning = generate_reasoning(cand, fv, rank_idx, score)

        rows.append(
            {
                "candidate_id": cid,
                "rank": rank_idx,
                "score": round(float(display_score), 6),
                "reasoning": reasoning,
            }
        )

    # ── 8. Validate and export ────────────────────────────────────────────────
    logger.info("=== Step 8: Exporting ===")
    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])

    # Validation assertions
    assert len(df) == TOP_K, f"Expected {TOP_K} rows, got {len(df)}"
    assert df["rank"].tolist() == list(range(1, TOP_K + 1)), "Ranks must be 1..100"
    assert df["candidate_id"].nunique() == TOP_K, "Duplicate candidate_ids detected"
    # Scores must be non-increasing
    scores_list = df["score"].tolist()
    for j in range(len(scores_list) - 1):
        assert scores_list[j] >= scores_list[j + 1] - 1e-9, (
            f"Score not non-increasing at rank {j+1}: {scores_list[j]} < {scores_list[j+1]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Submission written to %s", output_path)

    elapsed = time.time() - wall_start
    logger.info("=== Total wall-clock time: %.1f s ===", elapsed)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank 100K candidates for Redrob Intelligent Candidate Discovery Challenge."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=CANDIDATES_PATH,
        help="Path to candidates.jsonl.gz (default: ./candidates.jsonl.gz)",
    )
    parser.add_argument(
        "--jd",
        type=Path,
        default=JD_PATH,
        help="Path to job_description.md (default: ./job_description.md)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_CSV,
        help="Path for output submission.csv (default: ./output/submission.csv)",
    )
    args = parser.parse_args()
    rank_candidates(args.candidates, args.jd, args.out)


if __name__ == "__main__":
    main()