from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from config import (
    EXP_HARD_MAX,
    EXP_IDEAL_MAX,
    EXP_IDEAL_MIN,
    EXP_MIN_YEARS,
    FEATURE_WEIGHTS,
    PREFERRED_LOCATIONS,
    JD_CORE_SKILLS,
    JD_PREFERRED_SKILLS,
    LARGE_PRODUCT_COMPANY_KEYWORDS,
)
from jd_parser import ParsedJD
from parser import ParsedCandidate
from utils import clamp, get_logger, keyword_hit_rate, normalize_text

logger = get_logger(__name__)


@dataclass
class FeatureVector:
    candidate_id: str
    semantic_similarity: float = 0.0
    experience_score: float = 0.0
    title_relevance: float = 0.0
    skill_overlap: float = 0.0
    retrieval_relevance: float = 0.0
    ranking_relevance: float = 0.0
    production_ml_relevance: float = 0.0
    evaluation_relevance: float = 0.0
    project_relevance: float = 0.0
    company_relevance: float = 0.0
    career_stability: float = 0.0
    education_relevance: float = 0.0
    location_relevance: float = 0.0

    def to_weighted_score(self, weights: dict[str, float] = FEATURE_WEIGHTS) -> float:
        """Compute weighted sum of all features."""
        total = 0.0
        for feat, w in weights.items():
            total += getattr(self, feat, 0.0) * w
        return clamp(total, 0.0, 1.0)


# ─── Individual feature functions ────────────────────────────────────────────

def _experience_score(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """
    Bell-shaped score around the JD ideal experience range (5-9 years, peak 7).
    Penalises under-experienced and over-experienced equally but gently.
    """
    yoe = candidate.years_of_experience
    if yoe < 1.0:
        return 0.05
    if EXP_IDEAL_MIN <= yoe <= EXP_IDEAL_MAX:
        # Full score in ideal window
        mid = (EXP_IDEAL_MIN + EXP_IDEAL_MAX) / 2.0
        dist = abs(yoe - mid) / ((EXP_IDEAL_MAX - EXP_IDEAL_MIN) / 2.0)
        return clamp(1.0 - 0.15 * dist, 0.85, 1.0)
    if yoe < EXP_IDEAL_MIN:
        # Ramp from 0.3 at EXP_MIN_YEARS to 1.0 at EXP_IDEAL_MIN
        span = EXP_IDEAL_MIN - EXP_MIN_YEARS
        t = (yoe - EXP_MIN_YEARS) / max(span, 1e-6)
        return clamp(0.3 + 0.7 * t, 0.0, 1.0)
    # yoe > EXP_IDEAL_MAX
    # Gentle decay; very long careers may indicate research / management
    excess = yoe - EXP_IDEAL_MAX
    return clamp(1.0 - 0.04 * excess, 0.5, 1.0)


_RELEVANT_TITLE_TOKENS = {
    "machine learning", "ml engineer", "ai engineer", "applied scientist",
    "research engineer", "nlp engineer", "data scientist", "search engineer",
    "ranking engineer", "retrieval", "recommendation", "information retrieval",
    "senior engineer", "software engineer", "backend engineer",
}

_NEGATIVE_TITLE_TOKENS = {
    "marketing", "sales", "hr", "human resources", "accountant",
    "graphic designer", "content writer", "project manager",
    "business analyst", "civil engineer", "mechanical engineer",
    "finance", "operations", "support",
}


def _title_relevance(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """Score based on current and past titles."""
    titles = [candidate.current_title_normalized]
    for w in candidate.career_history:
        titles.append(w.title_normalized)
    combined = " ".join(titles)

    pos = sum(1 for tok in _RELEVANT_TITLE_TOKENS if tok in combined)
    neg = sum(1 for tok in _NEGATIVE_TITLE_TOKENS if tok in combined)

    base = clamp(pos * 0.25, 0.0, 1.0)
    penalty = clamp(neg * 0.2, 0.0, 0.6)
    return clamp(base - penalty, 0.0, 1.0)


def _skill_overlap(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """
    Fraction of JD core skill groups present in candidate's canonical groups.
    Bonus for preferred skills, with proficiency weighting.
    """
    cand_groups = candidate.canonical_skill_groups

    # Core skills (must-have)
    core_hits = sum(1 for g in jd.must_have_skills if g in cand_groups)
    core_score = core_hits / max(len(jd.must_have_skills), 1)

    # Preferred skills (bonus)
    pref_hits = sum(1 for g in jd.preferred_skills if g in cand_groups)
    pref_bonus = min(pref_hits * 0.08, 0.20)

    # Proficiency weighting: advanced/expert counts more
    proficiency_bonus = 0.0
    for s in candidate.skills:
        if s.canonical_group in jd.must_have_skills:
            if s.proficiency in ("advanced", "expert"):
                proficiency_bonus += 0.03
    proficiency_bonus = min(proficiency_bonus, 0.15)

    return clamp(core_score + pref_bonus + proficiency_bonus, 0.0, 1.0)


def _retrieval_relevance(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """Keyword hit rate for retrieval-specific terms in candidate text."""
    rate = keyword_hit_rate(candidate.full_text, jd.retrieval_keywords)
    # Non-linear: first hits matter most
    return clamp(1.0 - math.exp(-4.0 * rate), 0.0, 1.0)


def _ranking_relevance(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """Keyword hit rate for ranking / recommendation terms."""
    rate = keyword_hit_rate(candidate.full_text, jd.ranking_keywords)
    return clamp(1.0 - math.exp(-4.0 * rate), 0.0, 1.0)


def _production_ml_relevance(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """
    Score for evidence of production ML deployment.
    Combines keyword signals with the parsed has_production_experience flag.
    """
    base = 1.0 if candidate.has_production_experience else 0.0
    kw_rate = keyword_hit_rate(candidate.full_text, jd.production_keywords)
    kw_score = clamp(1.0 - math.exp(-3.0 * kw_rate), 0.0, 1.0)

    # If neither flag nor keyword, heavily penalised
    combined = 0.5 * base + 0.5 * kw_score
    if not candidate.has_production_experience and kw_score < 0.2:
        combined *= 0.3
    return clamp(combined, 0.0, 1.0)


def _evaluation_relevance(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """Evaluation framework presence in candidate text."""
    rate = keyword_hit_rate(candidate.full_text, jd.evaluation_keywords)
    return clamp(1.0 - math.exp(-5.0 * rate), 0.0, 1.0)


def _project_relevance(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """
    Score for descriptions that mention building systems (shipped, built,
    deployed, designed, led, architected) in relevant domains.
    """
    build_words = {
        "built", "designed", "architected", "shipped", "deployed",
        "developed", "implemented", "launched", "led", "owned",
    }
    domain_words = set(jd.retrieval_keywords + jd.ranking_keywords + ["search", "ranking", "recommendation"])

    score = 0.0
    for w in candidate.career_history:
        desc = w.description.lower()
        build_hit = any(bw in desc for bw in build_words)
        domain_hit = any(dw in desc for dw in domain_words)
        if build_hit and domain_hit:
            score += 0.25
        elif build_hit or domain_hit:
            score += 0.10
    return clamp(score, 0.0, 1.0)


def _company_relevance(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """
    Reward product-company experience; penalise pure service-company career.
    Large well-known tech companies get a bonus.
    """
    sc_frac = candidate.service_company_fraction

    # Product-company base score
    product_score = clamp(1.0 - sc_frac * 1.5, 0.0, 1.0)

    # Bonus for recognisable product companies
    all_companies = " ".join(
        [candidate.current_company_normalized] +
        [w.company_normalized for w in candidate.career_history]
    )
    big_tech_hit = any(kw in all_companies for kw in LARGE_PRODUCT_COMPANY_KEYWORDS)
    if big_tech_hit:
        product_score = min(product_score + 0.2, 1.0)

    return clamp(product_score, 0.0, 1.0)


def _career_stability(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """
    Penalise title-hoppers (avg tenure < 18 months).
    Reward stability (avg tenure >= 24 months).
    """
    history = candidate.career_history
    if not history:
        return 0.5
    avg_months = candidate.total_career_months / len(history)
    if avg_months >= 36:
        return 1.0
    if avg_months >= 24:
        return 0.85
    if avg_months >= 18:
        return 0.65
    if avg_months >= 12:
        return 0.45
    return 0.25  # very short tenures — concern


def _education_relevance(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """Reward CS / ML / stats degrees and top-tier institutions."""
    if not candidate.education:
        return 0.3  # no formal degree — neutral-ish for engineering roles

    cs_fields = {
        "computer science", "software engineering", "electrical engineering",
        "information technology", "data science", "statistics", "mathematics",
        "machine learning", "artificial intelligence",
    }
    score = 0.4  # baseline for having a degree
    for edu in candidate.education:
        field = edu.field_of_study.lower()
        if any(f in field for f in cs_fields):
            score += 0.2
        tier = edu.tier
        if tier == "tier_1":
            score += 0.2
        elif tier == "tier_2":
            score += 0.1
        degree = edu.degree.lower()
        if any(d in degree for d in ("mtech", "m.tech", "ms", "m.s.", "phd", "m.e.")):
            score += 0.1
    return clamp(score, 0.0, 1.0)


def _location_relevance(candidate: ParsedCandidate, jd: ParsedJD) -> float:
    """
    Score based on location alignment with JD preferred locations.
    JD prefers Pune/Noida but accepts Hyderabad, Mumbai, Delhi NCR.
    Penalises outside-India candidates (case-by-case, lower probability of hire).
    """
    loc = candidate.location_normalized
    country = candidate.country.lower() if candidate.country else ""

    # India-first
    in_india = "india" in country or any(
        city in loc
        for city in ["pune", "noida", "hyderabad", "mumbai", "delhi",
                     "bangalore", "bengaluru", "chennai", "kolkata",
                     "ahmedabad", "gurgaon", "gurugram"]
    )

    if not in_india:
        # JD says case-by-case but signals preference for India
        # Willing-to-relocate gets partial credit
        if candidate.signals and candidate.signals.willing_to_relocate:
            return 0.35
        return 0.20

    # Preferred JD cities
    top_cities = {"pune", "noida"}
    ok_cities = {"hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "ncr"}

    if any(city in loc for city in top_cities):
        return 1.0
    if any(city in loc for city in ok_cities):
        return 0.80
    if candidate.signals and candidate.signals.willing_to_relocate:
        return 0.65
    return 0.50  # India but other city


# ─── Main feature builder ─────────────────────────────────────────────────────

def build_features(
    candidate: ParsedCandidate,
    jd: ParsedJD,
    semantic_score: float,
) -> FeatureVector:
    """
    Compute all features for a single candidate.

    Args:
        candidate: Parsed candidate.
        jd:        Parsed job description.
        semantic_score: Pre-computed cosine similarity (from semantic_matcher).

    Returns:
        FeatureVector with all fields populated.
    """
    return FeatureVector(
        candidate_id=candidate.candidate_id,
        semantic_similarity=float(semantic_score),
        experience_score=_experience_score(candidate, jd),
        title_relevance=_title_relevance(candidate, jd),
        skill_overlap=_skill_overlap(candidate, jd),
        retrieval_relevance=_retrieval_relevance(candidate, jd),
        ranking_relevance=_ranking_relevance(candidate, jd),
        production_ml_relevance=_production_ml_relevance(candidate, jd),
        evaluation_relevance=_evaluation_relevance(candidate, jd),
        project_relevance=_project_relevance(candidate, jd),
        company_relevance=_company_relevance(candidate, jd),
        career_stability=_career_stability(candidate, jd),
        education_relevance=_education_relevance(candidate, jd),
        location_relevance=_location_relevance(candidate, jd),
    )


def build_features_batch(
    candidates: list[ParsedCandidate],
    jd: ParsedJD,
    semantic_scores: np.ndarray,
) -> list[FeatureVector]:
    """
    Build feature vectors for all candidates in batch.

    Args:
        candidates:       List of ParsedCandidate.
        jd:               Parsed JD.
        semantic_scores:  1-D array of semantic scores, aligned with candidates.

    Returns:
        List of FeatureVector objects.
    """
    features = []
    for i, cand in enumerate(candidates):
        fv = build_features(cand, jd, float(semantic_scores[i]))
        features.append(fv)
    return features