from __future__ import annotations

from behavioral_scorer import get_behavioral_sub_scores
from feature_engineering import FeatureVector
from honeypot_detector import compute_honeypot_penalty
from parser import ParsedCandidate
from utils import days_since, get_logger

logger = get_logger(__name__)


def _yoe_phrase(yoe: float) -> str:
    return f"{yoe:g}"


def _company_type_phrase(candidate: ParsedCandidate) -> str:
    sc_frac = candidate.service_company_fraction
    if sc_frac < 0.15:
        return "product company"
    if sc_frac < 0.50:
        return "mixed product/services background"
    return "primarily services background"


def _skill_highlights(candidate: ParsedCandidate, top_n: int = 3) -> list[str]:
    """Return top-N canonical skill groups actually present."""
    from config import JD_CORE_SKILLS, JD_PREFERRED_SKILLS
    priority = JD_CORE_SKILLS + JD_PREFERRED_SKILLS
    present = []
    seen = set()
    for group in priority:
        if group in candidate.canonical_skill_groups and group not in seen:
            present.append(group.replace("_", " "))
            seen.add(group)
        if len(present) >= top_n:
            break
    return present


def _engagement_phrase(candidate: ParsedCandidate) -> str:
    sig = candidate.signals
    if sig is None:
        return ""
    parts = []
    if sig.open_to_work:
        parts.append("actively open to work")
    if sig.recruiter_response_rate >= 0.70:
        parts.append("strong recruiter response rate")
    elif sig.recruiter_response_rate < 0.25:
        parts.append("low recruiter response rate")
    d = days_since(sig.last_active_date)
    if d <= 30:
        parts.append("recently active")
    elif d > 180:
        parts.append("inactive for over 6 months")
    if sig.notice_period_days > 90:
        parts.append(f"{sig.notice_period_days}-day notice period")
    elif sig.notice_period_days <= 30:
        parts.append("short notice period")
    if not parts:
        return ""
    return "; ".join(parts)


def _concern_phrase(
    candidate: ParsedCandidate,
    fv: FeatureVector,
    honeypot: dict[str, float],
) -> str:
    concerns = []
    if honeypot.get("total", 0.0) > 0.30:
        concerns.append("profile has consistency flags")
    if candidate.is_research_only:
        concerns.append("primarily research background without clear production deployment")
    if candidate.service_company_fraction > 0.80:
        concerns.append("career is largely services/consulting")
    if fv.production_ml_relevance < 0.30:
        concerns.append("limited production ML evidence")
    if fv.evaluation_relevance < 0.15:
        concerns.append("no clear evaluation framework experience")
    sig = candidate.signals
    if sig and sig.notice_period_days > 90:
        concerns.append(f"long notice period ({sig.notice_period_days} days)")
    return "; ".join(concerns[:2]) if concerns else ""


def _current_role_phrase(candidate: ParsedCandidate) -> str:
    title = candidate.current_title or "professional"
    yoe = _yoe_phrase(candidate.years_of_experience)
    company_type = _company_type_phrase(candidate)
    return f"{title} with {yoe} years of experience at a {company_type}"


def generate_reasoning(
    candidate: ParsedCandidate,
    fv: FeatureVector,
    rank: int,
    final_score: float,
) -> str:
    """
    Generate a factual 1-2 sentence reasoning string.

    Args:
        candidate: ParsedCandidate.
        fv:        FeatureVector for the candidate.
        rank:      Final rank (1-indexed).
        final_score: Final composite score.

    Returns:
        String with 1-2 sentences.
    """
    honeypot = compute_honeypot_penalty(candidate)
    skills_str = ", ".join(_skill_highlights(candidate, top_n=3))
    engagement = _engagement_phrase(candidate)
    concern = _concern_phrase(candidate, fv, honeypot)

    # ── Sentence 1: technical background ────────────────────────────────────
    role_phrase = _current_role_phrase(candidate)

    if skills_str:
        s1 = f"{role_phrase}, with demonstrated skills in {skills_str}."
    else:
        s1 = f"{role_phrase}."

    # Add production signal if strong
    if candidate.has_production_experience and fv.production_ml_relevance >= 0.5:
        s1 = s1.rstrip(".") + ", and evidence of production ML deployment."

    # ── Sentence 2: engagement / concerns ────────────────────────────────────
    if rank <= 20:
        # Strong positive — lead with match quality, note concerns if any
        if engagement:
            s2 = f"Strong JD alignment on retrieval/ranking profile; {engagement}."
        else:
            s2 = "Strong alignment with the JD's retrieval and ranking requirements."
        if concern:
            s2 = s2.rstrip(".") + f"; note: {concern}."
    elif rank <= 60:
        # Balanced
        if concern and engagement:
            s2 = f"Reasonable fit with {engagement}; concern: {concern}."
        elif concern:
            s2 = f"Reasonable fit for the role; concern: {concern}."
        elif engagement:
            s2 = f"Moderate alignment with JD requirements; {engagement}."
        else:
            s2 = "Moderate alignment with JD requirements."
    else:
        # Lower-ranked — honest about why they are included
        if concern:
            s2 = f"Adjacent skills justify inclusion at this rank; {concern}."
        else:
            s2 = (
                "Adjacent skills and acceptable engagement signals justify "
                "inclusion near the cut-off."
            )

    return f"{s1} {s2}".strip()


def generate_reasoning_batch(
    candidates: list[ParsedCandidate],
    feature_vectors: list[FeatureVector],
    ranks: list[int],
    scores: list[float],
) -> list[str]:
    """
    Generate reasoning strings for a list of candidates.

    Args:
        candidates:      List of ParsedCandidate objects.
        feature_vectors: Corresponding FeatureVector objects.
        ranks:           1-indexed ranks.
        scores:          Final composite scores.

    Returns:
        List of reasoning strings.
    """
    return [
        generate_reasoning(c, fv, r, s)
        for c, fv, r, s in zip(candidates, feature_vectors, ranks, scores)
    ]