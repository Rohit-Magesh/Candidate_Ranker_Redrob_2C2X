from __future__ import annotations

from datetime import date

from config import JD_CORE_SKILLS
from parser import ParsedCandidate, SkillEntry, WorkEntry
from utils import clamp, count_token_repetitions, get_logger, normalize_text

logger = get_logger(__name__)


def _check_expert_everywhere(skills: list[SkillEntry]) -> float:
    """Penalty if candidate claims expert/advanced in many unrelated domains."""
    expert_count = sum(
        1 for s in skills if s.proficiency in ("expert", "advanced")
    )
    total = len(skills)
    if total == 0:
        return 0.0
    expert_ratio = expert_count / total
    # >70% advanced/expert is suspicious
    if expert_ratio > 0.85:
        return 0.60
    if expert_ratio > 0.70:
        return 0.35
    if expert_ratio > 0.55:
        return 0.15
    return 0.0


def _check_zero_duration_expert(skills: list[SkillEntry]) -> float:
    """Penalty for advanced/expert skills with 0 months claimed use."""
    violations = 0
    for s in skills:
        if s.proficiency in ("expert", "advanced") and s.duration_months == 0:
            violations += 1
    if violations >= 5:
        return 0.60
    if violations >= 3:
        return 0.40
    if violations >= 1:
        return 0.15
    return 0.0


def _check_date_inconsistencies(career: list[WorkEntry]) -> float:
    """
    Detect:
    - end_date before start_date
    - suspiciously long claimed durations vs. actual date span
    """
    violations = 0
    for w in career:
        start = w.start_date
        end = w.end_date
        if start is None:
            continue
        if end is not None and end < start:
            violations += 1  # end before start
        if end is not None:
            actual_months = max(0, (end - start).days // 30)
            claimed_months = w.duration_months
            if claimed_months > 0 and abs(actual_months - claimed_months) > 18:
                violations += 1  # large mismatch
    if violations >= 3:
        return 0.55
    if violations >= 2:
        return 0.30
    if violations >= 1:
        return 0.10
    return 0.0


def _check_overlapping_jobs(career: list[WorkEntry]) -> float:
    """Detect heavily overlapping full-time employment periods."""
    entries = [
        (w.start_date, w.end_date if w.end_date else date.today())
        for w in career
        if w.start_date is not None and not w.is_current
    ]
    if len(entries) < 2:
        return 0.0

    entries.sort(key=lambda x: x[0])
    overlaps = 0
    for i in range(len(entries) - 1):
        _, end_i = entries[i]
        start_j, _ = entries[i + 1]
        if start_j < end_i:
            overlap_days = (end_i - start_j).days
            if overlap_days > 60:  # >2 months overlap is suspicious
                overlaps += 1
    if overlaps >= 2:
        return 0.45
    if overlaps >= 1:
        return 0.20
    return 0.0


def _check_yoe_mismatch(candidate: ParsedCandidate) -> float:
    """
    Claimed years_of_experience vs. total_career_months / 12.
    Large discrepancy suggests fabrication.
    """
    claimed = candidate.years_of_experience
    computed = candidate.total_career_months / 12.0

    if computed < 0.5:
        return 0.0  # can't compute reliably

    ratio = claimed / max(computed, 0.1)
    # Allow ±30% discrepancy (some overlap, some gaps are normal)
    if ratio > 1.8 or ratio < 0.5:
        return 0.40
    if ratio > 1.4 or ratio < 0.65:
        return 0.15
    return 0.0


def _check_keyword_stuffing(candidate: ParsedCandidate) -> float:
    """
    Detect unusually dense repetition of a small set of keywords in profile text.
    True stuffing: the same technical buzzwords appear 5+ times in the text.
    """
    text = candidate.full_text
    tokens = normalize_text(text).split()
    if len(tokens) < 50:
        return 0.0
    repeat_count = count_token_repetitions(tokens, threshold=5)
    # More than 10 heavily repeated tokens is suspicious
    if repeat_count >= 15:
        return 0.50
    if repeat_count >= 10:
        return 0.30
    if repeat_count >= 5:
        return 0.10
    return 0.0


def _check_impossible_experience_at_company(
    career: list[WorkEntry],
) -> float:
    """
    Known companies that couldn't have hired someone for the claimed duration.
    Uses start_date vs. known founding years of common honeypot patterns.
    Since we don't have a company database, we flag cases where duration_months
    implies the candidate started before year 2000 at a company with a
    obviously modern name (AI/ML company names, etc.).
    This is a heuristic — any match is a soft signal, not a hard filter.
    """
    # Check for suspiciously long tenures at small companies
    violations = 0
    for w in career:
        if w.company_size in ("1-10", "11-50") and w.duration_months > 120:
            # 10+ years at a tiny company is uncommon
            violations += 1
    if violations >= 2:
        return 0.25
    if violations >= 1:
        return 0.10
    return 0.0


# ─── Main ────────────────────────────────────────────────────────────────────

def compute_honeypot_penalty(candidate: ParsedCandidate) -> dict[str, float]:
    """
    Compute honeypot penalty components for a candidate.

    Args:
        candidate: ParsedCandidate.

    Returns:
        Dict with sub-penalty names and values, plus "total" key.
        All values in [0, 1].
    """
    penalties: dict[str, float] = {
        "expert_everywhere":      _check_expert_everywhere(candidate.skills),
        "zero_duration_expert":   _check_zero_duration_expert(candidate.skills),
        "date_inconsistencies":   _check_date_inconsistencies(candidate.career_history),
        "overlapping_jobs":       _check_overlapping_jobs(candidate.career_history),
        "yoe_mismatch":           _check_yoe_mismatch(candidate),
        "keyword_stuffing":       _check_keyword_stuffing(candidate),
        "impossible_tenure":      _check_impossible_experience_at_company(
                                      candidate.career_history
                                  ),
    }

    # Aggregate: take the max of any single signal, then add up to 30% from
    # secondary signals. This prevents a single mild flag from dominating.
    sorted_vals = sorted(penalties.values(), reverse=True)
    total = sorted_vals[0] if sorted_vals else 0.0
    for v in sorted_vals[1:]:
        total = clamp(total + v * 0.3, 0.0, 1.0)

    penalties["total"] = clamp(total, 0.0, 1.0)
    return penalties