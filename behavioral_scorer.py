from __future__ import annotations

from config import (
    BEHAVIORAL_MAX_MULTIPLIER,
    BEHAVIORAL_MIN_MULTIPLIER,
    BEHAVIORAL_WEIGHTS,
    LAST_ACTIVE_FRESH_DAYS,
    LAST_ACTIVE_STALE_DAYS,
    NOTICE_IDEAL_DAYS,
    NOTICE_LONG_DAYS,
    NOTICE_VERY_LONG_DAYS,
)
from parser import ParsedCandidate, RedrobSignals
from utils import clamp, days_since, get_logger

logger = get_logger(__name__)


def _recruiter_response_score(sig: RedrobSignals) -> float:
    """High response rate = reliable candidate."""
    r = sig.recruiter_response_rate
    if r >= 0.80:
        return 1.0
    if r >= 0.60:
        return 0.80
    if r >= 0.40:
        return 0.60
    if r >= 0.20:
        return 0.40
    return 0.15  # very low — likely unreachable


def _profile_completeness_score(sig: RedrobSignals) -> float:
    """Normalise 0-100 to 0-1 with a slight bonus for near-complete."""
    p = sig.profile_completeness / 100.0
    if p >= 0.90:
        return 1.0
    return p


def _interview_completion_score(sig: RedrobSignals) -> float:
    """Higher interview completion rate = more reliable."""
    r = sig.interview_completion_rate
    if r >= 0.90:
        return 1.0
    if r >= 0.70:
        return 0.80
    if r >= 0.50:
        return 0.60
    if r >= 0.30:
        return 0.40
    return 0.20


def _github_activity_score(sig: RedrobSignals) -> float:
    """
    GitHub activity 0-100 → 0-1.
    -1 means no GitHub linked; treated as neutral (0.4) for engineers.
    """
    ga = sig.github_activity_score
    if ga < 0:
        return 0.40   # no GitHub — mild negative for an AI engineer role
    return clamp(ga / 100.0, 0.0, 1.0)


def _last_active_score(sig: RedrobSignals) -> float:
    """
    Recent activity = ready to engage.
    Inactive for >180 days = strong penalty.
    """
    d = days_since(sig.last_active_date)
    if d <= LAST_ACTIVE_FRESH_DAYS:
        return 1.0
    if d <= 60:
        return 0.85
    if d <= 90:
        return 0.70
    if d <= LAST_ACTIVE_STALE_DAYS:
        return 0.50
    return 0.15  # >180 days — stale profile


def _open_to_work_score(sig: RedrobSignals) -> float:
    """Open-to-work flag and active applications signal genuine availability."""
    base = 1.0 if sig.open_to_work else 0.40
    # Active applications reinforce availability
    if sig.applications_30d > 5:
        base = min(base + 0.10, 1.0)
    return base


def _saved_by_recruiters_score(sig: RedrobSignals) -> float:
    """Being saved by multiple recruiters = market validation."""
    n = sig.saved_by_recruiters_30d
    if n >= 10:
        return 1.0
    if n >= 5:
        return 0.80
    if n >= 2:
        return 0.60
    if n >= 1:
        return 0.45
    return 0.30


def _search_appearances_score(sig: RedrobSignals) -> float:
    """High search appearances = profile is searchable and matching queries."""
    n = sig.search_appearance_30d
    if n >= 50:
        return 1.0
    if n >= 20:
        return 0.80
    if n >= 10:
        return 0.65
    if n >= 5:
        return 0.50
    return 0.30


def _notice_period_score(sig: RedrobSignals) -> float:
    """
    JD wants sub-30-day notice. Can buy out 30 days.
    Long notice periods penalised progressively.
    """
    days = sig.notice_period_days
    if days <= NOTICE_IDEAL_DAYS:
        return 1.0
    if days <= NOTICE_LONG_DAYS:
        # Linear decay from 1.0 at 30d to 0.65 at 60d
        t = (days - NOTICE_IDEAL_DAYS) / (NOTICE_LONG_DAYS - NOTICE_IDEAL_DAYS)
        return 1.0 - 0.35 * t
    if days <= NOTICE_VERY_LONG_DAYS:
        # Decay from 0.65 at 60d to 0.40 at 90d
        t = (days - NOTICE_LONG_DAYS) / (NOTICE_VERY_LONG_DAYS - NOTICE_LONG_DAYS)
        return 0.65 - 0.25 * t
    # >90 days — significant concern
    return max(0.15, 0.40 - 0.01 * (days - NOTICE_VERY_LONG_DAYS))


def compute_behavioral_multiplier(candidate: ParsedCandidate) -> float:
    """
    Compute the behavioral multiplier for a candidate.

    Aggregates sub-scores from all signals into a weighted mean,
    then maps the result to [BEHAVIORAL_MIN_MULTIPLIER, BEHAVIORAL_MAX_MULTIPLIER].

    Args:
        candidate: ParsedCandidate with populated signals.

    Returns:
        Float multiplier in [0.72, 1.15].
    """
    sig = candidate.signals
    if sig is None:
        return 1.0  # no signals → neutral

    sub_scores: dict[str, float] = {
        "recruiter_response_rate":   _recruiter_response_score(sig),
        "profile_completeness":      _profile_completeness_score(sig),
        "interview_completion_rate": _interview_completion_score(sig),
        "github_activity_score":     _github_activity_score(sig),
        "last_active_recency":       _last_active_score(sig),
        "open_to_work":              _open_to_work_score(sig),
        "saved_by_recruiters":       _saved_by_recruiters_score(sig),
        "search_appearances":        _search_appearances_score(sig),
        "notice_period":             _notice_period_score(sig),
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for key, weight in BEHAVIORAL_WEIGHTS.items():
        score = sub_scores.get(key, 0.5)
        weighted_sum += score * weight
        total_weight += weight

    raw_score = weighted_sum / max(total_weight, 1e-9)   # in [0, 1]

    # Map [0, 1] → [BEHAVIORAL_MIN_MULTIPLIER, BEHAVIORAL_MAX_MULTIPLIER]
    span = BEHAVIORAL_MAX_MULTIPLIER - BEHAVIORAL_MIN_MULTIPLIER
    multiplier = BEHAVIORAL_MIN_MULTIPLIER + span * raw_score

    return clamp(multiplier, BEHAVIORAL_MIN_MULTIPLIER, BEHAVIORAL_MAX_MULTIPLIER)


def get_behavioral_sub_scores(candidate: ParsedCandidate) -> dict[str, float]:
    """
    Return the individual behavioral sub-scores for reasoning / debugging.
    """
    sig = candidate.signals
    if sig is None:
        return {}
    return {
        "recruiter_response_rate":   _recruiter_response_score(sig),
        "profile_completeness":      _profile_completeness_score(sig),
        "interview_completion_rate": _interview_completion_score(sig),
        "github_activity_score":     _github_activity_score(sig),
        "last_active_recency":       _last_active_score(sig),
        "open_to_work":              _open_to_work_score(sig),
        "saved_by_recruiters":       _saved_by_recruiters_score(sig),
        "search_appearances":        _search_appearances_score(sig),
        "notice_period":             _notice_period_score(sig),
    }