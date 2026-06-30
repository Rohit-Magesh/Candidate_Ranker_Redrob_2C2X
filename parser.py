from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from config import SERVICE_COMPANY_KEYWORDS, SKILL_ALIASES
from utils import (
    get_logger,
    normalize_text,
    parse_date,
    safe_float,
    safe_int,
    today,
    years_between,
)

logger = get_logger(__name__)


@dataclass
class WorkEntry:
    company: str
    company_normalized: str
    title: str
    title_normalized: str
    start_date: Optional[date]
    end_date: Optional[date]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str
    description_normalized: str
    is_service_company: bool


@dataclass
class SkillEntry:
    name: str
    name_normalized: str
    canonical_group: str          # mapped via SKILL_ALIASES or "other"
    proficiency: str              # beginner / intermediate / advanced / expert
    endorsements: int
    duration_months: int


@dataclass
class EducationEntry:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    tier: str                      # tier_1 … tier_4 / unknown


@dataclass
class RedrobSignals:
    profile_completeness: float
    signup_date: Optional[date]
    last_active_date: Optional[date]
    open_to_work: bool
    profile_views_30d: int
    applications_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    salary_min_lpa: float
    salary_max_lpa: float
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float   # -1 if not linked
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float   # -1 if no history
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass
class ParsedCandidate:
    candidate_id: str
    headline: str
    summary: str
    location: str
    location_normalized: str
    country: str
    years_of_experience: float
    current_title: str
    current_title_normalized: str
    current_company: str
    current_company_normalized: str
    current_company_size: str
    current_industry: str

    career_history: list[WorkEntry] = field(default_factory=list)
    education: list[EducationEntry] = field(default_factory=list)
    skills: list[SkillEntry] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)

    signals: Optional[RedrobSignals] = None

    # ── Derived / convenience fields set by parser ──────────────────────────
    full_text: str = ""                    # concatenated text for embedding
    canonical_skill_groups: set[str] = field(default_factory=set)
    total_career_months: int = 0
    service_company_fraction: float = 0.0
    has_production_experience: bool = False
    is_research_only: bool = False


# ────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ────────────────────────────────────────────────────────────────────────────

def _normalize_company(name: str) -> str:
    n = normalize_text(name)
    n = n.replace("pvt ltd", "").replace("private limited", "")
    n = n.replace("ltd", "").replace("inc", "").replace("llc", "")
    n = n.replace("technologies", "tech").replace("solutions", "sol")
    return n.strip()


def _is_service_company(company_normalized: str) -> bool:
    for kw in SERVICE_COMPANY_KEYWORDS:
        if kw in company_normalized:
            return True
    return False


def _normalize_skill(name: str) -> tuple[str, str]:
    """Return (normalized_name, canonical_group)."""
    norm = normalize_text(name)
    canonical = SKILL_ALIASES.get(norm, SKILL_ALIASES.get(name.lower().strip(), "other"))
    return norm, canonical


def _parse_work_entry(raw: dict) -> WorkEntry:
    company = raw.get("company", "")
    comp_norm = _normalize_company(company)
    title = raw.get("title", "")
    start = parse_date(raw.get("start_date"))
    end_raw = raw.get("end_date")
    end = parse_date(end_raw) if end_raw else None
    desc = raw.get("description", "")
    return WorkEntry(
        company=company,
        company_normalized=comp_norm,
        title=title,
        title_normalized=normalize_text(title),
        start_date=start,
        end_date=end,
        duration_months=safe_int(raw.get("duration_months", 0)),
        is_current=bool(raw.get("is_current", False)),
        industry=raw.get("industry", ""),
        company_size=raw.get("company_size", ""),
        description=desc,
        description_normalized=normalize_text(desc),
        is_service_company=_is_service_company(comp_norm),
    )


def _parse_skill(raw: dict) -> SkillEntry:
    name = raw.get("name", "")
    norm, canonical = _normalize_skill(name)
    return SkillEntry(
        name=name,
        name_normalized=norm,
        canonical_group=canonical,
        proficiency=raw.get("proficiency", "beginner"),
        endorsements=safe_int(raw.get("endorsements", 0)),
        duration_months=safe_int(raw.get("duration_months", 0)),
    )


def _parse_education(raw: dict) -> EducationEntry:
    return EducationEntry(
        institution=raw.get("institution", ""),
        degree=raw.get("degree", ""),
        field_of_study=raw.get("field_of_study", ""),
        start_year=safe_int(raw.get("start_year", 0)),
        end_year=safe_int(raw.get("end_year", 0)),
        tier=raw.get("tier", "unknown"),
    )


def _parse_signals(raw: dict) -> RedrobSignals:
    salary_range = raw.get("expected_salary_range_inr_lpa", {}) or {}
    return RedrobSignals(
        profile_completeness=safe_float(raw.get("profile_completeness_score", 0)),
        signup_date=parse_date(raw.get("signup_date")),
        last_active_date=parse_date(raw.get("last_active_date")),
        open_to_work=bool(raw.get("open_to_work_flag", False)),
        profile_views_30d=safe_int(raw.get("profile_views_received_30d", 0)),
        applications_30d=safe_int(raw.get("applications_submitted_30d", 0)),
        recruiter_response_rate=safe_float(raw.get("recruiter_response_rate", 0)),
        avg_response_time_hours=safe_float(raw.get("avg_response_time_hours", 0)),
        skill_assessment_scores=dict(raw.get("skill_assessment_scores", {}) or {}),
        connection_count=safe_int(raw.get("connection_count", 0)),
        endorsements_received=safe_int(raw.get("endorsements_received", 0)),
        notice_period_days=safe_int(raw.get("notice_period_days", 0)),
        salary_min_lpa=safe_float(salary_range.get("min", 0)),
        salary_max_lpa=safe_float(salary_range.get("max", 0)),
        preferred_work_mode=raw.get("preferred_work_mode", "flexible"),
        willing_to_relocate=bool(raw.get("willing_to_relocate", False)),
        github_activity_score=safe_float(raw.get("github_activity_score", -1)),
        search_appearance_30d=safe_int(raw.get("search_appearance_30d", 0)),
        saved_by_recruiters_30d=safe_int(raw.get("saved_by_recruiters_30d", 0)),
        interview_completion_rate=safe_float(raw.get("interview_completion_rate", 0)),
        offer_acceptance_rate=safe_float(raw.get("offer_acceptance_rate", -1)),
        verified_email=bool(raw.get("verified_email", False)),
        verified_phone=bool(raw.get("verified_phone", False)),
        linkedin_connected=bool(raw.get("linkedin_connected", False)),
    )


def _build_full_text(parsed: "ParsedCandidate") -> str:
    """Build a single text string that captures the essence of the candidate."""
    parts: list[str] = [
        parsed.headline,
        parsed.summary,
        parsed.current_title,
    ]
    for w in parsed.career_history:
        parts.append(w.title)
        parts.append(w.description)
    for s in parsed.skills:
        parts.append(s.name)
    for e in parsed.education:
        parts.append(e.field_of_study)
        parts.append(e.degree)
    for cert in parsed.certifications:
        parts.append(cert)
    return " ".join(p for p in parts if p)


def _has_production_experience(career: list[WorkEntry]) -> bool:
    """
    True if any role description mentions production, deployment, shipping,
    live, scale, serving, API, inference, or real users.
    """
    prod_kws = {
        "production", "deploy", "deployed", "live", "serving", "inference",
        "scale", "real users", "shipped", "launched", "api", "service",
        "production system", "production environment",
    }
    for w in career:
        desc_lower = w.description.lower()
        if any(kw in desc_lower for kw in prod_kws):
            return True
    return False


def _is_research_only(career: list[WorkEntry], current_title: str) -> bool:
    """
    True if career is dominated by academic / research roles with no
    product-company or engineering role.
    """
    research_kws = {"research", "researcher", "phd", "lab", "university",
                    "academic", "intern", "faculty", "professor"}
    product_kws = {"engineer", "developer", "scientist", "lead", "architect",
                   "manager", "director", "analyst", "product"}

    titles = [normalize_text(w.title) for w in career] + [normalize_text(current_title)]
    research_count = sum(1 for t in titles if any(kw in t for kw in research_kws))
    product_count = sum(1 for t in titles if any(kw in t for kw in product_kws))
    if not titles:
        return False
    return research_count > 0 and product_count == 0


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def parse_candidate(raw: dict) -> ParsedCandidate:
    """
    Convert a raw candidate dict from JSONL into a ParsedCandidate dataclass.

    Args:
        raw: Raw dict with keys matching the Redrob schema.

    Returns:
        ParsedCandidate with all fields populated.
    """
    profile = raw.get("profile", {}) or {}
    cid = raw.get("candidate_id", "UNKNOWN")

    career = [_parse_work_entry(w) for w in (raw.get("career_history") or [])]
    edu = [_parse_education(e) for e in (raw.get("education") or [])]
    skills = [_parse_skill(s) for s in (raw.get("skills") or [])]
    certs = [c.get("name", "") for c in (raw.get("certifications") or [])]
    signals = _parse_signals(raw.get("redrob_signals") or {})

    # Total career months
    total_months = sum(w.duration_months for w in career)

    # Service-company fraction
    sc_months = sum(w.duration_months for w in career if w.is_service_company)
    sc_frac = sc_months / max(total_months, 1)

    # Canonical skill groups
    canonical_groups = {s.canonical_group for s in skills}

    current_title = profile.get("current_title", "")
    location = profile.get("location", "")

    p = ParsedCandidate(
        candidate_id=cid,
        headline=profile.get("headline", ""),
        summary=profile.get("summary", ""),
        location=location,
        location_normalized=normalize_text(location),
        country=profile.get("country", ""),
        years_of_experience=safe_float(profile.get("years_of_experience", 0)),
        current_title=current_title,
        current_title_normalized=normalize_text(current_title),
        current_company=profile.get("current_company", ""),
        current_company_normalized=_normalize_company(profile.get("current_company", "")),
        current_company_size=profile.get("current_company_size", ""),
        current_industry=profile.get("current_industry", ""),
        career_history=career,
        education=edu,
        skills=skills,
        certifications=certs,
        signals=signals,
        canonical_skill_groups=canonical_groups,
        total_career_months=total_months,
        service_company_fraction=sc_frac,
        has_production_experience=_has_production_experience(career),
        is_research_only=_is_research_only(career, current_title),
    )
    p.full_text = _build_full_text(p)
    return p


def parse_candidates_batch(raws: list[dict]) -> list[ParsedCandidate]:
    """Parse a list of raw candidate dicts. Logs warnings for failures."""
    parsed = []
    for raw in raws:
        try:
            parsed.append(parse_candidate(raw))
        except Exception as exc:
            cid = raw.get("candidate_id", "?")
            logger.warning("Failed to parse candidate %s: %s", cid, exc)
    return parsed
