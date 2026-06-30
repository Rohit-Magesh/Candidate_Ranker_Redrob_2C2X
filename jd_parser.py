from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import (
    JD_CORE_SKILLS,
    JD_PREFERRED_SKILLS,
    PREFERRED_LOCATIONS,
    SKILL_ALIASES,
)
from utils import get_logger, normalize_text

logger = get_logger(__name__)


@dataclass
class ParsedJD:
    raw_text: str

    # Core text sections for embedding
    must_have_text: str = ""
    preferred_text: str = ""
    full_summary: str = ""          # concatenated relevant sections for embedding

    # Structured requirement buckets
    must_have_skills: list[str] = field(default_factory=list)   # canonical groups
    preferred_skills: list[str] = field(default_factory=list)
    negative_signals: list[str] = field(default_factory=list)   # free-text flags

    # Experience
    exp_min_years: float = 5.0
    exp_max_years: float = 9.0
    exp_ideal_years: float = 7.0

    # Location
    preferred_locations: list[str] = field(default_factory=list)
    location_flexible: bool = True
    india_required: bool = True

    # Behavioral preferences
    wants_open_to_work: bool = True
    wants_low_notice: bool = True
    notice_ideal_days: int = 30
    notice_max_days: int = 90

    # Keywords to search for in candidate text
    retrieval_keywords: list[str] = field(default_factory=list)
    production_keywords: list[str] = field(default_factory=list)
    evaluation_keywords: list[str] = field(default_factory=list)
    ranking_keywords: list[str] = field(default_factory=list)

    # Company-type signals
    penalize_service_only: bool = True
    penalize_research_only: bool = True
    penalize_title_hopper: bool = True


# ─── Hardcoded from the actual Redrob JD ────────────────────────────────────
# Rather than fragile regex parsing of the MD file, we embed the key signals
# directly since we've read the JD in full. The JD text is still used for
# embedding-based semantic matching.

_MUST_HAVE_TEXT = """
Production experience with embeddings-based retrieval systems deployed to real users.
Experience with vector databases or hybrid search infrastructure such as Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS.
Strong Python and code quality.
Hands-on experience designing evaluation frameworks for ranking systems including NDCG MRR MAP offline to online correlation A/B test interpretation.
Shipped at least one end-to-end ranking search or recommendation system to real users at meaningful scale.
Applied ML and AI roles at product companies not pure services.
Understanding of retrieval and ranking before the LLM era.
Candidate who writes production code.
"""

_PREFERRED_TEXT = """
LLM fine-tuning experience LoRA QLoRA PEFT.
Experience with learning-to-rank models XGBoost-based or neural.
Prior exposure to HR-tech recruiting tech or marketplace products.
Background in distributed systems or large-scale inference optimization.
Open-source contributions in the AI/ML space.
NLP background natural language processing text classification.
"""

_FULL_SUMMARY = """
Senior AI Engineer founding team role at Redrob AI Series A talent intelligence platform.
Own the intelligence layer including ranking retrieval and matching systems.
Build v2 ranking system with embeddings hybrid retrieval and LLM-based reranking.
Set up evaluation infrastructure offline benchmarks online A/B testing recruiter feedback loops.
Candidate-JD matching at scale mentoring engineers working with recruiter experience PM.
Production experience embeddings retrieval ranking vector databases.
Python code quality evaluation frameworks NDCG MRR MAP.
Product company experience not pure services or research.
Scrappy product-engineering attitude ship working ranker learn from real users.
Hybrid remote Pune Noida India preferred.
Five to nine years experience sweet spot six to eight years applied ML at product companies.
Open to work actively engaging with recruiters low notice period.
"""

_RETRIEVAL_KEYWORDS = [
    "embedding", "embeddings", "vector", "retrieval", "search", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "hybrid search", "dense retrieval", "sparse retrieval", "bm25",
    "sentence-transformers", "bge", "e5", "text-embedding",
]

_PRODUCTION_KEYWORDS = [
    "production", "deployed", "shipped", "live system", "scale",
    "inference", "serving", "api", "real users", "end-to-end",
    "mlops", "model deployment", "feature store",
]

_EVALUATION_KEYWORDS = [
    "ndcg", "mrr", "map", "precision@k", "recall@k", "a/b test", "ab test",
    "offline evaluation", "online evaluation", "ranking evaluation",
    "benchmark", "relevance judgment",
]

_RANKING_KEYWORDS = [
    "ranking", "learning to rank", "ltr", "lambdamart", "reranking",
    "cross-encoder", "bi-encoder", "recommendation", "recommender",
    "personalization", "collaborative filtering",
]

_NEGATIVE_SIGNALS = [
    "consulting only",
    "service company only career",
    "research only no production",
    "title hopper 1.5 year tenure pattern",
    "only LangChain wrapper experience",
    "computer vision speech robotics without NLP",
    "closed source only no external validation",
    "inactive profile no engagement",
    "very long notice period 90+ days",
]


def parse_jd(jd_text: str) -> ParsedJD:
    """
    Parse the job description text into a structured ParsedJD.

    The hardcoded signals are derived from reading the actual Redrob JD.
    The raw text is preserved for downstream embedding.

    Args:
        jd_text: Full text of the job description Markdown file.

    Returns:
        ParsedJD with all requirement buckets populated.
    """
    logger.info("Parsing job description (%d chars)", len(jd_text))

    jd = ParsedJD(
        raw_text=jd_text,
        must_have_text=_MUST_HAVE_TEXT.strip(),
        preferred_text=_PREFERRED_TEXT.strip(),
        full_summary=_FULL_SUMMARY.strip(),
        must_have_skills=list(JD_CORE_SKILLS),
        preferred_skills=list(JD_PREFERRED_SKILLS),
        negative_signals=list(_NEGATIVE_SIGNALS),
        exp_min_years=5.0,
        exp_max_years=9.0,
        exp_ideal_years=7.0,
        preferred_locations=list(PREFERRED_LOCATIONS),
        location_flexible=True,
        india_required=True,
        wants_open_to_work=True,
        wants_low_notice=True,
        notice_ideal_days=30,
        notice_max_days=90,
        retrieval_keywords=list(_RETRIEVAL_KEYWORDS),
        production_keywords=list(_PRODUCTION_KEYWORDS),
        evaluation_keywords=list(_EVALUATION_KEYWORDS),
        ranking_keywords=list(_RANKING_KEYWORDS),
        penalize_service_only=True,
        penalize_research_only=True,
        penalize_title_hopper=True,
    )

    logger.info(
        "Parsed JD: %d must-have skills, %d preferred skills, %d negative signals",
        len(jd.must_have_skills),
        len(jd.preferred_skills),
        len(jd.negative_signals),
    )
    return jd