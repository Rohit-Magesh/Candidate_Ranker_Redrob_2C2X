"""
config.py — Centralized configuration for all scoring weights and parameters.
Grounded in the actual Redrob challenge schema and JD.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"

CANDIDATES_PATH = BASE_DIR / "candidates.jsonl.gz"
JD_PATH = BASE_DIR / "job_description.md"
OUTPUT_CSV = OUTPUT_DIR / "submission.csv"

# ─── Model ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 256
EMBEDDING_CACHE_FILE = CACHE_DIR / "embeddings.npz"

# ─── Top-K ────────────────────────────────────────────────────────────────────
TOP_K = 100

# ─── Feature weights (should sum to 1.0) ─────────────────────────────────────
FEATURE_WEIGHTS: dict[str, float] = {
    "semantic_similarity":     0.28,
    "experience_score":        0.12,
    "title_relevance":         0.08,
    "skill_overlap":           0.10,
    "retrieval_relevance":     0.08,
    "ranking_relevance":       0.06,
    "production_ml_relevance": 0.06,
    "evaluation_relevance":    0.04,
    "project_relevance":       0.05,
    "company_relevance":       0.04,
    "career_stability":        0.03,
    "education_relevance":     0.03,
    "location_relevance":      0.03,
}

# ─── Behavioral signal weights ────────────────────────────────────────────────
BEHAVIORAL_WEIGHTS: dict[str, float] = {
    "recruiter_response_rate":   0.22,
    "profile_completeness":      0.14,
    "interview_completion_rate": 0.14,
    "github_activity_score":     0.12,
    "last_active_recency":       0.12,
    "open_to_work":              0.10,
    "saved_by_recruiters":       0.08,
    "search_appearances":        0.05,
    "notice_period":             0.03,
}

# Behavioral multiplier range — never zeroes out a good semantic score
BEHAVIORAL_MIN_MULTIPLIER = 0.72
BEHAVIORAL_MAX_MULTIPLIER = 1.15

# ─── Penalty weights ─────────────────────────────────────────────────────────
PENALTY_WEIGHTS: dict[str, float] = {
    "impossible_history":       0.30,
    "honeypot_score":           0.25,
    "keyword_stuffing":         0.15,
    "inactive_profile":         0.10,
    "poor_recruiter_response":  0.10,
    "service_company_only":     0.05,
    "research_only":            0.05,
}

MAX_TOTAL_PENALTY = 0.50   # final_score *= (1 - clamped_penalty)

# ─── Notice period thresholds (JD explicitly wants <30d, can buy out 30d) ─────
NOTICE_IDEAL_DAYS = 30
NOTICE_LONG_DAYS = 60
NOTICE_VERY_LONG_DAYS = 90

# ─── Last-active thresholds ───────────────────────────────────────────────────
LAST_ACTIVE_FRESH_DAYS = 30
LAST_ACTIVE_STALE_DAYS = 180

# ─── Experience targets (JD says 5-9 years, sweet-spot 6-8) ──────────────────
EXP_MIN_YEARS = 3.0
EXP_IDEAL_MIN = 5.0
EXP_IDEAL_MAX = 9.0
EXP_HARD_MAX = 20.0

# ─── Skill ontology — raw name → canonical group ─────────────────────────────
SKILL_ALIASES: dict[str, str] = {
    # Information retrieval / search
    "information retrieval": "information_retrieval",
    "ir": "information_retrieval",
    "dense retrieval": "information_retrieval",
    "sparse retrieval": "information_retrieval",
    "bm25": "information_retrieval",
    "tf-idf": "information_retrieval",
    "tfidf": "information_retrieval",
    "neural ir": "information_retrieval",
    # Search infrastructure
    "elasticsearch": "search_infrastructure",
    "opensearch": "search_infrastructure",
    "solr": "search_infrastructure",
    "hybrid search": "search_infrastructure",
    "semantic search": "search_infrastructure",
    "neural search": "search_infrastructure",
    "full-text search": "search_infrastructure",
    "search infrastructure": "search_infrastructure",
    # Vector / embeddings
    "embeddings": "vector_search",
    "vector search": "vector_search",
    "vector database": "vector_search",
    "vector db": "vector_search",
    "faiss": "vector_search",
    "annoy": "vector_search",
    "hnswlib": "vector_search",
    "pinecone": "vector_search",
    "weaviate": "vector_search",
    "milvus": "vector_search",
    "qdrant": "vector_search",
    "chroma": "vector_search",
    "chromadb": "vector_search",
    "sentence-transformers": "vector_search",
    "sentence transformers": "vector_search",
    "text embeddings": "vector_search",
    "bi-encoder": "vector_search",
    "bge": "vector_search",
    "e5": "vector_search",
    # Recommendation systems
    "recommendation systems": "recommender_systems",
    "recommendation system": "recommender_systems",
    "recommender system": "recommender_systems",
    "recommender systems": "recommender_systems",
    "collaborative filtering": "recommender_systems",
    "matrix factorization": "recommender_systems",
    "personalization": "recommender_systems",
    "item2vec": "recommender_systems",
    # Ranking
    "learning to rank": "ranking_systems",
    "ltr": "ranking_systems",
    "lambdamart": "ranking_systems",
    "reranking": "ranking_systems",
    "re-ranking": "ranking_systems",
    "cross-encoder": "ranking_systems",
    "cross encoder": "ranking_systems",
    "ranking": "ranking_systems",
    "xgboost ranking": "ranking_systems",
    # RAG
    "rag": "rag",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    # Production ML / MLOps
    "mlops": "production_ml",
    "ml ops": "production_ml",
    "model serving": "production_ml",
    "model deployment": "production_ml",
    "feature store": "production_ml",
    "kubeflow": "production_ml",
    "airflow": "production_ml",
    "mlflow": "production_ml",
    "bentoml": "production_ml",
    "triton": "production_ml",
    "ray serve": "production_ml",
    "torchserve": "production_ml",
    # Evaluation
    "ndcg": "evaluation_frameworks",
    "mrr": "evaluation_frameworks",
    "map@k": "evaluation_frameworks",
    "precision@k": "evaluation_frameworks",
    "recall@k": "evaluation_frameworks",
    "a/b testing": "evaluation_frameworks",
    "ab testing": "evaluation_frameworks",
    "online evaluation": "evaluation_frameworks",
    "offline evaluation": "evaluation_frameworks",
    "offline metrics": "evaluation_frameworks",
    "ranking evaluation": "evaluation_frameworks",
    # Python / core ML libs
    "python": "python",
    "pyspark": "python",
    "pytorch": "python",
    "tensorflow": "python",
    "scikit-learn": "python",
    "sklearn": "python",
    "numpy": "python",
    "pandas": "python",
    "huggingface": "python",
    "transformers": "python",
    "langchain": "python",   # lower weight — framework not system
    # NLP
    "nlp": "nlp",
    "natural language processing": "nlp",
    "text classification": "nlp",
    "named entity recognition": "nlp",
    "ner": "nlp",
    # LLM fine-tuning (preferred but not must-have)
    "lora": "llm_finetune",
    "qlora": "llm_finetune",
    "peft": "llm_finetune",
    "fine-tuning": "llm_finetune",
    "fine tuning llms": "llm_finetune",
    "finetuning": "llm_finetune",
}

# Canonical skill groups that are directly required by JD
JD_CORE_SKILLS: list[str] = [
    "information_retrieval",
    "search_infrastructure",
    "vector_search",
    "recommender_systems",
    "ranking_systems",
    "production_ml",
    "evaluation_frameworks",
    "python",
]

# Preferred but not disqualifying
JD_PREFERRED_SKILLS: list[str] = [
    "rag",
    "nlp",
    "llm_finetune",
]

# ─── Service companies (negatives per JD) ────────────────────────────────────
SERVICE_COMPANY_KEYWORDS: list[str] = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "hcltech",
    "tech mahindra", "mphasis", "hexaware", "niit technologies",
    "mastech", "l&t infotech", "ltimindtree",
]

# Product company size indicators (not definitive, just signals)
LARGE_PRODUCT_COMPANY_KEYWORDS: list[str] = [
    "google", "meta", "amazon", "microsoft", "apple", "netflix",
    "flipkart", "swiggy", "zomato", "ola", "paytm", "razorpay",
    "cred", "meesho", "phonepe", "groww", "zepto",
]

# ─── Location signals (JD prefers Pune/Noida, open to Hyd/Mumbai/Delhi NCR) ──
PREFERRED_LOCATIONS: list[str] = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
    "bangalore", "bengaluru",
]
INDIA_REQUIRED = True  # JD says outside India is case-by-case

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

# ─── Determinism ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42