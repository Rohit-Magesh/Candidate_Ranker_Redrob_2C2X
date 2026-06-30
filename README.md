# Redrob Intelligent Candidate Discovery & Ranking System

Semantic ranking pipeline for the **Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge**.

Ranks 100,000 candidates against a job description and outputs the top 100 as `submission.csv`.

---

## Architecture

```
project/
├── rank.py               ← Main pipeline (run this to produce submission)
├── precompute.py         ← Optional: pre-generate embeddings cache
├── config.py             ← All weights and parameters
├── loader.py             ← Streaming gzip JSONL reader
├── parser.py             ← Structured feature extraction from raw profiles
├── jd_parser.py          ← Parse JD into weighted requirements
├── semantic_matcher.py   ← Sentence-transformer embedding + cosine     similarity
├── feature_engineering.py← Numerical feature vectors
├── behavioral_scorer.py  ← Redrob platform signal → multiplier
├── honeypot_detector.py  ← Detect impossible/fraudulent profiles
├── reasoning.py          ← Factual per-candidate reasoning strings
├── utils.py              ← Shared helpers
├── requirements.txt
├── models/               ← Sentence-transformer model cache (auto-populated)
├── cache/                ← Embedding cache (embeddings.npz)
└── output/               ← Submission CSV written here
```

---

## Installation

### 1. Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip

# CPU-only PyTorch (required by sentence-transformers; much smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# All other deps
pip install -r requirements.txt
```

### 3. Place the dataset

Copy the competition files to the project root:

```
project/
├── candidates.jsonl.gz        ← 100,000 candidates
└── job_description.md         ← Provided JD
```

---

## Usage

### Option A — Direct ranking (5-minute budget includes embedding)

```bash
python rank.py
# or with explicit paths:
python rank.py --candidates ./candidates.jsonl.gz --jd ./job_description.md --out ./output/submission.csv
```

Expected runtime: **3–5 minutes on a 16 GB CPU machine** (first run includes embedding ~100K texts).

### Option B — Precompute embeddings first (recommended for iteration)

```bash
# Step 1: precompute once (may take longer than 5 min for first run)
python precompute.py --candidates ./candidates.jsonl.gz

# Step 2: rank (loads from cache — runs in ~60-90 seconds)
python rank.py
```

---

## Output

`output/submission.csv` with columns:

| Column | Description |
|---|---|
| `candidate_id` | `CAND_XXXXXXX` format |
| `rank` | 1 (best) to 100 (100th best) |
| `score` | Composite score in [0, 1], non-increasing with rank |
| `reasoning` | 1–2 factual sentences grounded in the candidate's profile |

---

## Ranking methodology

1. **Semantic similarity** (30%): Sentence-transformer embeddings of candidate full text vs. JD summary. Model: `all-MiniLM-L6-v2` (CPU-fast, 384-dim).

2. **Feature engineering** (70%): 12 numerical features including experience score, title relevance, skill overlap (mapped to canonical JD-relevant groups), retrieval/ranking/production-ML relevance, evaluation framework evidence, project relevance, company type, career stability, education, and location.

3. **Behavioral multiplier**: Redrob platform signals (recruiter response rate, last-active recency, open-to-work, GitHub activity, interview completion, notice period, etc.) adjust the base score multiplicatively in [0.72, 1.15].

4. **Penalties**: Honeypot detection (impossible experience, expert-everywhere, date inconsistencies, keyword stuffing), service-company-only careers, research-only backgrounds, no production deployment evidence. Max cumulative penalty is capped at 50% score reduction.

### Scoring formula

```
base_score        = weighted_sum(feature_vector)
behavioral_score  = base_score × behavioral_multiplier
final_score       = behavioral_score × (1 − penalty)
```

---

## Configuration

All weights are in `config.py`:

- `FEATURE_WEIGHTS` — weight of each feature in the base score
- `BEHAVIORAL_WEIGHTS` — weight of each platform signal
- `PENALTY_WEIGHTS` — weight of each penalty type
- `EXP_IDEAL_MIN / MAX` — ideal experience range
- `NOTICE_IDEAL_DAYS` — ideal notice period

---

## Compute constraints

| Constraint | Limit | This system |
|---|---|---|
| Runtime | ≤5 min | ~1–2 min with cache, ~4 min first run |
| RAM | ≤16 GB | ~4–6 GB peak |
| Compute | CPU only | ✅ no GPU used |
| Network | Off | ✅ no external calls |

---

## Example commands

```bash
# Full run with defaults
python rank.py

# Custom paths
python rank.py \
  --candidates /data/candidates.jsonl.gz \
  --jd /data/job_description.md \
  --out /data/output/my_submission.csv

# Precompute then rank
python precompute.py --candidates ./candidates.jsonl.gz
python rank.py

# Force re-embed (ignore cache)
python precompute.py --force
python rank.py
```