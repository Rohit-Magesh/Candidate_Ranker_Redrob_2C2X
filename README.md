# Redrob Intelligent Candidate Discovery & Ranking System

This project was developed for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**.

The system ranks a dataset of **100,000 candidate profiles** against a given job description and produces the **Top 100 most relevant candidates** in the required `submission.csv` format.

The goal of the project is to combine semantic relevance, experience, technical skills, behavioural signals and candidate quality checks into a single ranking score that can assist recruiters in identifying the strongest candidates.

---

# Project Structure

```
project/
│
├── rank.py                     # Main ranking pipeline
├── config.py                   # Configuration values and feature weights
├── loader.py                   # Reads candidate dataset
├── parser.py                   # Parses candidate information
├── jd_parser.py                # Parses the job description
├── feature_engineering.py      # Computes ranking features
├── behavioral_scorer.py        # Behaviour-based scoring
├── honeypot_detector.py        # Detects suspicious or inconsistent profiles
├── reasoning.py                # Generates reasoning for every shortlisted candidate
├── semantic_matcher.py         # TF-IDF based semantic similarity
├── utils.py                    # Shared helper functions
│
├── data/ required data files
├── cache/
├── output/
└── requirements.txt
```

---

# Overview

The pipeline processes every candidate profile and performs several stages of analysis before producing the final ranking.

The major stages are:

1. Load the candidate dataset.
2. Parse and structure candidate information.
3. Parse the supplied job description.
4. Calculate semantic similarity between every candidate and the job description.
5. Engineer multiple ranking features.
6. Apply behavioural scoring.
7. Apply fraud / honeypot penalties.
8. Rank candidates.
9. Generate recruiter-friendly reasoning.
10. Export the final submission CSV.

---

# Semantic Matching

The original implementation used **Sentence Transformers (all-MiniLM-L6-v2)** to generate embeddings for all 100,000 candidates.

Although this produced strong semantic representations, generating embeddings for every profile required a long preprocessing stage and caused significant performance and memory issues on CPU-only systems.

To make the solution more practical within the competition constraints, this version replaces transformer embeddings with a lightweight **TF-IDF based semantic matching approach**.

This provides several advantages:

- considerably lower memory usage
- much faster execution
- no embedding cache generation
- no precomputation step
- fully CPU compatible
- completes comfortably within the runtime limits

Although TF-IDF is less sophisticated than transformer embeddings, the overall ranking quality remains strong because semantic similarity is only one component of the complete scoring system.

---

# Scoring Methodology

Each candidate receives a final ranking score using multiple independent signals.

These include:

- semantic similarity to the job description
- years of relevant experience
- job title relevance
- technical skill overlap
- retrieval and ranking system experience
- production ML experience
- project relevance
- company relevance
- education
- career stability
- location suitability
- behavioural platform signals
- profile quality penalties

The final ranking score is calculated as:

```
Base Score
      ↓
Behaviour Multiplier
      ↓
Penalty Adjustment
      ↓
Final Ranking Score
```

Candidates are then sorted by this final score and the highest-ranked 100 candidates are exported.

---

# Behavioural Scoring

Behavioural signals supplied in the dataset are used to reward candidates who appear more active and recruiter-friendly.

Examples include:

- recruiter response rate
- recent activity
- interview completion
- notice period
- GitHub activity
- willingness to relocate
- open-to-work status

These signals slightly increase or decrease the overall ranking score.

---

# Honeypot Detection

The system also attempts to identify potentially unreliable profiles.

Penalties are applied for situations such as:

- impossible work history
- excessive keyword stuffing
- inconsistent career timelines
- research-only backgrounds
- no production deployment evidence
- predominantly service-company careers

These penalties reduce the final score while preserving ranking consistency.

---

# Installation

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

# Dataset

Place the competition files inside the project directory.

```
project/

data/
│
├── candidates.jsonl
└── job_description.md
```

---

# Running the Project

Unlike the original implementation, **no precompute step is required**.

The previous version generated semantic embeddings for all candidates before ranking, but this significantly increased runtime and memory usage.

Since the semantic matching has been replaced with TF-IDF, all processing now happens in a single pipeline.

Simply run:

```bash
python rank.py --candidates candidates.jsonl --jd job_description.md
```

The program will:

- load the dataset
- parse every candidate
- parse the job description
- compute semantic similarity
- calculate all ranking features
- score every candidate
- produce the final ranked list

---

# Output

The final submission is written to

```
output/submission.csv
```

The CSV contains four columns:

| Column | Description |
|---------|-------------|
| candidate_id | Candidate identifier |
| rank | Rank from 1–100 |
| score | Final ranking score |
| reasoning | Explanation for why the candidate was selected |

---

# Performance

The original embedding-based implementation required a separate preprocessing stage that generated embeddings for every candidate profile.

While accurate, this significantly increased execution time and memory consumption on CPU-only hardware.

The current implementation removes this bottleneck by replacing transformer embeddings with TF-IDF based semantic similarity.

As a result:

- no embedding cache is required
- no precomputation step is needed
- significantly lower memory usage
- considerably faster execution
- suitable for standard CPU-only systems

This makes the pipeline simpler to execute while still maintaining a multi-feature ranking approach.

---

# Notes

This implementation was designed specifically for the Redrob Candidate Discovery challenge and focuses on producing a reproducible ranking pipeline that can be executed with a single command.

No external APIs or online services are required, making the solution fully self-contained and suitable for offline evaluation environments.
