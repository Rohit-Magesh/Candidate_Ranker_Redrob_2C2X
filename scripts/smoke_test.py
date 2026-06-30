from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import JD_PATH
from rank import rank_candidates
from utils import get_logger

logger = get_logger("smoke_test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Path to a small sample candidates JSON file")
    parser.add_argument("--jd", type=Path, default=JD_PATH)
    args = parser.parse_args()

    if not args.candidates.exists():
        logger.error("Candidates file not found: %s", args.candidates)
        sys.exit(1)

    # Convert plain JSON to temp JSONL.gz for the pipeline
    import gzip
    import orjson

    with open(args.candidates, "rb") as f:
        records = json.load(f)

    with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with gzip.open(tmp_path, "wb") as gz:
            for rec in records:
                gz.write(orjson.dumps(rec) + b"\n")

    out_path = Path(tempfile.mktemp(suffix=".csv"))

    try:
        df = rank_candidates(tmp_path, args.jd, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Validate output
    print(f"\n{'='*60}")
    print(f"Smoke test output: {len(df)} rows")
    print(df.head(10).to_string())
    print(f"\nScore range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
    assert df["rank"].tolist() == list(range(1, len(df) + 1)), "Rank sequence broken"
    assert df["candidate_id"].nunique() == len(df), "Duplicate IDs"
    print("\n✅ Smoke test passed")

    out_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()