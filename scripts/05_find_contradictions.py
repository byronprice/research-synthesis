#!/usr/bin/env python3
"""
Phase 5: Detect contradictions across paper summaries.

Reads:  data/summaries/*.json
Writes: data/contradictions/candidates.jsonl    (unverified)
        data/contradictions/contradictions.jsonl (LLM-verified)

Run from project root:
  python scripts/05_find_contradictions.py [--candidates-only] [--sim-threshold 0.85]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.contradiction.candidate_finder import CandidateFinder
from src.contradiction.verifier import ContradictionVerifier
from src.embedding.embedder import Embedder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries-dir", default="data/summaries")
    p.add_argument("--output-dir", default="data/contradictions")
    p.add_argument("--sim-threshold", type=float, default=0.85)
    p.add_argument("--max-candidates", type=int, default=500,
                   help="Max candidates to send to LLM for verification")
    p.add_argument("--candidates-only", action="store_true",
                   help="Find candidates but skip LLM verification")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder()
    finder = CandidateFinder(embedder, sim_threshold=args.sim_threshold)
    candidates = finder.find(args.summaries_dir)

    # Save candidates
    candidates_path = out_dir / "candidates.jsonl"
    with open(candidates_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict()) + "\n")
    logger.info(f"Saved {len(candidates)} candidates → {candidates_path}")

    if args.candidates_only or not candidates:
        return

    # LLM verification
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Set ANTHROPIC_API_KEY to run LLM verification")
        sys.exit(1)

    verifier = ContradictionVerifier(api_key=api_key)
    verified = verifier.verify_batch(candidates, max_candidates=args.max_candidates)

    out_path = out_dir / "contradictions.jsonl"
    with open(out_path, "w") as f:
        for v in verified:
            f.write(json.dumps(v.to_dict()) + "\n")

    true_count = sum(1 for v in verified if v.verdict == "TRUE_CONTRADICTION")
    apparent_count = sum(1 for v in verified if v.verdict == "APPARENT_CONTRADICTION")
    logger.info(
        f"Verified contradictions → {out_path}\n"
        f"  TRUE_CONTRADICTION:     {true_count}\n"
        f"  APPARENT_CONTRADICTION: {apparent_count}"
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/05_find_contradictions.log", level="DEBUG", rotation="50 MB")
    Path("logs").mkdir(exist_ok=True)
    main()
