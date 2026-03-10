#!/usr/bin/env python3
"""
Phase 3: Generate structured per-paper summaries via Claude API.

Reads:
  data/extracted/papers_metadata.jsonl
  data/extracted/papers_fulltext.jsonl

Writes:
  data/summaries/{paper_id}.json  — one file per paper

Run from project root:
  python scripts/03_summarize_papers.py [--limit N] [--resume] [--model MODEL]

Set ANTHROPIC_API_KEY in environment or pass via --api-key.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.summarization.per_paper import PaperSummarizer


def parse_args():
    p = argparse.ArgumentParser(description="Summarize papers via Claude API")
    p.add_argument("--metadata", default="data/extracted/papers_metadata.jsonl")
    p.add_argument("--fulltext", default="data/extracted/papers_fulltext.jsonl")
    p.add_argument("--output-dir", default="data/summaries")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Skip already-summarized papers")
    p.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model to use (haiku=cheap, sonnet=better quality)",
    )
    p.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    p.add_argument("--delay", type=float, default=0.1, help="Seconds to sleep between API calls")
    return p.parse_args()


def load_jsonl(path: str) -> dict[int, dict]:
    records = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["paper_id"]] = rec
    return records


def main():
    args = parse_args()

    for path in [args.metadata, args.fulltext]:
        if not Path(path).exists():
            logger.error(f"File not found: {path}. Run earlier pipeline steps first.")
            sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_jsonl(args.metadata)
    fulltext = load_jsonl(args.fulltext)

    paper_ids = sorted(set(metadata.keys()) & set(fulltext.keys()))
    if args.limit:
        paper_ids = paper_ids[: args.limit]

    if args.resume:
        existing = {int(p.stem) for p in out_dir.glob("*.json")}
        paper_ids = [pid for pid in paper_ids if pid not in existing]
        logger.info(f"Resuming: {len(existing)} already done, {len(paper_ids)} remaining")

    logger.info(f"Summarizing {len(paper_ids)} papers with {args.model}")

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("No API key. Set ANTHROPIC_API_KEY or pass --api-key.")
        sys.exit(1)

    summarizer = PaperSummarizer(api_key=api_key, model=args.model)

    stats = {"success": 0, "skipped": 0, "failed": 0}

    for pid in tqdm(paper_ids, desc="Summarizing"):
        meta = metadata[pid]
        ft = fulltext[pid]

        # Skip papers with no usable text
        if not meta.get("abstract") and not ft.get("full_text"):
            stats["skipped"] += 1
            continue

        summary = summarizer.summarize(meta, ft)
        if summary is None:
            stats["failed"] += 1
            continue

        out_path = out_dir / f"{pid}.json"
        with open(out_path, "w") as f:
            json.dump(summary.model_dump(), f, indent=2)
        stats["success"] += 1

        if args.delay > 0:
            time.sleep(args.delay)

    logger.info(
        f"Done. Success: {stats['success']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}"
    )
    # Rough cost estimate for haiku
    if "haiku" in args.model:
        cost_estimate = stats["success"] * 0.004
        logger.info(f"Estimated API cost (claude-haiku): ~${cost_estimate:.2f}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/03_summarize_papers.log", level="DEBUG", rotation="50 MB")
    Path("logs").mkdir(exist_ok=True)
    main()
