#!/usr/bin/env python3
"""
Phase 6a: Generate instruction-tuning data for fine-tuning.

Reads:  data/extracted/papers_metadata.jsonl
        data/extracted/papers_fulltext.jsonl
        data/summaries/*.json
        data/clusters/meta_summaries/*.json

Writes: data/training_data/category_{A,B,C,D}.jsonl
        data/training_data/combined_train.jsonl

Run from project root:
  python scripts/06_generate_training_data.py [--limit N] [--skip-c]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data_generator import TrainingDataGenerator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data/extracted/papers_metadata.jsonl")
    p.add_argument("--fulltext", default="data/extracted/papers_fulltext.jsonl")
    p.add_argument("--summaries-dir", default="data/summaries")
    p.add_argument("--clusters-dir", default="data/clusters/meta_summaries")
    p.add_argument("--output-dir", default="data/training_data")
    p.add_argument("--limit", type=int, default=None, help="Max papers to process")
    p.add_argument("--skip-c", action="store_true", help="Skip Category C (API comparison calls)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_jsonl(path: str) -> dict[int, dict]:
    records = {}
    if not Path(path).exists():
        return records
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["paper_id"]] = rec
    return records


def load_summaries(summaries_dir: str) -> dict[int, dict]:
    summaries = {}
    for fpath in Path(summaries_dir).glob("*.json"):
        try:
            with open(fpath) as f:
                s = json.load(f)
                summaries[s["paper_id"]] = s
        except Exception:
            pass
    return summaries


def load_cluster_metas(clusters_dir: str) -> list[dict]:
    metas = []
    for fpath in Path(clusters_dir).glob("*.json"):
        try:
            with open(fpath) as f:
                metas.append(json.load(f))
        except Exception:
            pass
    return metas


def write_jsonl(records: list[dict], path: Path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Wrote {len(records)} records → {path}")


def main():
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Set ANTHROPIC_API_KEY")
        sys.exit(1)

    metadata = load_jsonl(args.metadata)
    fulltext = load_jsonl(args.fulltext)
    summaries = load_summaries(args.summaries_dir)
    cluster_metas = load_cluster_metas(args.clusters_dir)

    paper_ids = sorted(set(metadata.keys()) & set(summaries.keys()))
    if args.limit:
        paper_ids = paper_ids[: args.limit]
    logger.info(f"Processing {len(paper_ids)} papers")

    gen = TrainingDataGenerator(api_key=api_key, model=args.model)

    cat_a, cat_b, cat_c, cat_d = [], [], [], []

    # Categories A and B (no API calls needed for B)
    for pid in tqdm(paper_ids, desc="Cat A+B"):
        meta = metadata[pid]
        ft = fulltext.get(pid, {})
        summary = summaries[pid]

        a = gen.generate_category_a(meta, ft, summary)
        if a:
            cat_a.append(a)

        b = gen.generate_category_b(meta, summary)
        if b:
            cat_b.append(b)

    write_jsonl(cat_a, out_dir / "category_a_summarization.jsonl")
    write_jsonl(cat_b, out_dir / "category_b_claims.jsonl")

    # Category C: pairwise comparisons (API calls)
    if not args.skip_c and len(paper_ids) >= 2:
        n_pairs = max(1, len(paper_ids) // 5)  # ~20% of paper count
        pairs = random.sample(
            [(i, j) for i in range(len(paper_ids)) for j in range(i+1, len(paper_ids))],
            min(n_pairs, 500),
        )
        for i, j in tqdm(pairs, desc="Cat C"):
            pid_a, pid_b = paper_ids[i], paper_ids[j]
            c = gen.generate_category_c(
                metadata[pid_a], summaries[pid_a],
                metadata[pid_b], summaries[pid_b],
            )
            if c:
                cat_c.append(c)
        write_jsonl(cat_c, out_dir / "category_c_comparisons.jsonl")

    # Category D: literature reviews from cluster metas (API calls)
    for meta in tqdm(cluster_metas, desc="Cat D"):
        d = gen.generate_category_d(meta)
        if d:
            cat_d.append(d)
    write_jsonl(cat_d, out_dir / "category_d_litreview.jsonl")

    # Combine and shuffle
    combined = cat_a + cat_b + cat_c + cat_d
    random.shuffle(combined)
    write_jsonl(combined, out_dir / "combined_train.jsonl")

    logger.info(
        f"Training data summary:\n"
        f"  A (summarization): {len(cat_a)}\n"
        f"  B (claims):        {len(cat_b)}\n"
        f"  C (comparisons):   {len(cat_c)}\n"
        f"  D (lit review):    {len(cat_d)}\n"
        f"  Total:             {len(combined)}"
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/06_generate_training_data.log", level="DEBUG", rotation="50 MB")
    Path("logs").mkdir(exist_ok=True)
    main()
