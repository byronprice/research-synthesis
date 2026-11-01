#!/usr/bin/env python3
"""
Phase 4: Claim-level clustering and cross-paper synthesis.

Loads all claims from data/summaries/ and data/reviews/, embeds them,
clusters via UMAP + HDBSCAN, then synthesizes each cluster with Claude
to produce consensus statements, boundary conditions, and emergent
hypotheses not stated in any individual paper.

Reads:
  data/summaries/{paper_id}.json
  data/reviews/{paper_id}.json
  data/extracted/papers_metadata.jsonl

Writes:
  data/clusters/claim_assignments.jsonl  — one record per claim with cluster_id
  data/clusters/{cluster_id}.json        — synthesis for each cluster

Run from project root:
  python scripts/04_cluster_and_meta.py [--no-synthesis] [--model MODEL]

Set ANTHROPIC_API_KEY for synthesis step.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.embedder import Embedder
from src.summarization.clustering import ClaimClusterer
from src.summarization.claim_synthesizer import ClaimSynthesizer


def parse_args():
    p = argparse.ArgumentParser(description="Cluster claims and synthesize cross-paper insights")
    p.add_argument("--summaries-dir", default="data/summaries")
    p.add_argument("--reviews-dir", default="data/reviews")
    p.add_argument("--metadata", default="data/extracted/papers_metadata.jsonl")
    p.add_argument("--output-dir", default="data/clusters")
    p.add_argument("--no-synthesis", action="store_true",
                   help="Only cluster, skip LLM synthesis step")
    p.add_argument("--resume", action="store_true",
                   help="Skip clusters that already have synthesis files")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--api-key", default=None)
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--delay", type=float, default=0.2,
                   help="Seconds to sleep between API calls")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Cluster ──────────────────────────────────────────────────────
    embedder = Embedder()
    clusterer = ClaimClusterer(
        embedder=embedder,
        hdbscan_min_cluster_size=args.min_cluster_size,
    )

    claims, labels = clusterer.cluster(
        summaries_dir=args.summaries_dir,
        reviews_dir=args.reviews_dir,
        metadata_path=args.metadata,
    )

    # Save claim assignments
    assignments_path = out_dir / "claim_assignments.jsonl"
    with open(assignments_path, "w") as f:
        for claim, label in zip(claims, labels):
            rec = claim.to_dict()
            rec["cluster_id"] = int(label)
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Saved {len(claims)} claim assignments to {assignments_path}")

    if args.no_synthesis:
        logger.info("--no-synthesis: skipping LLM synthesis step")
        return

    # ── Step 2: Group claims by cluster ─────────────────────────────────────
    cluster_claims: dict[int, list] = defaultdict(list)
    for claim, label in zip(claims, labels):
        if label == -1:
            continue  # skip noise
        cluster_claims[int(label)].append(claim)

    # Summarize cluster stats
    sizes = sorted(len(v) for v in cluster_claims.values())
    logger.info(
        f"{len(cluster_claims)} clusters | "
        f"size range: {sizes[0]}-{sizes[-1]} | "
        f"median: {sizes[len(sizes)//2]}"
    )

    # ── Step 3: Synthesize each cluster ─────────────────────────────────────
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("No API key. Set ANTHROPIC_API_KEY or pass --api-key.")
        sys.exit(1)

    synthesizer = ClaimSynthesizer(api_key=api_key, model=args.model)
    stats = {"success": 0, "skipped": 0, "failed": 0, "single_paper": 0}

    for cluster_id, cluster_claims_list in tqdm(
        sorted(cluster_claims.items()), desc="Synthesizing clusters"
    ):
        out_path = out_dir / f"{cluster_id}.json"
        if args.resume and out_path.exists():
            stats["skipped"] += 1
            continue

        # Skip single-paper clusters (no cross-paper synthesis possible)
        n_papers = len({c.paper_id for c in cluster_claims_list})
        if n_papers < 2:
            stats["single_paper"] += 1
            continue

        synthesis = synthesizer.synthesize(cluster_id, cluster_claims_list)
        if synthesis is None:
            stats["failed"] += 1
            continue

        with open(out_path, "w") as f:
            json.dump(synthesis.to_dict(), f, indent=2)
        stats["success"] += 1

        if args.delay > 0:
            time.sleep(args.delay)

    logger.info(
        f"Done. Success: {stats['success']}, Skipped: {stats['skipped']}, "
        f"Single-paper: {stats['single_paper']}, Failed: {stats['failed']}"
    )

    # Print a few examples of emergent hypotheses
    _print_sample_hypotheses(out_dir)


def _print_sample_hypotheses(out_dir: Path):
    """Print emergent hypotheses from a few clusters as a sanity check."""
    files = sorted(out_dir.glob("[0-9]*.json"))[:5]
    if not files:
        return
    print("\n" + "="*60)
    print("SAMPLE EMERGENT HYPOTHESES")
    print("="*60)
    for fpath in files:
        try:
            with open(fpath) as f:
                s = json.load(f)
            print(f"\nCluster {s['cluster_id']} ({s['n_papers']} papers, "
                  f"{s['n_claims']} claims) [{', '.join(s.get('collections_spanned', [])[:3])}]")
            print(f"Consensus: {s.get('consensus', '')[:120]}...")
            for h in s.get("emergent_hypotheses", []):
                print(f"  * {h}")
        except Exception:
            pass


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/04_cluster_and_meta.log", level="DEBUG", rotation="50 MB")
    Path("logs").mkdir(exist_ok=True)
    main()
