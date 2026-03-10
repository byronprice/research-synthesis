#!/usr/bin/env python3
"""
Phase 4: Cluster paper summaries and generate cluster meta-summaries.

Reads:  data/summaries/*.json
Writes: data/clusters/cluster_assignments.json
        data/clusters/meta_summaries/cluster_{N}.json

Run from project root:
  python scripts/04_cluster_and_meta.py
"""

import json
import os
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.embedder import Embedder
from src.summarization.clustering import PaperClusterer
from src.summarization.meta_summary import MetaSummarizer


def main():
    summaries_dir = "data/summaries"
    out_dir = Path("data/clusters")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_summaries = len(list(Path(summaries_dir).glob("*.json")))
    if n_summaries == 0:
        logger.error("No summaries found. Run 03_summarize_papers.py first.")
        sys.exit(1)
    logger.info(f"Found {n_summaries} paper summaries")

    embedder = Embedder()

    # Cluster
    clusterer = PaperClusterer(embedder)
    assignments = clusterer.cluster(summaries_dir)

    assignments_path = out_dir / "cluster_assignments.json"
    with open(assignments_path, "w") as f:
        json.dump(assignments, f, indent=2)
    logger.info(f"Cluster assignments → {assignments_path}")

    # Meta-summaries
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Set ANTHROPIC_API_KEY to generate meta-summaries")
        sys.exit(1)

    meta_summarizer = MetaSummarizer(api_key=api_key)
    meta_summarizer.summarize_clusters(
        summaries_dir=summaries_dir,
        assignments={int(k): v for k, v in assignments.items()},
        output_dir=str(out_dir / "meta_summaries"),
    )
    logger.info("Done.")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/04_cluster_and_meta.log", level="DEBUG", rotation="50 MB")
    Path("logs").mkdir(exist_ok=True)
    main()
