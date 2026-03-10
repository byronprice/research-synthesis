#!/usr/bin/env python3
"""
Phase 2: Embed paper chunks and load into Qdrant.

Reads:
  data/extracted/papers_metadata.jsonl
  data/extracted/papers_fulltext.jsonl

Writes:
  data/qdrant_db/  (Qdrant embedded database)

Run from project root:
  python scripts/02_build_index.py [--resume]
"""

import argparse
import os
import sys
from pathlib import Path

# Enable MPS fallback for ops not supported natively on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.embedder import Embedder
from src.embedding.indexer import Indexer


def parse_args():
    p = argparse.ArgumentParser(description="Embed papers and build Qdrant index")
    p.add_argument("--config", default="configs/retrieval_config.yaml")
    p.add_argument("--metadata", default="data/extracted/papers_metadata.jsonl")
    p.add_argument("--fulltext", default="data/extracted/papers_fulltext.jsonl")
    p.add_argument("--qdrant-path", default="data/qdrant_db")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Embedding batch size (lower = less RAM; default 8 for M1)")
    p.add_argument("--resume", action="store_true", help="Skip already-indexed papers")
    p.add_argument("--device", default="cpu",
                   help="Device for embedding model: cpu | cuda | mps (default: cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.metadata).exists():
        logger.error(f"Metadata file not found: {args.metadata}. Run 01_extract_all.py first.")
        sys.exit(1)
    if not Path(args.fulltext).exists():
        logger.error(f"Fulltext file not found: {args.fulltext}. Run 01_extract_all.py first.")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    emb_config = config["embedding"]
    embedder = Embedder(
        model_name=emb_config["model"],
        batch_size=args.batch_size,   # --batch-size controls embedding batch size
        device=args.device,
    )

    indexer = Indexer(qdrant_path=args.qdrant_path)
    indexer.index_papers(
        metadata_path=args.metadata,
        fulltext_path=args.fulltext,
        embedder=embedder,
        batch_size=args.batch_size,
        skip_existing=args.resume,
    )

    # Quick sanity check — reuse the indexer's open client to avoid lock conflict
    info = indexer.client.get_collection("papers_dense")
    logger.info(f"Collection 'papers_dense' has {info.points_count} points")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/02_build_index.log", level="DEBUG", rotation="50 MB")
    Path("logs").mkdir(exist_ok=True)
    main()
