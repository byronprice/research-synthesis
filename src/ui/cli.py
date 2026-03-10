#!/usr/bin/env python3
"""
Command-line interface for querying the research synthesis system.

Run from project root:
  python src/ui/cli.py "What is theta-gamma coupling?"
  python src/ui/cli.py --mode contradiction "Is dopamine reward prediction error?"
  python src/ui/cli.py --mode litreview "olfactory coding" --collection Olfaction
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from src.embedding.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.inference.rag_chain import RAGChain


def parse_args():
    p = argparse.ArgumentParser(description="Query the neuroscience literature corpus")
    p.add_argument("query", nargs="?", help="Your question")
    p.add_argument("--mode", choices=["qa", "contradiction", "litreview"], default="qa")
    p.add_argument("--collection", help="Filter to a Zotero collection")
    p.add_argument("--year-min", type=int)
    p.add_argument("--year-max", type=int)
    p.add_argument("--top-k", type=int, default=8, help="Number of chunks after reranking")
    p.add_argument("--qdrant-path", default="data/qdrant_db")
    p.add_argument("--backend", choices=["claude", "mlx"], default="claude")
    p.add_argument("--model-path", default="outputs/llama-3.1-8b-neuro-merged",
                   help="Path to merged MLX model (only for --backend mlx)")
    p.add_argument("--json", action="store_true", help="Output result as JSON")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.query:
        # Interactive mode
        print("Neuroscience Literature Assistant (Ctrl+C to exit)")
        print("Type your question:")
        args.query = input("> ").strip()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.backend == "claude" and not api_key:
        logger.error("Set ANTHROPIC_API_KEY for Claude backend")
        sys.exit(1)

    embedder = Embedder()
    retriever = Retriever(qdrant_path=args.qdrant_path, embedder=embedder)
    reranker = Reranker()
    chain = RAGChain(
        retriever=retriever,
        reranker=reranker,
        backend=args.backend,
        api_key=api_key,
        model_path=args.model_path,
        top_k_reranked=args.top_k,
    )

    filters = {}
    if args.collection:
        filters["collections"] = args.collection
    if args.year_min:
        filters["year_min"] = args.year_min
    if args.year_max:
        filters["year_max"] = args.year_max

    result = chain.query(args.query, filters=filters, mode=args.mode)

    if args.json:
        print(json.dumps({
            "answer": result.answer,
            "sources": result.sources,
            "query": result.query,
        }, indent=2))
    else:
        print("\n" + "="*60)
        print(result.answer)
        print("\n" + "-"*40)
        print("Sources:")
        for i, s in enumerate(result.sources, 1):
            print(f"  {i}. {s['title']} ({s['year']}) [{s['section']}]")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    main()
