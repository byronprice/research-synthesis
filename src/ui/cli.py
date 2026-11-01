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
from src.retrieval.metadata_search import find_paper
from src.inference.rag_chain import RAGChain


def parse_args():
    p = argparse.ArgumentParser(description="Query the neuroscience literature corpus")
    p.add_argument("query", nargs="?", help="Your question")
    p.add_argument("--mode", choices=["qa", "contradiction", "litreview", "summary", "review"], default="qa")
    p.add_argument("--collection", help="Filter to a Zotero collection")
    p.add_argument("--year-min", type=int)
    p.add_argument("--year-max", type=int)
    p.add_argument("--top-k", type=int, default=12, help="Number of chunks after reranking")
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

    embedder = Embedder()
    retriever = Retriever(qdrant_path=args.qdrant_path, embedder=embedder)

    filters = {}
    if args.collection:
        filters["collections"] = args.collection
    if args.year_min:
        filters["year_min"] = args.year_min
    if args.year_max:
        filters["year_max"] = args.year_max

    if args.mode == "summary":
        _run_summary(args, retriever, filters)
        return

    if args.mode == "review":
        _run_review(args, retriever, filters)
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.backend == "claude" and not api_key:
        logger.error("Set ANTHROPIC_API_KEY for Claude backend")
        sys.exit(1)

    reranker = Reranker()
    chain = RAGChain(
        retriever=retriever,
        reranker=reranker,
        backend=args.backend,
        api_key=api_key,
        model_path=args.model_path,
        top_k_reranked=args.top_k,
    )

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


def _run_summary(args, retriever, filters):
    # Try metadata search first (reliable for author/year queries)
    meta_hits = find_paper(args.query)
    paper_id = None
    title, year = "", ""
    if meta_hits:
        paper_id = meta_hits[0]["paper_id"]
        title = meta_hits[0].get("title", "")
        year = meta_hits[0].get("year", "")
    else:
        # Fall back to vector search
        chunks = retriever.retrieve(args.query, top_k=10, filters={**filters, "chunk_level": 0})
        if not chunks:
            chunks = retriever.retrieve(args.query, top_k=10, filters=filters)
        if not chunks:
            print("No matching paper found.")
            return
        paper_id = chunks[0].paper_id
        title = chunks[0].title
        year = chunks[0].year

    summary_path = Path(f"data/summaries/{paper_id}.json")
    if not summary_path.exists():
        print(f"Found paper: {title} ({year})")
        print("No summary available yet — run 03_summarize_papers.py first.")
        return

    with open(summary_path) as f:
        s = json.load(f)

    if args.json:
        print(json.dumps(s, indent=2))
        return

    print("\n" + "="*60)
    print(f"{s.get('title', title)} ({s.get('year', year)})")
    print(f"{s.get('journal', '')}".strip())
    print("="*60)

    if s.get("relevance"):
        print("\nRELEVANCE")
        print(f"  {s['relevance']}")

    print("\nKEY FINDINGS")
    for finding in s.get("key_findings", []):
        print(f"  • {finding}")

    m = s.get("methods", {})
    print(f"\nMETHODS  [{m.get('model_system', '')}]")
    if m.get("techniques"):
        print(f"  Techniques: {', '.join(m['techniques'])}")
    if m.get("analysis"):
        print(f"  Analysis:   {', '.join(m['analysis'])}")

    print("\nCLAIMS")
    for c in s.get("claims", []):
        quant = f" ({c['quantitative']})" if c.get("quantitative") and not c["quantitative"].startswith("<") else ""
        print(f"  [{c.get('direction','?'):>13}]  {c['claim']}{quant}")

    if s.get("limitations"):
        print("\nLIMITATIONS")
        for lim in s["limitations"]:
            print(f"  • {lim}")

    if s.get("open_questions"):
        print("\nOPEN QUESTIONS")
        for q in s["open_questions"]:
            print(f"  • {q}")

    if s.get("keywords"):
        print(f"\nKeywords: {', '.join(s['keywords'])}")


def _run_review(args, retriever, filters):
    meta_hits = find_paper(args.query)
    paper_id = None
    title, year = "", ""
    if meta_hits:
        paper_id = meta_hits[0]["paper_id"]
        title = meta_hits[0].get("title", "")
        year = meta_hits[0].get("year", "")
    else:
        chunks = retriever.retrieve(args.query, top_k=10, filters={**filters, "chunk_level": 0})
        if not chunks:
            chunks = retriever.retrieve(args.query, top_k=10, filters=filters)
        if not chunks:
            print("No matching paper found.")
            return
        paper_id = chunks[0].paper_id
        title = chunks[0].title
        year = chunks[0].year

    review_path = Path(f"data/reviews/{paper_id}.json")
    if not review_path.exists():
        print(f"Found paper: {title} ({year})")
        print("No review available yet — run 03_summarize_papers.py first.")
        return

    with open(review_path) as f:
        r = json.load(f)

    if args.json:
        print(json.dumps(r, indent=2))
        return

    print("\n" + "="*60)
    print(f"{r.get('title', title)} ({r.get('year', year)})")
    print(f"{r.get('journal', '')}".strip())
    if r.get("quality_score"):
        print(f"Quality: {r['quality_score']}/5  —  {r.get('quality_rationale', '')}")
    print("="*60)

    VERDICT_LABEL = {
        "justified": "JUSTIFIED",
        "partially_justified": "PARTIAL",
        "not_justified": "NOT JUSTIFIED",
    }

    for i, c in enumerate(r.get("claims", []), 1):
        verdict = VERDICT_LABEL.get(c.get("verdict", ""), c.get("verdict", ""))
        print(f"\nCLAIM {i}  [{verdict}]")
        print(f"  {c['claim']}")
        print(f"\n  Ideal experiments:")
        print(f"    {c['ideal_experiments']}")
        print(f"\n  What the paper does:")
        print(f"    {c['paper_justification']}")
        if c.get("required_experiments"):
            print(f"\n  Additional experiments needed:")
            print(f"    {c['required_experiments']}")

    if r.get("overall_assessment"):
        print("\n" + "-"*40)
        print("OVERALL ASSESSMENT")
        print(f"  {r['overall_assessment']}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    main()
