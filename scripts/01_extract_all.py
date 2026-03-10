#!/usr/bin/env python3
"""
Phase 1: Extract text from all Zotero PDFs and write JSONL output files.

Outputs:
  data/extracted/papers_metadata.jsonl  — one JSON record per paper (Zotero metadata)
  data/extracted/papers_fulltext.jsonl  — one JSON record per paper (extracted sections)

Run from project root:
  python scripts/01_extract_all.py [--limit N] [--workers N] [--resume]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pdf_extractor import ExtractorConfig, PDFExtractor
from src.extraction.zotero_reader import PaperRecord, ZoteroReader


def parse_args():
    p = argparse.ArgumentParser(description="Extract text from Zotero PDFs")
    p.add_argument("--config", default="configs/extraction_config.yaml")
    p.add_argument("--limit", type=int, default=None, help="Process only first N papers (for testing)")
    p.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    p.add_argument("--resume", action="store_true", help="Skip papers already in output files")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_worker(args: tuple) -> dict:
    """Worker function for multiprocessing (must be top-level for pickling)."""
    paper_dict, extractor_config_dict = args
    extractor = PDFExtractor(ExtractorConfig(**extractor_config_dict))
    paper = PaperRecord(**{k: v for k, v in paper_dict.items() if k != "pdf_path"}, pdf_path=Path(paper_dict["pdf_path"]))
    result = extractor.extract(paper.paper_id, paper.pdf_path)
    return result.to_dict()


def main():
    args = parse_args()
    config = load_config(args.config)

    out_dir = Path("data/extracted")
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "papers_metadata.jsonl"
    fulltext_path = out_dir / "papers_fulltext.jsonl"

    # Set up Zotero reader
    reader = ZoteroReader(
        db_path=config["zotero"]["db_path"],
        storage_path=config["zotero"]["storage_path"],
    )
    papers = reader.get_all_papers()
    logger.info(f"Total papers in Zotero with PDFs: {len(papers)}")

    if args.limit:
        papers = papers[: args.limit]
        logger.info(f"Limited to {args.limit} papers")

    # Resume: load already-processed paper IDs from fulltext output
    done_ids: set[int] = set()
    if args.resume and fulltext_path.exists():
        with open(fulltext_path) as f:
            for line in f:
                record = json.loads(line)
                done_ids.add(record["paper_id"])
        logger.info(f"Resuming: {len(done_ids)} papers already extracted")

    papers_to_process = [p for p in papers if p.paper_id not in done_ids]
    logger.info(f"Papers to extract: {len(papers_to_process)}")

    # Write metadata for ALL papers (including already-done ones) if starting fresh
    if not args.resume:
        with open(metadata_path, "w") as f:
            for paper in papers:
                f.write(json.dumps(paper.to_dict()) + "\n")
        logger.info(f"Wrote metadata for {len(papers)} papers → {metadata_path}")
    else:
        # Append metadata for new papers only
        with open(metadata_path, "a") as f:
            for paper in papers_to_process:
                f.write(json.dumps(paper.to_dict()) + "\n")

    if not papers_to_process:
        logger.info("Nothing to extract. Done.")
        return

    # Extraction config dict for worker processes
    ec = config["extraction"]
    extractor_config_dict = {
        "timeout_seconds": ec.get("timeout_seconds", 60),
        "min_text_chars": ec.get("min_text_chars", 500),
        "max_section_chars": ec.get("max_section_chars", 8000),
        "max_fulltext_chars": ec.get("max_fulltext_chars", 40000),
        "strip_references": ec.get("strip_references", True),
    }

    # Prepare worker args
    worker_args = [(p.to_dict(), extractor_config_dict) for p in papers_to_process]

    # Stats
    stats = {"success": 0, "timeout": 0, "error": 0, "too_short": 0}

    mode = "a" if args.resume else "w"
    with open(fulltext_path, mode) as out_f:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_extract_worker, arg): arg for arg in worker_args}
                for future in tqdm(as_completed(futures), total=len(worker_args), desc="Extracting PDFs"):
                    try:
                        result = future.result()
                        out_f.write(json.dumps(result) + "\n")
                        out_f.flush()
                        _update_stats(stats, result)
                    except Exception as exc:
                        logger.error(f"Worker exception: {exc}")
                        stats["error"] += 1
        else:
            # Single-process (easier to debug)
            extractor = PDFExtractor(ExtractorConfig(**extractor_config_dict))
            for paper in tqdm(papers_to_process, desc="Extracting PDFs"):
                result = extractor.extract(paper.paper_id, paper.pdf_path)
                result_dict = result.to_dict()
                out_f.write(json.dumps(result_dict) + "\n")
                out_f.flush()
                _update_stats(stats, result_dict)

    logger.info(
        f"Extraction complete.\n"
        f"  Success:   {stats['success']}\n"
        f"  Too short: {stats['too_short']}\n"
        f"  Timeout:   {stats['timeout']}\n"
        f"  Error:     {stats['error']}\n"
        f"Output: {fulltext_path}"
    )


def _update_stats(stats: dict, result: dict):
    error = result.get("error")
    if error == "timeout":
        stats["timeout"] += 1
    elif error == "too_short":
        stats["too_short"] += 1
    elif error:
        stats["error"] += 1
    else:
        stats["success"] += 1


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/01_extract_all.log", level="DEBUG", rotation="50 MB")
    Path("logs").mkdir(exist_ok=True)
    main()
