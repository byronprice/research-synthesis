#!/usr/bin/env bash
# Ingest new papers from Zotero into the research synthesis system.
# Runs phases 1-3 with --resume so only new papers are processed.
# Run from project root with conda env active:
#   conda activate neuro-synthesis
#   bash scripts/ingest_new_papers.sh

set -e

cd "$(dirname "$0")/.."

echo "=== Phase 1: Extracting new PDFs ==="
python scripts/01_extract_all.py --workers 4 --resume

echo "=== Phase 2: Indexing new chunks ==="
python scripts/02_build_index.py --device mps --batch-size 4 --resume

echo "=== Phase 3: Summarizing new papers ==="
python scripts/03_summarize_papers.py --resume

echo "=== Done. New papers are now searchable. ==="
echo "To update clustering and meta-summaries, run: python scripts/04_cluster_and_meta.py"
