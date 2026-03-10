# Research Synthesis

A local pipeline for reading, summarizing, and synthesizing a large corpus of neuroscience PDFs using open-source LLMs. Combines fine-tuning (domain adaptation) with RAG (factual grounding) to produce structured per-paper summaries, cross-paper trend and contradiction analysis, and an interactive Q&A interface.

## What it does

- **Extracts** text from PDFs via Zotero, with section-aware chunking
- **Indexes** all chunks into a local Qdrant vector database using nomic-embed-text
- **Summarizes** each paper into structured JSON (key findings, methods, directional claims, limitations) via Claude API
- **Clusters** papers by topic using UMAP + HDBSCAN and generates cluster-level meta-summaries
- **Detects contradictions** between papers via claim similarity + LLM verification
- **Fine-tunes** Llama 3.1 8B on domain-specific instruction data using QLoRA (for free local inference)
- **Serves** an interactive Q&A interface via Chainlit with RAG, contradiction queries, and literature review generation

## Architecture

```
PDFs (Zotero)
    │
    ▼
1. Extraction: PyMuPDF → section-aware JSON
    │
    ▼
2. Indexing: nomic-embed-text → Qdrant vector DB
    │
    ├──► 3. Summarization: Claude API → per-paper structured JSON
    │         │
    │         ▼
    │    4. Clustering: UMAP + HDBSCAN → topic clusters + meta-summaries
    │         │
    │         └──► 5. Contradiction detection: claim similarity + LLM verify
    │
    └──► 6. Fine-tuning data: Claude API → instruction-response pairs
              → QLoRA on Llama 3.1 8B → merged model
                   │
                   ▼
              7. Interactive Q&A: Chainlit + RAG → fine-tuned model
```

## Setup

**Requirements:** Python 3.11, [Conda](https://docs.conda.io/), [Zotero](https://www.zotero.org/) with PDFs

```bash
conda create -n neuro-synthesis python=3.11
conda activate neuro-synthesis
pip install -r requirements/base.txt
```

For fine-tuning on a GPU cluster:
```bash
pip install -r requirements/training.txt
```

For local inference on Apple Silicon:
```bash
pip install -r requirements/inference.txt
```

**Apple Silicon note:** This project was developed on an M1 Pro with x86_64 Anaconda (Rosetta). The `requirements/base.txt` pins are set accordingly (`torch<=2.2.2`, `sentence-transformers<3`, `transformers<5`, `numpy<2`). On native ARM or Linux, remove the upper bounds.

## Configuration

Edit `configs/extraction_config.yaml` to point at your Zotero installation:

```yaml
zotero:
  db_path: "~/Zotero/zotero.sqlite"
  storage_path: "~/Zotero/storage"
```

Set your Anthropic API key (needed for phases 3, 4, 5, 6):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Running the pipeline

```bash
conda activate neuro-synthesis
cd research-synthesis

# Phase 1: Extract text from all PDFs (~2-4 hrs for 2500 papers)
python scripts/01_extract_all.py --workers 4 --resume

# Phase 2: Build Qdrant vector index (~30 min on Apple Silicon GPU)
python scripts/02_build_index.py --device mps --batch-size 4

# Phase 3: Summarize papers via Claude API (~$12 haiku / ~$80 sonnet)
python scripts/03_summarize_papers.py --resume --model claude-haiku-4-5-20251001

# Phase 4: Cluster papers and generate meta-summaries
python scripts/04_cluster_and_meta.py

# Phase 5: Find contradictions between papers
python scripts/05_find_contradictions.py

# Phase 6: Generate fine-tuning training data (~$15-20)
python scripts/06_generate_training_data.py

# Phase 7: Fine-tune on GPU cluster (submit SLURM job)
sbatch scripts/07_train.sh

# Phase 8: Merge LoRA adapter
python scripts/08_merge_adapter.py --adapter outputs/lora-adapter --output outputs/merged-model

# Run the web UI
chainlit run src/ui/app.py
```

All scripts support `--resume` to continue interrupted runs.

## Key technology choices

| Component | Choice | Rationale |
|---|---|---|
| Base model | Llama 3.1 8B Instruct | 128K context, Apache 2.0 |
| Fine-tuning | Unsloth + QLoRA + TRL | 2-3x faster than vanilla PEFT |
| PDF extraction | PyMuPDF | Fast, good two-column layout handling |
| Embeddings | nomic-embed-text-v1.5 | 8K context, 768-dim, Apache 2.0 |
| Vector DB | Qdrant (embedded) | Best filtering at 10K-50K vectors |
| Clustering | UMAP + HDBSCAN | No fixed K, handles arbitrary topology |
| Reranking | cross-encoder/ms-marco-MiniLM | Improves RAG precision |
| Local inference | MLX (Apple Silicon) | Faster than llama.cpp on M1 |
| UI | Chainlit | Minimal code, streaming + citations |

## Project structure

```
research-synthesis/
├── configs/               # YAML configs for extraction, retrieval, training
├── requirements/          # base.txt, training.txt (cluster), inference.txt (M1)
├── scripts/               # Numbered pipeline scripts (01-08)
└── src/
    ├── extraction/        # Zotero reader, PDF extractor
    ├── embedding/         # nomic-embed wrapper, Qdrant indexer
    ├── summarization/     # Per-paper summarizer, clustering, meta-summaries
    ├── contradiction/     # Candidate finder, LLM verifier
    ├── training/          # Training data generator, fine-tuning script
    ├── retrieval/         # Qdrant retriever, cross-encoder reranker
    ├── inference/         # RAG chain (Claude or MLX backend)
    └── ui/                # Chainlit web app, CLI
```

## Interactive Q&A

The Chainlit UI supports four query modes:

- Default: grounded Q&A with citations
- `?contra <query>` — contradiction analysis across papers
- `?litreview <query>` — generate a literature review paragraph
- `[collection: Olfaction]` — filter to a specific Zotero collection

The CLI (`src/ui/cli.py`) supports the same modes with `--mode` and `--collection` flags.

## Data

Extracted text, summaries, and the vector database are not included in this repository (copyright-sensitive, ~650MB). Run the pipeline from your own Zotero library to regenerate them.
