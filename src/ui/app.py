"""
Chainlit web UI for the research synthesis system.

Run from project root:
  chainlit run src/ui/app.py

Requires: pip install -r requirements/inference.txt
Set ANTHROPIC_API_KEY in environment.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chainlit as cl
from loguru import logger

from src.embedding.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.metadata_search import find_paper, get_authors, format_authors
from src.inference.rag_chain import RAGChain

# ── Initialize components once at startup ────────────────────────────────────

QDRANT_PATH = "data/qdrant_db"
BACKEND = os.environ.get("RAG_BACKEND", "claude")  # "claude" or "mlx"
MODEL_PATH = os.environ.get("MLX_MODEL_PATH", "outputs/llama-3.1-8b-neuro-merged")
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

_embedder = None
_retriever = None
_chain = None


def get_retriever() -> Retriever:
    global _embedder, _retriever
    if _retriever is None:
        _embedder = Embedder()
        _retriever = Retriever(qdrant_path=QDRANT_PATH, embedder=_embedder)
    return _retriever


def get_chain() -> RAGChain:
    global _chain
    if _chain is None:
        logger.info("Initializing RAG chain...")
        retriever = get_retriever()
        reranker = Reranker()
        _chain = RAGChain(
            retriever=retriever,
            reranker=reranker,
            backend=BACKEND,
            api_key=API_KEY,
            model_path=MODEL_PATH,
        )
        logger.info("RAG chain ready.")
    return _chain


# ── Chainlit handlers ─────────────────────────────────────────────────────────

WELCOME_MSG = """\
# Neuroscience Literature Assistant

Ask questions about your research paper library. Examples:

- *What is the evidence for theta-gamma coupling in memory consolidation?*
- *Are there contradictions about dopamine signaling during reward?*
- *Write a literature review on olfactory coding in piriform cortex*
- *What methods are used to study place cells?*

**Filter by collection**: add `[collection: Dopamine]` to your query.

**Modes** (default is Q&A):
- `?contra` — highlight contradictions across papers
- `?litreview` — generate an academic literature review paragraph
- `?summary` — retrieve the stored summary for a specific paper (e.g. `?summary Zheng 2026 representational drift`)
- `?review` — retrieve the critical review for a specific paper (e.g. `?review Bhatt 2020 place cells`)
"""


@cl.on_chat_start
async def on_start():
    await cl.Message(content=WELCOME_MSG).send()
    # Pre-warm the chain
    try:
        get_chain()
    except Exception as e:
        await cl.Message(content=f"⚠️ Initialization error: {e}").send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()

    # Parse mode prefix
    mode = "qa"
    filters = {}

    if query.startswith("?contra"):
        mode = "contradiction"
        query = query[len("?contra"):].strip()
    elif query.startswith("?litreview"):
        mode = "litreview"
        query = query[len("?litreview"):].strip()
    elif query.startswith("?summary"):
        mode = "summary"
        query = query[len("?summary"):].strip()
    elif query.startswith("?review"):
        mode = "review"
        query = query[len("?review"):].strip()

    # Parse [collection: X] filter
    import re
    col_match = re.search(r"\[collection:\s*([^\]]+)\]", query, re.IGNORECASE)
    if col_match:
        filters["collections"] = col_match.group(1).strip()
        query = re.sub(r"\[collection:[^\]]+\]", "", query).strip()

    if mode == "summary":
        await _handle_summary(query, filters)
        return

    if mode == "review":
        await _handle_review(query, filters)
        return

    chain = get_chain()

    # Show thinking indicator
    async with cl.Step(name="Retrieving relevant papers...") as step:
        try:
            result = chain.query(query, filters=filters, mode=mode)
            step.output = f"Found {len(result.sources)} relevant sources"
        except Exception as e:
            await cl.Message(content=f"Error: {e}").send()
            return

    # Format sources
    sources_text = ""
    if result.sources:
        lines = ["**Sources:**"]
        for i, s in enumerate(result.sources, 1):
            lines.append(f"{i}. *{s['title']}* ({s['year']}) — {s['journal']} [{s['section']}]")
        sources_text = "\n".join(lines)

    # Send answer
    await cl.Message(
        content=result.answer,
        elements=[
            cl.Text(name="Sources", content=sources_text, display="inline")
        ] if sources_text else [],
    ).send()


async def _handle_summary(query: str, filters: dict):
    # Try metadata search first (reliable for author/year queries)
    meta_hits = find_paper(query)
    paper_id = None
    title, year = "", ""
    if meta_hits:
        paper_id = meta_hits[0]["paper_id"]
        title = meta_hits[0].get("title", "")
        year = meta_hits[0].get("year", "")
    else:
        retriever = get_retriever()
        chunks = retriever.retrieve(query, top_k=10, filters={**filters, "chunk_level": 0})
        if not chunks:
            chunks = retriever.retrieve(query, top_k=10, filters=filters)
        if not chunks:
            await cl.Message(content="No matching paper found.").send()
            return
        paper_id = chunks[0].paper_id
        title = chunks[0].title
        year = chunks[0].year

    summary_path = Path(f"data/summaries/{paper_id}.json")
    if not summary_path.exists():
        await cl.Message(
            content=f"Found **{title}** ({year}) but no summary exists yet — run `03_summarize_papers.py` first."
        ).send()
        return

    with open(summary_path) as f:
        s = json.load(f)

    m = s.get("methods", {})
    findings = "\n".join(f"- {f}" for f in s.get("key_findings", []))
    claims = "\n".join(
        f"- **{c.get('direction', '?')}**: {c['claim']}" +
        (f" *({c['quantitative']})*" if c.get("quantitative") and not c["quantitative"].startswith("<") else "")
        for c in s.get("claims", [])
    )
    limitations = "\n".join(f"- {l}" for l in s.get("limitations", []))
    open_qs = "\n".join(f"- {q}" for q in s.get("open_questions", []))
    keywords = ", ".join(s.get("keywords", []))

    authors = format_authors(get_authors(paper_id))
    author_line = f"*{authors}*  \n" if authors else ""
    content = f"## {s.get('title', title)} ({s.get('year', year)})\n{author_line}*{s.get('journal', '')}*\n\n"
    if s.get("relevance"):
        content += f"### Relevance\n{s['relevance']}\n\n"
    content += f"### Key Findings\n{findings}\n\n"
    content += f"### Methods [{m.get('model_system', '')}]\n"
    if m.get("techniques"):
        content += f"**Techniques:** {', '.join(m['techniques'])}  \n"
    if m.get("analysis"):
        content += f"**Analysis:** {', '.join(m['analysis'])}\n"
    content += f"\n### Claims\n{claims}\n"
    if limitations:
        content += f"\n### Limitations\n{limitations}\n"
    if open_qs:
        content += f"\n### Open Questions\n{open_qs}\n"
    if keywords:
        content += f"\n**Keywords:** {keywords}"

    await cl.Message(content=content).send()


async def _handle_review(query: str, filters: dict):
    meta_hits = find_paper(query)
    paper_id = None
    title, year = "", ""
    if meta_hits:
        paper_id = meta_hits[0]["paper_id"]
        title = meta_hits[0].get("title", "")
        year = meta_hits[0].get("year", "")
    else:
        retriever = get_retriever()
        chunks = retriever.retrieve(query, top_k=10, filters={**filters, "chunk_level": 0})
        if not chunks:
            chunks = retriever.retrieve(query, top_k=10, filters=filters)
        if not chunks:
            await cl.Message(content="No matching paper found.").send()
            return
        paper_id = chunks[0].paper_id
        title = chunks[0].title
        year = chunks[0].year

    review_path = Path(f"data/reviews/{paper_id}.json")
    if not review_path.exists():
        await cl.Message(
            content=f"Found **{title}** ({year}) but no review exists yet — run `03_summarize_papers.py` first."
        ).send()
        return

    with open(review_path) as f:
        r = json.load(f)

    authors = format_authors(get_authors(paper_id))
    author_line = f"*{authors}*  \n" if authors else ""
    content = f"## {r.get('title', title)} ({r.get('year', year)})\n{author_line}*{r.get('journal', '')}*\n\n"
    if r.get("quality_score"):
        content += f"**Quality: {r['quality_score']}/5** — {r.get('quality_rationale', '')}\n\n"

    VERDICT_BADGE = {
        "justified": "Justified",
        "partially_justified": "Partially justified",
        "not_justified": "Not justified",
    }

    for i, c in enumerate(r.get("claims", []), 1):
        badge = VERDICT_BADGE.get(c.get("verdict", ""), c.get("verdict", ""))
        content += f"### Claim {i} — {badge}\n"
        content += f"**{c['claim']}**\n\n"
        content += f"**Ideal experiments:** {c['ideal_experiments']}\n\n"
        content += f"**What the paper does:** {c['paper_justification']}\n\n"
        if c.get("required_experiments"):
            content += f"**Additional experiments needed:** {c['required_experiments']}\n\n"

    if r.get("overall_assessment"):
        content += f"---\n### Overall Assessment\n{r['overall_assessment']}\n"

    await cl.Message(content=content).send()
