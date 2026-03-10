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
from src.inference.rag_chain import RAGChain

# ── Initialize components once at startup ────────────────────────────────────

QDRANT_PATH = "data/qdrant_db"
BACKEND = os.environ.get("RAG_BACKEND", "claude")  # "claude" or "mlx"
MODEL_PATH = os.environ.get("MLX_MODEL_PATH", "outputs/llama-3.1-8b-neuro-merged")
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

_embedder = None
_chain = None


def get_chain() -> RAGChain:
    global _embedder, _chain
    if _chain is None:
        logger.info("Initializing RAG chain...")
        _embedder = Embedder()
        retriever = Retriever(qdrant_path=QDRANT_PATH, embedder=_embedder)
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
**Modes**: prefix with `?contra` for contradiction analysis, `?litreview` for a review paragraph.
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
    chain = get_chain()
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

    # Parse [collection: X] filter
    import re
    col_match = re.search(r"\[collection:\s*([^\]]+)\]", query, re.IGNORECASE)
    if col_match:
        filters["collections"] = col_match.group(1).strip()
        query = re.sub(r"\[collection:[^\]]+\]", "", query).strip()

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
