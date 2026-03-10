"""
RAG chain: retrieve → rerank → assemble context → generate answer.

Supports two backends:
  1. Claude API (for development and when no local model is loaded)
  2. Local MLX model (for M1 deployment, fast inference)

Usage:
    chain = RAGChain(
        retriever=retriever,
        reranker=reranker,
        backend="claude",       # or "mlx"
        api_key="...",          # only needed for "claude" backend
        model_path="...",       # only needed for "mlx" backend
    )
    result = chain.query("What is the evidence for theta-gamma coupling?")
    print(result.answer)
    print(result.sources)
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.retrieval.retriever import RetrievedChunk, Retriever
from src.retrieval.reranker import Reranker


SYSTEM_PROMPT = """\
You are a neuroscience research assistant with access to a curated library \
of research papers. Answer questions based on the provided paper excerpts. \
Always cite the specific papers you draw from using (Author/Title, Year) format. \
If the provided context does not contain enough information to answer, say so \
rather than speculating."""

QA_PROMPT_TEMPLATE = """\
{context}

---
QUESTION: {question}

Answer based only on the papers above. Cite sources inline."""


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]    # list of {title, year, journal, section, score}
    query: str


class RAGChain:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        backend: str = "claude",
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        claude_model: str = "claude-haiku-4-5-20251001",
        top_k_dense: int = 40,
        top_k_reranked: int = 8,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.backend = backend
        self.top_k_dense = top_k_dense
        self.top_k_reranked = top_k_reranked

        if backend == "claude":
            import anthropic
            self.claude = anthropic.Anthropic(api_key=api_key)
            self.claude_model = claude_model
        elif backend == "mlx":
            self._load_mlx_model(model_path)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'claude' or 'mlx'")

    def query(
        self,
        question: str,
        filters: Optional[dict] = None,
        mode: str = "qa",  # qa | contradiction | litreview
    ) -> RAGResult:
        """
        Full RAG pipeline: retrieve → rerank → generate.

        mode:
          qa           — standard question answering
          contradiction — emphasize conflicting findings in the answer
          litreview    — generate an academic literature review paragraph
        """
        # 1. Retrieve
        chunks = self.retriever.retrieve(question, top_k=self.top_k_dense, filters=filters)
        if not chunks:
            return RAGResult(
                answer="No relevant papers found for this query.",
                sources=[],
                query=question,
            )

        # 2. Rerank
        top_chunks = self.reranker.rerank(question, chunks, top_k=self.top_k_reranked)

        # 3. Assemble context
        context = self._format_context(top_chunks)
        sources = self._extract_sources(top_chunks)

        # 4. Generate
        prompt = self._build_prompt(question, context, mode)
        answer = self._generate(prompt)

        return RAGResult(answer=answer, sources=sources, query=question)

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            header = f"[{i}] {c.title} ({c.year}) — {c.journal} [{c.section}]"
            parts.append(f"{header}\n{c.text}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        seen = set()
        sources = []
        for c in chunks:
            key = (c.paper_id, c.section)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "paper_id": c.paper_id,
                    "title": c.title,
                    "year": c.year,
                    "journal": c.journal,
                    "section": c.section,
                    "score": round(c.score, 3),
                })
        return sources

    def _build_prompt(self, question: str, context: str, mode: str) -> str:
        if mode == "contradiction":
            question = (
                f"{question}\n\nPay special attention to conflicting findings across papers. "
                "Explicitly note where papers disagree and what might explain the discrepancy."
            )
        elif mode == "litreview":
            question = (
                f"Write a concise academic literature review paragraph (200-300 words) addressing: "
                f"{question}\nUse academic prose and cite papers as (Author, Year)."
            )

        return QA_PROMPT_TEMPLATE.format(context=context, question=question)

    def _generate(self, prompt: str) -> str:
        if self.backend == "claude":
            return self._generate_claude(prompt)
        elif self.backend == "mlx":
            return self._generate_mlx(prompt)

    def _generate_claude(self, prompt: str) -> str:
        response = self.claude.messages.create(
            model=self.claude_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _generate_mlx(self, prompt: str) -> str:
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        result = self._mlx_generate(full_prompt, max_tokens=1024)
        return result

    def _load_mlx_model(self, model_path: str):
        try:
            from mlx_lm import load, generate
            self._mlx_model, self._mlx_tokenizer = load(model_path)
            self._mlx_gen_fn = generate
            logger.info(f"Loaded MLX model from {model_path}")
        except ImportError:
            raise ImportError("Install mlx-lm: pip install mlx-lm")

    def _mlx_generate(self, prompt: str, max_tokens: int = 1024) -> str:
        from mlx_lm import generate
        return generate(
            self._mlx_model,
            self._mlx_tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
