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
from pathlib import Path
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
        top_k_reranked: int = 12,
        reviews_dir: str = "data/reviews",
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.backend = backend
        self.top_k_dense = top_k_dense
        self.top_k_reranked = top_k_reranked
        self.reviews_dir = Path(reviews_dir)
        self._quality_scores: Optional[dict[int, int]] = None  # loaded lazily

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
        # 1. Retrieve — use more chunks for litreview
        dense_k = self.top_k_dense * 2 if mode == "litreview" else self.top_k_dense
        rerank_k = max(20, self.top_k_reranked) if mode == "litreview" else self.top_k_reranked
        chunks = self.retriever.retrieve(question, top_k=dense_k, filters=filters)
        if not chunks:
            return RAGResult(
                answer="No relevant papers found for this query.",
                sources=[],
                query=question,
            )

        # 2. Rerank
        top_chunks = self.reranker.rerank(question, chunks, top_k=rerank_k)

        # 2b. In litreview mode, boost scores by paper quality and re-sort
        if mode == "litreview":
            top_chunks = self._apply_quality_boost(top_chunks)

        # 3. Assemble context
        context = self._format_context(top_chunks)
        sources = self._extract_sources(top_chunks)

        # 4. Generate
        prompt = self._build_prompt(question, context, mode)
        answer = self._generate(prompt)

        return RAGResult(answer=answer, sources=sources, query=question)

    def _load_quality_scores(self) -> dict[int, int]:
        """Load {paper_id: quality_score} from data/reviews/ lazily."""
        if self._quality_scores is not None:
            return self._quality_scores
        scores = {}
        if self.reviews_dir.exists():
            for fpath in self.reviews_dir.glob("*.json"):
                try:
                    with open(fpath) as f:
                        r = json.load(f)
                    pid = r.get("paper_id")
                    score = r.get("quality_score")
                    if pid is not None and score is not None:
                        scores[int(pid)] = int(score)
                except Exception:
                    pass
        self._quality_scores = scores
        logger.info(f"Loaded quality scores for {len(scores)} papers")
        return scores

    def _apply_quality_boost(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Re-score chunks by multiplying rerank score by a quality factor.
        Quality 3 (average) = neutral (×1.0). Scale: score/3.
        Papers with no review are treated as average (score 3).
        """
        scores = self._load_quality_scores()
        if not scores:
            return chunks
        boosted = []
        for c in chunks:
            quality = scores.get(c.paper_id, 3)
            factor = quality / 3.0
            boosted.append(RetrievedChunk(
                paper_id=c.paper_id,
                title=c.title,
                year=c.year,
                journal=c.journal,
                section=c.section,
                text=c.text,
                score=c.score * factor,
                chunk_index=c.chunk_index,
            ))
        boosted.sort(key=lambda c: c.score, reverse=True)
        return boosted

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            header = f"[{i}] {c.title} ({c.year}) — {c.journal} [{c.section}]"
            parts.append(f"{header}\n{c.text}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        seen = {}
        for c in chunks:
            if c.paper_id not in seen or c.score > seen[c.paper_id]["score"]:
                seen[c.paper_id] = {
                    "paper_id": c.paper_id,
                    "title": c.title,
                    "year": c.year,
                    "journal": c.journal,
                    "section": c.section,
                    "score": round(c.score, 3),
                }
        return list(seen.values())

    def _build_prompt(self, question: str, context: str, mode: str) -> str:
        if mode == "contradiction":
            question = (
                f"{question}\n\nPay special attention to conflicting findings across papers. "
                "Explicitly note where papers disagree and what might explain the discrepancy."
            )
        elif mode == "litreview":
            question = (
                f"Write an academic literature review (300-500 words) addressing: "
                f"{question}\nUse academic prose, cite papers frequently as (Author, Year), "
                "and aim to reference as many of the provided papers as appropriate."
            )

        return QA_PROMPT_TEMPLATE.format(context=context, question=question)

    def _generate(self, prompt: str) -> str:
        if self.backend == "claude":
            return self._generate_claude(prompt)
        elif self.backend == "mlx":
            return self._generate_mlx(prompt)

    def _generate_claude(self, prompt: str, max_tokens: int = 2048) -> str:
        response = self.claude.messages.create(
            model=self.claude_model,
            max_tokens=max_tokens,
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
