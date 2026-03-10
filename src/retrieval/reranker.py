"""
Cross-encoder reranking of retrieved chunks.

Uses ms-marco-MiniLM to score (query, chunk) pairs and return top-k.

Usage:
    reranker = Reranker()
    top_chunks = reranker.rerank(query, chunks, top_k=8)
"""

from loguru import logger
from sentence_transformers import CrossEncoder

from src.retrieval.retriever import RetrievedChunk

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL, device: str = None):
        logger.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """
        Score each (query, chunk.text) pair and return top_k chunks by score.
        """
        if not chunks:
            return []

        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Sort by cross-encoder score descending
        scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # Update .score field to reflect reranker score
        result = []
        for score, chunk in top:
            chunk.score = float(score)
            result.append(chunk)

        return result
