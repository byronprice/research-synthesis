"""
Dense retrieval from Qdrant with optional metadata filtering.

Usage:
    retriever = Retriever(qdrant_path="data/qdrant_db", embedder=embedder)
    chunks = retriever.retrieve(
        query="what is the role of theta oscillations in memory?",
        top_k=40,
        filters={"collections": "Olfaction", "year_min": 2018},
    )
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from src.embedding.embedder import Embedder
from src.embedding.indexer import COLLECTION_NAME


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: int
    title: str
    year: str
    journal: str
    collections: list[str]
    section: str
    text: str
    score: float   # cosine similarity from Qdrant


class Retriever:
    def __init__(self, qdrant_path: str, embedder: Embedder):
        self.client = QdrantClient(path=qdrant_path)
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 40,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Embed query and retrieve top_k chunks.

        filters: optional dict with keys:
          collections: str or list[str]  — filter to specific Zotero collections
          year_min: int                  — earliest year
          year_max: int                  — latest year
          section: str                   — filter to specific section (e.g. "results")
          chunk_level: int               — 0=abstract, 1=section, 2=paragraph
        """
        query_vec = self.embedder.embed_query(query)
        qdrant_filter = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        chunks = []
        for r in results:
            p = r.payload
            chunks.append(RetrievedChunk(
                chunk_id=p.get("chunk_id", ""),
                paper_id=p.get("paper_id", 0),
                title=p.get("title", ""),
                year=p.get("year", ""),
                journal=p.get("journal", ""),
                collections=p.get("collections", []),
                section=p.get("section", ""),
                text=p.get("text", ""),
                score=r.score,
            ))
        return chunks

    def _build_filter(self, filters: dict) -> Filter:
        conditions = []

        if "collections" in filters:
            cols = filters["collections"]
            if isinstance(cols, str):
                cols = [cols]
            for col in cols:
                conditions.append(
                    FieldCondition(key="collections", match=MatchValue(value=col))
                )

        if "year_min" in filters or "year_max" in filters:
            conditions.append(FieldCondition(
                key="year",
                range=Range(
                    gte=str(filters.get("year_min", "0")),
                    lte=str(filters.get("year_max", "9999")),
                ),
            ))

        if "section" in filters:
            conditions.append(
                FieldCondition(key="section", match=MatchValue(value=filters["section"]))
            )

        if "chunk_level" in filters:
            conditions.append(
                FieldCondition(key="chunk_level", match=MatchValue(value=filters["chunk_level"]))
            )

        if not conditions:
            return None

        from qdrant_client.models import Filter as QFilter, Must
        return Filter(must=conditions)
