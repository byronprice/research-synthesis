"""
Chunk papers and load them into a Qdrant vector database.

Chunking strategy (hierarchical, section-aware):
  Level 0: abstract only (1 chunk per paper) — coarse retrieval
  Level 1: per-section chunks (introduction, methods, results, discussion)
  Level 2: paragraph-level sub-chunks for sections that exceed max tokens

Each chunk carries full paper metadata for filtered retrieval.

Usage:
    indexer = Indexer(qdrant_path="data/qdrant_db")
    indexer.index_papers(papers_metadata, papers_fulltext, embedder)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    PointStruct,
    VectorParams,
)

from src.embedding.embedder import VECTOR_DIM, Embedder


COLLECTION_NAME = "papers_dense"

# Approximate chars per token (rough but good enough for chunking decisions)
CHARS_PER_TOKEN = 4

# Chunking thresholds (in characters, derived from token targets × CHARS_PER_TOKEN)
SECTION_MAX_CHARS = 900 * CHARS_PER_TOKEN      # 3600 chars → split into paragraphs
PARAGRAPH_TARGET_CHARS = 300 * CHARS_PER_TOKEN  # 1200 chars per paragraph chunk
PARAGRAPH_OVERLAP_CHARS = 50 * CHARS_PER_TOKEN  # 200 chars overlap


@dataclass
class Chunk:
    chunk_id: str          # "{paper_id}_{level}_{section}_{index}"
    paper_id: int
    title: str
    year: str
    journal: str
    collections: list[str]
    section: str           # "abstract", "introduction", "methods", etc. or "full_text"
    chunk_level: int       # 0=abstract, 1=section, 2=paragraph
    chunk_index: int
    text: str
    vector: Optional[list[float]] = None

    def to_payload(self) -> dict:
        """Qdrant payload (metadata stored alongside the vector)."""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "collections": self.collections,
            "section": self.section,
            "chunk_level": self.chunk_level,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


class Indexer:
    def __init__(self, qdrant_path: str = "data/qdrant_db"):
        self.client = QdrantClient(path=qdrant_path)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ),
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists")

    def index_papers(
        self,
        metadata_path: str,
        fulltext_path: str,
        embedder: Embedder,
        batch_size: int = 64,
        skip_existing: bool = True,
    ):
        """
        Load metadata + fulltext JSONL, generate chunks, embed, and upsert into Qdrant.
        """
        # Load metadata keyed by paper_id
        meta: dict[int, dict] = {}
        with open(metadata_path) as f:
            for line in f:
                rec = json.loads(line)
                meta[rec["paper_id"]] = rec

        # Get already-indexed paper IDs if resuming
        existing_ids: set[int] = set()
        if skip_existing:
            # Scroll through collection to find indexed paper IDs
            # For large collections use scroll API
            try:
                result = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=10000,
                    with_payload=["paper_id"],
                    with_vectors=False,
                )
                for point in result[0]:
                    existing_ids.add(point.payload["paper_id"])
                logger.info(f"Found {len(existing_ids)} already-indexed papers")
            except Exception:
                pass

        # Process fulltext records
        chunks_buffer: list[Chunk] = []
        total_indexed = 0
        papers_done = 0
        total_papers = sum(1 for _ in open(fulltext_path)) - len(existing_ids)

        with open(fulltext_path) as f:
            for line in f:
                rec = json.loads(line)
                pid = rec["paper_id"]

                if pid in existing_ids:
                    continue
                if rec.get("error") and not rec.get("full_text"):
                    papers_done += 1
                    continue

                m = meta.get(pid, {})
                paper_chunks = self._chunk_paper(rec, m)
                chunks_buffer.extend(paper_chunks)
                papers_done += 1

                # Flush when buffer reaches the embedder's batch size.
                if len(chunks_buffer) >= batch_size:
                    total_indexed += self._embed_and_upsert(chunks_buffer, embedder)
                    chunks_buffer = []
                    if total_indexed % 200 == 0 or papers_done % 100 == 0:
                        logger.info(
                            f"Progress: {papers_done}/{total_papers} papers, "
                            f"{total_indexed} chunks indexed"
                        )

        if chunks_buffer:
            total_indexed += self._embed_and_upsert(chunks_buffer, embedder)

        logger.info(f"Indexing complete. Total chunks upserted: {total_indexed}")

    def _embed_and_upsert(self, chunks: list[Chunk], embedder: Embedder) -> int:
        texts = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)

        points = []
        for chunk, vec in zip(chunks, vectors):
            # Use a deterministic integer ID derived from chunk_id string
            point_id = abs(hash(chunk.chunk_id)) % (2**63)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload=chunk.to_payload(),
                )
            )

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    def _chunk_paper(self, fulltext_rec: dict, meta: dict) -> list[Chunk]:
        """Generate multi-level chunks for one paper."""
        pid = fulltext_rec["paper_id"]
        title = meta.get("title", "")
        year = meta.get("year", "")
        journal = meta.get("journal", "")
        collections = meta.get("collections", [])
        abstract = meta.get("abstract", "")
        sections: dict = fulltext_rec.get("sections", {})

        chunks: list[Chunk] = []

        # Level 0: abstract chunk
        abstract_text = sections.get("abstract", abstract).strip()
        if not abstract_text and abstract:
            abstract_text = abstract
        if abstract_text:
            chunks.append(Chunk(
                chunk_id=f"{pid}_0_abstract_0",
                paper_id=pid,
                title=title, year=year, journal=journal, collections=collections,
                section="abstract", chunk_level=0, chunk_index=0,
                text=f"Title: {title}\nAbstract: {abstract_text}",
            ))

        # Level 1 + 2: sections
        priority_sections = [
            "introduction", "background", "methods", "results",
            "discussion", "conclusion", "related_work",
        ]
        for section_name in priority_sections:
            text = sections.get(section_name, "").strip()
            if not text:
                continue

            if len(text) <= SECTION_MAX_CHARS:
                # Level 1: whole section fits
                chunks.append(Chunk(
                    chunk_id=f"{pid}_1_{section_name}_0",
                    paper_id=pid,
                    title=title, year=year, journal=journal, collections=collections,
                    section=section_name, chunk_level=1, chunk_index=0,
                    text=text,
                ))
            else:
                # Level 2: split into paragraph chunks with overlap
                paragraphs = _split_paragraphs(text, PARAGRAPH_TARGET_CHARS, PARAGRAPH_OVERLAP_CHARS)
                for i, para in enumerate(paragraphs):
                    chunks.append(Chunk(
                        chunk_id=f"{pid}_2_{section_name}_{i}",
                        paper_id=pid,
                        title=title, year=year, journal=journal, collections=collections,
                        section=section_name, chunk_level=2, chunk_index=i,
                        text=para,
                    ))

        # Fallback: if no sections detected, chunk full_text
        if len(chunks) <= 1:
            full_text = fulltext_rec.get("full_text", "").strip()
            if full_text:
                paragraphs = _split_paragraphs(full_text, PARAGRAPH_TARGET_CHARS, PARAGRAPH_OVERLAP_CHARS)
                for i, para in enumerate(paragraphs):
                    chunks.append(Chunk(
                        chunk_id=f"{pid}_2_full_text_{i}",
                        paper_id=pid,
                        title=title, year=year, journal=journal, collections=collections,
                        section="full_text", chunk_level=2, chunk_index=i,
                        text=para,
                    ))

        return chunks


def _split_paragraphs(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """
    Split text into chunks of approximately target_chars with overlap.

    PDF-extracted text often uses single newlines for line wrapping with no
    double newlines, so we try double-newline splits first and fall back to
    single-newline splits (grouping lines into target-sized chunks).
    """
    # Try paragraph-level split first
    paragraphs = re.split(r"\n\n+", text)
    if len(paragraphs) == 1:
        # No double newlines — split on single newlines (individual wrapped lines)
        paragraphs = text.split("\n")

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if not current:
            current = para
        elif len(current) + 1 + len(para) <= target_chars:
            current = current + " " + para
        else:
            chunks.append(current)
            overlap = current[-overlap_chars:] if len(current) > overlap_chars else current
            current = (overlap + " " + para).strip() if overlap else para

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:target_chars]]
