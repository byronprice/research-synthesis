"""
Find candidate paper-pair contradictions by:
1. Extracting all directional claims from summaries
2. Finding claim pairs with high semantic similarity but opposing directions

Only contradiction candidates (not verified) are produced here.
Use verifier.py to LLM-verify each candidate.

Usage:
    finder = CandidateFinder(embedder)
    candidates = finder.find(summaries_dir="data/summaries", sim_threshold=0.85)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.embedding.embedder import Embedder


OPPOSING_DIRECTIONS = {
    frozenset({"increase", "decrease"}),
    frozenset({"increase", "no_effect"}),
    frozenset({"decrease", "no_effect"}),
}


@dataclass
class ClaimRecord:
    paper_id: int
    title: str
    year: str
    claim_text: str
    direction: str
    quantitative: str
    conditions: str


@dataclass
class ContraCandidate:
    claim_a: ClaimRecord
    claim_b: ClaimRecord
    similarity: float

    def to_dict(self) -> dict:
        return {
            "paper_a_id": self.claim_a.paper_id,
            "paper_a_title": self.claim_a.title,
            "paper_a_year": self.claim_a.year,
            "claim_a": self.claim_a.claim_text,
            "direction_a": self.claim_a.direction,
            "quantitative_a": self.claim_a.quantitative,
            "conditions_a": self.claim_a.conditions,
            "paper_b_id": self.claim_b.paper_id,
            "paper_b_title": self.claim_b.title,
            "paper_b_year": self.claim_b.year,
            "claim_b": self.claim_b.claim_text,
            "direction_b": self.claim_b.direction,
            "quantitative_b": self.claim_b.quantitative,
            "conditions_b": self.claim_b.conditions,
            "similarity": round(self.similarity, 4),
        }


class CandidateFinder:
    def __init__(self, embedder: Embedder, sim_threshold: float = 0.85):
        self.embedder = embedder
        self.sim_threshold = sim_threshold

    def find(self, summaries_dir: str) -> list[ContraCandidate]:
        """
        Find all contradiction candidates across all summaries.
        Returns list of ContraCandidate pairs.
        """
        claims = self._load_claims(summaries_dir)
        logger.info(f"Loaded {len(claims)} claims from {summaries_dir}")

        if len(claims) < 2:
            return []

        # Embed all claim texts
        texts = [c.claim_text for c in claims]
        logger.info("Embedding claims for contradiction search...")
        embeddings = self.embedder.embed_documents(texts)

        # Cosine similarity matrix (embeddings are already normalized)
        sim_matrix = embeddings @ embeddings.T

        candidates: list[ContraCandidate] = []
        n = len(claims)

        logger.info(f"Scanning {n*(n-1)//2:,} claim pairs (sim threshold={self.sim_threshold})")

        for i in range(n):
            for j in range(i + 1, n):
                # Different papers only
                if claims[i].paper_id == claims[j].paper_id:
                    continue
                sim = float(sim_matrix[i, j])
                if sim < self.sim_threshold:
                    continue
                # Check opposing directions
                dir_pair = frozenset({claims[i].direction, claims[j].direction})
                if dir_pair not in OPPOSING_DIRECTIONS:
                    continue
                candidates.append(ContraCandidate(
                    claim_a=claims[i],
                    claim_b=claims[j],
                    similarity=sim,
                ))

        # Sort by similarity descending
        candidates.sort(key=lambda c: c.similarity, reverse=True)
        logger.info(f"Found {len(candidates)} candidate contradiction pairs")
        return candidates

    def _load_claims(self, summaries_dir: str) -> list[ClaimRecord]:
        claims = []
        for fpath in sorted(Path(summaries_dir).glob("*.json")):
            try:
                with open(fpath) as f:
                    s = json.load(f)
                for claim in s.get("claims", []):
                    direction = claim.get("direction", "unclear")
                    if direction in ("unclear", "bidirectional"):
                        continue  # can't detect direction-based contradictions
                    claims.append(ClaimRecord(
                        paper_id=s["paper_id"],
                        title=s.get("title", ""),
                        year=s.get("year", ""),
                        claim_text=claim.get("claim", ""),
                        direction=direction,
                        quantitative=claim.get("quantitative", ""),
                        conditions=claim.get("conditions", ""),
                    ))
            except Exception as e:
                logger.warning(f"Could not load {fpath}: {e}")
        return claims
