"""
Claim-level clustering for cross-paper synthesis.

Loads structured claims from both per-paper summaries (data/summaries/)
and critical reviews (data/reviews/), embeds claim text, then clusters
using UMAP + HDBSCAN to find groups of related claims across papers.

Claims from summaries carry: direction, quantitative, conditions.
Claims from reviews carry: verdict, ideal_experiments, paper_justification, required_experiments.

Output:
  data/clusters/claim_assignments.jsonl  — one record per claim with cluster_id
  data/clusters/cluster_index.json       — {cluster_id: [claim indices]} index

Usage:
    clusterer = ClaimClusterer(embedder)
    claims, labels = clusterer.cluster(
        summaries_dir="data/summaries",
        reviews_dir="data/reviews",
        metadata_path="data/extracted/papers_metadata.jsonl",
    )
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.embedding.embedder import Embedder


@dataclass
class UnifiedClaim:
    paper_id: int
    title: str
    year: str
    journal: str
    collections: list[str]
    claim_text: str
    source: str              # "summary" | "review"
    # Summary-only fields (empty string if source == "review")
    direction: str = ""
    quantitative: str = ""
    conditions: str = ""
    # Review-only fields (empty string if source == "summary")
    verdict: str = ""        # justified | partially_justified | not_justified
    ideal_experiments: str = ""
    paper_justification: str = ""
    required_experiments: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "collections": self.collections,
            "claim_text": self.claim_text,
            "source": self.source,
            "direction": self.direction,
            "quantitative": self.quantitative,
            "conditions": self.conditions,
            "verdict": self.verdict,
            "ideal_experiments": self.ideal_experiments,
            "paper_justification": self.paper_justification,
            "required_experiments": self.required_experiments,
        }


class ClaimClusterer:
    def __init__(
        self,
        embedder: Embedder,
        umap_components: int = 10,
        umap_neighbors: int = 15,
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: int = 3,
    ):
        self.embedder = embedder
        self.umap_components = umap_components
        self.umap_neighbors = umap_neighbors
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples

    def cluster(
        self,
        summaries_dir: str = "data/summaries",
        reviews_dir: str = "data/reviews",
        metadata_path: str = "data/extracted/papers_metadata.jsonl",
    ) -> tuple[list[UnifiedClaim], np.ndarray]:
        """
        Load claims, embed, and cluster.
        Returns (claims, cluster_labels) where labels[i] is the cluster id
        for claims[i]. Noise points are labeled -1.
        """
        metadata = self._load_metadata(metadata_path)
        claims = self._load_claims(summaries_dir, reviews_dir, metadata)
        logger.info(f"Loaded {len(claims)} claims total "
                    f"({sum(1 for c in claims if c.source == 'summary')} summary, "
                    f"{sum(1 for c in claims if c.source == 'review')} review)")

        if len(claims) < self.hdbscan_min_cluster_size:
            logger.error("Not enough claims to cluster")
            return claims, np.full(len(claims), -1)

        texts = [c.claim_text for c in claims]
        logger.info("Embedding claims...")
        embeddings = self.embedder.embed_documents(texts)

        logger.info("Running UMAP...")
        import umap
        reducer = umap.UMAP(
            n_components=self.umap_components,
            n_neighbors=self.umap_neighbors,
            metric="cosine",
            min_dist=0.1,
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)

        logger.info("Running HDBSCAN...")
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
        return claims, labels

    def _load_metadata(self, metadata_path: str) -> dict[int, dict]:
        """Return {paper_id: {collections, title, year, journal}} from metadata JSONL."""
        meta = {}
        path = Path(metadata_path)
        if not path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return meta
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                pid = rec.get("paper_id")
                if pid is not None:
                    meta[int(pid)] = {
                        "collections": rec.get("collections", []),
                        "title": rec.get("title", ""),
                        "year": rec.get("year", ""),
                        "journal": rec.get("journal", ""),
                    }
        return meta

    def _load_claims(
        self,
        summaries_dir: str,
        reviews_dir: str,
        metadata: dict[int, dict],
    ) -> list[UnifiedClaim]:
        claims = []
        claims.extend(self._load_summary_claims(summaries_dir, metadata))
        claims.extend(self._load_review_claims(reviews_dir, metadata))
        return claims

    def _load_summary_claims(
        self, summaries_dir: str, metadata: dict[int, dict]
    ) -> list[UnifiedClaim]:
        claims = []
        for fpath in sorted(Path(summaries_dir).glob("*.json")):
            try:
                with open(fpath) as f:
                    s = json.load(f)
                pid = int(s["paper_id"])
                meta = metadata.get(pid, {})
                for claim in s.get("claims", []):
                    text = claim.get("claim", "").strip()
                    if not text:
                        continue
                    claims.append(UnifiedClaim(
                        paper_id=pid,
                        title=s.get("title", meta.get("title", "")),
                        year=s.get("year", meta.get("year", "")),
                        journal=s.get("journal", meta.get("journal", "")),
                        collections=meta.get("collections", []),
                        claim_text=text,
                        source="summary",
                        direction=claim.get("direction", ""),
                        quantitative=claim.get("quantitative", ""),
                        conditions=claim.get("conditions", ""),
                    ))
            except Exception as e:
                logger.warning(f"Could not load summary {fpath}: {e}")
        return claims

    def _load_review_claims(
        self, reviews_dir: str, metadata: dict[int, dict]
    ) -> list[UnifiedClaim]:
        claims = []
        for fpath in sorted(Path(reviews_dir).glob("*.json")):
            try:
                with open(fpath) as f:
                    r = json.load(f)
                pid = int(r["paper_id"])
                meta = metadata.get(pid, {})
                for claim in r.get("claims", []):
                    text = claim.get("claim", "").strip()
                    if not text:
                        continue
                    claims.append(UnifiedClaim(
                        paper_id=pid,
                        title=r.get("title", meta.get("title", "")),
                        year=r.get("year", meta.get("year", "")),
                        journal=r.get("journal", meta.get("journal", "")),
                        collections=meta.get("collections", []),
                        claim_text=text,
                        source="review",
                        verdict=claim.get("verdict", ""),
                        ideal_experiments=claim.get("ideal_experiments", ""),
                        paper_justification=claim.get("paper_justification", ""),
                        required_experiments=claim.get("required_experiments", ""),
                    ))
            except Exception as e:
                logger.warning(f"Could not load review {fpath}: {e}")
        return claims
