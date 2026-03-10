"""
Cluster paper summaries into topic groups using UMAP + HDBSCAN.

Embeds each paper's title + key_findings + keywords (not full text)
to get a topic-level representation, then clusters.

Zotero collection memberships are used as validation labels.

Usage:
    clusterer = PaperClusterer(embedder)
    assignments = clusterer.cluster(summaries_dir="data/summaries")
    # assignments: {paper_id: cluster_id, ...}  (-1 = noise/unclustered)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.embedding.embedder import Embedder


class PaperClusterer:
    def __init__(
        self,
        embedder: Embedder,
        umap_components: int = 10,
        umap_metric: str = "cosine",
        hdbscan_min_cluster_size: int = 8,
        hdbscan_min_samples: int = 3,
        random_state: int = 42,
    ):
        self.embedder = embedder
        self.umap_components = umap_components
        self.umap_metric = umap_metric
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.random_state = random_state

    def cluster(self, summaries_dir: str) -> dict[int, int]:
        """
        Load all summaries, embed, cluster, return {paper_id: cluster_id}.
        cluster_id == -1 means noise (unclustered).
        """
        summaries = self._load_summaries(summaries_dir)
        if not summaries:
            raise ValueError(f"No summaries found in {summaries_dir}")

        logger.info(f"Clustering {len(summaries)} paper summaries")

        paper_ids = [s["paper_id"] for s in summaries]
        texts = [self._summary_to_text(s) for s in summaries]

        # Embed
        logger.info("Embedding summaries for clustering...")
        embeddings = self.embedder.embed_documents(texts)

        # UMAP dimensionality reduction
        logger.info(f"Running UMAP (n_components={self.umap_components})...")
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")

        reducer = umap.UMAP(
            n_components=self.umap_components,
            metric=self.umap_metric,
            random_state=self.random_state,
            low_memory=True,
        )
        reduced = reducer.fit_transform(embeddings)

        # HDBSCAN clustering
        # Prefer sklearn's built-in HDBSCAN (no numba/llvmlite dependency).
        # Fall back to standalone hdbscan package if sklearn < 1.3.
        logger.info("Running HDBSCAN clustering...")
        try:
            from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
            clusterer = SklearnHDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(reduced)
        except ImportError:
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.hdbscan_min_cluster_size,
                    min_samples=self.hdbscan_min_samples,
                    metric="euclidean",
                    cluster_selection_method="eom",
                )
                labels = clusterer.fit_predict(reduced)
            except ImportError:
                raise ImportError(
                    "HDBSCAN not available. Either upgrade scikit-learn >= 1.3 or:\n"
                    "  conda install -c conda-forge numba llvmlite\n"
                    "  pip install hdbscan"
                )

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")

        assignments = {pid: int(label) for pid, label in zip(paper_ids, labels)}

        # Log cluster sizes
        from collections import Counter
        counts = Counter(labels)
        for cluster_id, count in sorted(counts.items()):
            tag = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
            logger.debug(f"  {tag}: {count} papers")

        return assignments

    def _load_summaries(self, summaries_dir: str) -> list[dict]:
        path = Path(summaries_dir)
        summaries = []
        for fpath in sorted(path.glob("*.json")):
            try:
                with open(fpath) as f:
                    summaries.append(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load {fpath}: {e}")
        return summaries

    def _summary_to_text(self, summary: dict) -> str:
        """Combine title + findings + keywords into a topic-representative string."""
        parts = [summary.get("title", "")]
        parts.extend(summary.get("key_findings", []))
        parts.extend(summary.get("keywords", []))
        return " ".join(p for p in parts if p)
