"""
Wrapper around nomic-embed-text-v1.5 for generating embeddings.

nomic-embed-text requires a task prefix on inputs:
  - Documents: "search_document: {text}"
  - Queries:   "search_query: {text}"

Usage:
    embedder = Embedder()
    vecs = embedder.embed_documents(["text1", "text2"])
    qvec = embedder.embed_query("what is theta-gamma coupling?")
"""

from typing import Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "
VECTOR_DIM = 768


class Embedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,   # required for nomic-embed-text
            device=device,
        )
        # Cap sequence length to 512 tokens. nomic-bert-2048 supports up to 8K,
        # but MPS's scaled_dot_product_attention allocates a buffer proportional
        # to max_seq_length^2 — at 2048 tokens this causes "Invalid buffer size: 12GB".
        self.model.max_seq_length = 512
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of document texts.
        Returns float32 array of shape (N, VECTOR_DIM).
        """
        prefixed = [DOC_PREFIX + t for t in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns float32 array of shape (VECTOR_DIM,).
        """
        prefixed = QUERY_PREFIX + text
        result = self._encode([prefixed])
        return result[0]

    def embed_batch(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Generic batch embed with explicit prefix choice."""
        prefix = QUERY_PREFIX if is_query else DOC_PREFIX
        prefixed = [prefix + t for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # cosine sim = dot product on unit vectors
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)
