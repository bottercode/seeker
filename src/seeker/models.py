"""Lazily-loaded ML models (embedding + reranker), shared process-wide."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from seeker.config import get_settings


def _device() -> str:
    s = get_settings()
    if s.device != "auto":
        return s.device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embed_model, device=_device())


@lru_cache
def reranker():
    from sentence_transformers import CrossEncoder

    s = get_settings()
    return CrossEncoder(s.rerank_model, device=_device(), max_length=s.rerank_max_length)


def embed_query(text: str) -> np.ndarray:
    s = get_settings()
    return embedder().encode(s.query_instruction + text, normalize_embeddings=True, show_progress_bar=False)


def embed_docs(texts: list[str]) -> np.ndarray:
    """Documents are embedded without the query instruction (matches the precomputed vectors)."""
    return embedder().encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def rerank_scores(query: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    return [float(x) for x in reranker().predict([(query, d) for d in docs], batch_size=32, show_progress_bar=False)]
