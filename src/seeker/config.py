from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All knobs, overridable via SEEKER_* env vars or a .env file."""

    model_config = SettingsConfigDict(env_prefix="SEEKER_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    table: str = "pages"

    # Source dataset (FineWeb-Edu with precomputed bge-small-en-v1.5 vectors).
    source_uri: str = "hf://datasets/lance-format/fineweb-edu/data/train.lance"

    # Must match the model that produced the source vectors (verified: cosine 1.0).
    embed_model: str = "BAAI/bge-small-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # Reranking in `auto` mode. On the known-item eval (100k pages) it did not beat plain
    # hybrid fusion and costs ~230 ms on an M2, so it is off by default.
    # off | blend (fuse reranker rank with hybrid rank) | pure (reranker order only)
    rerank: str = "off"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"  # or BAAI/bge-reranker-base (much slower)
    rerank_depth: int = 30  # candidates sent to the cross-encoder
    rerank_max_length: int = 256  # tokens per (query, doc) pair
    rerank_chars: int = 1000  # leading characters of each page given to the reranker
    # Host quality: drop spam hosts; nudge hybrid ranking toward known sites (Tranco/.gov/.edu).
    spam_filter: bool = True
    # Popularity prior weight (RRF scores are ~0.03). Off: it did not improve the relevance eval.
    prior_weight: float = 0.0
    candidate_k: int = 100  # candidates pulled from each retriever before fusion
    rrf_k: int = 60

    device: str = "auto"  # auto | cpu | mps | cuda

    # Optional HyDE query rewriting through a local Ollama model ("" = off).
    hyde_model: str = ""
    ollama_url: str = "http://localhost:11434"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lancedb"


@lru_cache
def get_settings() -> Settings:
    return Settings()
