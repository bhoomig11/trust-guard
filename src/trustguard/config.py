"""Application settings loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for TrustGuard.

    Every field can be overridden with an environment variable prefixed by
    ``TRUSTGUARD_`` (for example ``TRUSTGUARD_TOP_K=6``). The Anthropic key is
    read from the standard ``ANTHROPIC_API_KEY`` variable.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRUSTGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-haiku-5-5"
    llm_effort: str = "low"
    llm_thinking: str = "adaptive"
    llm_max_tokens: int = 4096
    agent_max_steps: int = 4

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_md"

    chroma_dir: Path = Path("./chroma_data")
    collection_name: str = "policies"
    data_dir: Path = Path("./data/policies")
    log_path: Path = Path("./logs/governance.jsonl")

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_similarity: float = 0.25
    auto_ingest: bool = True
    max_upload_mb: int = 10


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached `Settings` instance."""
    return Settings()
