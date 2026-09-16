from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "Document Q&A"
    app_env: str = "development"
    debug: bool = True

    openai_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "production-rag-index"

    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()