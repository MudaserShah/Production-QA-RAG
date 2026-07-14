# config.py — Settings for Document Butler.
# Reads from .env automatically via pydantic-settings.

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"

    # Embeddings: "huggingface" (free, 384-dim) | "openai" (paid, 1536-dim)
    embedding_provider: str = "huggingface"

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "document_butler"

    # File storage
    upload_dir: str = "uploads_tmp"       # Temp folder during processing (files deleted after)
    uploads_dir: str = "uploads"          # Permanent storage for original uploaded files
    markdown_dir: str = "markdown_files"  # Generated markdown file per uploaded document
    max_upload_size_mb: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
