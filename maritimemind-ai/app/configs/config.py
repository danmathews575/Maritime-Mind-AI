import os
import secrets
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class MaritimeMindSettings(BaseSettings):
    """
    Centralized configuration for MaritimeMind AI.
    Loads from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Security
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_hex(32))
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # LLM Inference
    LLM_PROVIDER: str = "nvidia"  # "nvidia", "ollama", "gemini", or "openai"
    LLM_FALLBACK_ORDER: str = "nvidia,gemini,ollama"  # Comma-separated fallback chain
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:8b"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""

    # Embedding Models
    TEXT_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    TEXT_EMBEDDING_DIM: int = 384
    MULTILINGUAL_ENABLED: bool = True
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
    IMAGE_EMBEDDING_DIM: int = 512

    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MIN_CHUNK_LENGTH: int = 50

    # Image Extraction
    MIN_IMAGE_WIDTH: int = 100
    MIN_IMAGE_HEIGHT: int = 100

    # Vector Store & Storage
    QDRANT_HOST: str = "local" # Use "local" to bypass docker, "localhost" to use docker
    QDRANT_PATH: str = "./vector_store/qdrant_local"
    QDRANT_PORT: int = 6333
    TEXT_COLLECTION_NAME: str = "maritimemind_multilingual_text"
    IMAGE_COLLECTION_NAME: str = "maritimemind_multilingual_images"
    LEGACY_TEXT_COLLECTION: str = "maritime_text_chunks"  # Phase pre-15.1 (English-only, 768-dim)
    LEGACY_IMAGE_COLLECTION: str = "maritime_image_chunks"
    BM25_INDEX_PATH: str = "./vector_store/bm25_multilingual.pkl"
    LEGACY_BM25_INDEX_PATH: str = "./vector_store/bm25_index.pkl"
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 600  # 10 minutes — eliminates repeated demo query latency

    # Retrieval
    TOP_K_RESULTS: int = 10
    RRF_K: int = 60
    CONFIDENCE_THRESHOLD: float = 0.6
    RERANKING_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Storage Paths
    DATA_DIR: str = "./data"
    RAW_PDF_DIR: str = "./data/raw_pdfs"
    EXTRACTED_TEXT_DIR: str = "./data/extracted_text"
    EXTRACTED_IMAGES_DIR: str = "./data/extracted_images"
    PROCESSED_CHUNKS_DIR: str = "./data/processed_chunks"
    METADATA_DIR: str = "./data/metadata"

    # Logging
    LOG_FILE: str = "./logs/maritimemind.log"
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = 10_485_760  # 10MB
    LOG_BACKUP_COUNT: int = 30

    # Embedding
    EMBEDDING_BATCH_SIZE: int = 32

    # Device Configuration
    DEVICE: str = "cpu"  # "cpu" or "cuda"

    # Future Phases (stubs — uncomment/configure when needed)
    OCR_ENABLED: bool = True
    EASYOCR_LANGUAGES: list = ["en"]  # Phase 3: EasyOCR language list
    NEO4J_URI: Optional[str] = None
    STREAMING_ENABLED: bool = False
    MAX_CONVERSATION_HISTORY: int = 10

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:80",
        "http://localhost",
    ]

# Singleton instance
settings = MaritimeMindSettings()


def get_settings() -> MaritimeMindSettings:
    """Returns the singleton settings instance."""
    return settings
