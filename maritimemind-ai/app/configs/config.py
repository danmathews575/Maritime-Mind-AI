import os
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

    # LLM Inference
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:8b"

    # Embedding Models
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "laion2b_s34b_b79k"

    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MIN_CHUNK_LENGTH: int = 50

    # Image Extraction
    MIN_IMAGE_WIDTH: int = 100
    MIN_IMAGE_HEIGHT: int = 100

    # Vector Store
    CHROMADB_PERSIST_DIR: str = "./vector_store/chromadb"
    TEXT_COLLECTION_NAME: str = "maritime_text_chunks"
    IMAGE_COLLECTION_NAME: str = "maritime_image_chunks"
    BM25_INDEX_PATH: str = "./vector_store/bm25_index.pkl"

    # Retrieval
    TOP_K_RESULTS: int = 10
    RRF_K: int = 60
    CONFIDENCE_THRESHOLD: float = 0.6

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

# Singleton instance
settings = MaritimeMindSettings()


def get_settings() -> MaritimeMindSettings:
    """Returns the singleton settings instance."""
    return settings
