import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration for MaritimeMind AI.
    Loads from environment variables or .env file.
    """
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Base Directories (Dynamic, no hardcoded strings)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_PDF_DIR: Path = DATA_DIR / "raw_pdfs"
    EXTRACTED_TEXT_DIR: Path = DATA_DIR / "extracted_text"
    EXTRACTED_IMAGES_DIR: Path = DATA_DIR / "extracted_images"
    METADATA_DIR: Path = DATA_DIR / "metadata"
    PROCESSED_CHUNKS_DIR: Path = DATA_DIR / "processed_chunks"
    
    VECTOR_STORE_DIR: Path = BASE_DIR / "vector_store"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Embedding Configurations
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    IMAGE_EMBEDDING_MODEL: str = "ViT-B-32"
    OPEN_CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
    
    # Vector Database (ChromaDB)
    CHROMA_PERSIST_DIRECTORY: str = str(VECTOR_STORE_DIR / "chromadb")
    TEXT_COLLECTION_NAME: str = "maritime_text_chunks"
    IMAGE_COLLECTION_NAME: str = "maritime_image_chunks"
    
    # Hardware & Inference
    DEVICE: str = "cpu"  # Options: 'cuda', 'mps', or 'cpu' (Future GPU switching)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:8b"
    
    # Chunking Parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    
    # Retrieval Settings
    TOP_K_VECTOR: int = 5
    TOP_K_BM25: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Image Extraction Settings
    MIN_IMAGE_WIDTH: int = 100
    MIN_IMAGE_HEIGHT: int = 100
    EXTRACT_IMAGE_FORMAT: str = "png"
    
    # LangGraph Config (Future Orchestration)
    MAX_ROUTING_RETRIES: int = 3
    MAX_AGENT_STEPS: int = 10
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = 10485760  # 10 MB
    LOG_BACKUP_COUNT: int = 30
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
