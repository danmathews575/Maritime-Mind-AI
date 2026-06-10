"""
app/api/routes/health.py
=========================
Health-check and system-stats endpoints.
"""
import logging
import os
from fastapi import APIRouter

from app.api.schemas import HealthResponse, StatsResponse
from app.configs.config import get_settings
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService
from app.services.embedding import TextEmbeddingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check() -> HealthResponse:
    """
    Returns system health status.
    Checks: LLM providers (with fallback chain), vector store, BM25 index, embedding model.
    """
    llm_providers = {}
    llm_fallback_order = []
    ollama_ok = False
    vector_ok = False
    bm25_ok = False
    embedding_ok = False

    # --- LLM provider check (multi-provider with fallback) ---
    try:
        llm = LLMService()
        llm_providers = llm.health_check()
        llm_fallback_order = llm.fallback_order
        ollama_ok = llm_providers.get("ollama", False)
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")

    # --- Vector store check ---
    try:
        vs = VectorStoreService()
        stats = vs.get_collection_stats()
        vector_ok = stats.get("text_collection", {}).get("count", 0) >= 0
    except Exception as e:
        logger.warning(f"Vector store check failed: {e}")

    # --- BM25 check ---
    try:
        bm25_ok = os.path.exists(settings.BM25_INDEX_PATH)
    except Exception:
        pass

    # --- Embedding model check ---
    try:
        emb = TextEmbeddingService()
        _ = emb.embed_query("test")
        embedding_ok = True
    except Exception as e:
        logger.warning(f"Embedding model check failed: {e}")

    # --- Determine overall status ---
    any_llm_available = any(llm_providers.values())
    overall = "healthy"
    if not vector_ok or not embedding_ok:
        overall = "degraded"
    if not any_llm_available:
        overall = "degraded"
    if not vector_ok and not embedding_ok:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        ollama_available=ollama_ok,
        vector_store_ready=vector_ok,
        bm25_index_ready=bm25_ok,
        embedding_model_ready=embedding_ok,
        llm_providers=llm_providers,
        llm_fallback_order=llm_fallback_order,
    )


@router.get("/stats", response_model=StatsResponse, summary="System statistics")
async def get_stats() -> StatsResponse:
    """Returns counts and model configuration info."""
    text_count = 0
    image_count = 0

    try:
        vs = VectorStoreService()
        stats = vs.get_collection_stats()
        text_count = stats.get("text_collection", {}).get("count", 0)
        image_count = stats.get("image_collection", {}).get("count", 0)
    except Exception as e:
        logger.warning(f"Could not fetch vector store stats: {e}")

    return StatsResponse(
        text_chunk_count=text_count,
        image_count=image_count,
        ollama_model=settings.OLLAMA_MODEL,
        embedding_model=settings.TEXT_EMBEDDING_MODEL,
        bm25_index_path=settings.BM25_INDEX_PATH,
        qdrant_path=settings.QDRANT_PATH,
    )
