"""
app/api/main.py
================
FastAPI application factory for MaritimeMind AI.

Startup lifecycle
-----------------
1. Load settings
2. Pre-warm the text-embedding model (avoids cold-start on first request)
3. Pre-load the BM25 index into memory
4. Mount static file server for extracted diagrams

Shutdown lifecycle
------------------
Nothing special required — ChromaDB is embedded (file-based).

Run with:
    uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.configs.config import get_settings
from app.utils.logger import setup_logger

# Routes
from app.api.routes import health as health_router
from app.api.routes import sessions as sessions_router
from app.api.routes import query as query_router
from app.api.routes import ingestion as ingestion_router

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
setup_logger("maritimemind.api")
logger = logging.getLogger("maritimemind.api")
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown hooks
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm expensive models at startup so the first request is fast."""
    logger.info("MaritimeMind AI API starting up …")

    # Pre-warm text embedding model
    try:
        from app.services.embedding import TextEmbeddingService
        emb = TextEmbeddingService()
        emb.embed_query("warmup")
        logger.info("Text embedding model pre-warmed ✓")
    except Exception as e:
        logger.warning(f"Could not pre-warm embedding model: {e}")

    # Pre-load BM25 index
    try:
        from app.services.bm25_index import BM25IndexService
        bm25 = BM25IndexService()
        if not bm25.is_built and os.path.exists(settings.BM25_INDEX_PATH):
            bm25.load()
            logger.info("BM25 index loaded ✓")
        elif not bm25.is_built:
            logger.warning("BM25 index not found — run ingestion first")
    except Exception as e:
        logger.warning(f"Could not load BM25 index: {e}")

    yield  # ← Application runs here

    logger.info("MaritimeMind AI API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MaritimeMind AI",
    description=(
        "Offline-capable, multimodal maritime intelligence platform. "
        "Submit natural-language questions about ship manuals and receive "
        "grounded answers with source citations and engineering diagrams."
    ),
    version="0.9.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS (allow Streamlit dev server on port 8501) ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",   # React / any future frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static files — serve extracted diagram images ───────────────────────────
_images_dir = os.path.abspath(settings.EXTRACTED_IMAGES_DIR)
os.makedirs(_images_dir, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=_images_dir), name="static-images")

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health_router.router)
app.include_router(sessions_router.router)
app.include_router(query_router.router)
app.include_router(ingestion_router.router)


# ─── Request logging middleware ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({elapsed_ms:.1f} ms)"
    )
    return response


# ─── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception for {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "INTERNAL_ERROR"},
    )


# ─── Root endpoint ────────────────────────────────────────────────────────────
@app.get("/", tags=["root"], summary="API root")
async def root():
    return {
        "name": "MaritimeMind AI",
        "version": "0.9.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
