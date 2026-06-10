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
Nothing special required — Qdrant handles persistence.

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

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.configs.limiter import limiter

from app.configs.config import get_settings
from app.utils.logger import setup_logger

# Observability (Arize Phoenix)
try:
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    
    # Register Phoenix Tracer globally
    tracer_provider = register(
        project_name="maritimemind-ai",
        endpoint="http://localhost:4317", # OpenTelemetry gRPC endpoint
    )
    LangChainInstrumentor().instrument()
    _phoenix_enabled = True
except ImportError:
    _phoenix_enabled = False
    logger = logging.getLogger("maritimemind.api")
    logger.warning("Arize Phoenix instrumentation not installed. Tracing disabled.")

# Routes
from app.api.routes import health as health_router
from app.api.routes import sessions as sessions_router
from app.api.routes import query as query_router
from app.api.routes import ingestion as ingestion_router
from app.api.routes import auth as auth_router

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
    """Pre-warm all expensive models at startup to eliminate cold-start during demo."""
    import asyncio
    logger.info("MaritimeMind AI API starting up…")

    def _warm_all_models():
        # 1. Text embedding model
        try:
            from app.services.embedding import TextEmbeddingService
            emb = TextEmbeddingService()
            emb.embed_query("maritime engine manual technical warmup query")
            logger.info("Text embedding model pre-warmed")
        except Exception as e:
            logger.warning(f"Could not pre-warm embedding model: {e}")

        # 2. CLIP image embedding model (largest cold-start contributor)
        try:
            from app.services.clip_embedding import ImageEmbeddingService
            clip = ImageEmbeddingService()
            clip.embed_text("maritime diagram schematic")
            logger.info("CLIP image model pre-warmed")
        except Exception as e:
            logger.warning(f"Could not pre-warm CLIP model: {e}")

        # 3. BM25 index
        try:
            from app.services.bm25_index import BM25IndexService
            bm25 = BM25IndexService()
            if not bm25.is_built and os.path.exists(settings.BM25_INDEX_PATH):
                bm25.load()
                logger.info("BM25 index loaded")
            elif not bm25.is_built:
                logger.warning("BM25 index not found — run ingestion first")
        except Exception as e:
            logger.warning(f"Could not load BM25 index: {e}")

        # 4. Cross-encoder reranker
        try:
            from app.retrieval.reranker import RerankerService
            RerankerService()  # Loads the cross-encoder
            logger.info("Reranker model pre-warmed")
        except Exception as e:
            logger.warning(f"Could not pre-warm reranker: {e}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _warm_all_models)
    logger.info("All models ready — API is warm.")

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
    version="0.9.2",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with Phoenix
if _phoenix_enabled:
    pass

# ─── Static files — serve extracted diagram images ───────────────────────────
_images_dir = os.path.abspath(settings.EXTRACTED_IMAGES_DIR)
os.makedirs(_images_dir, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=_images_dir), name="static-images")

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health_router.router)
app.include_router(sessions_router.router)
app.include_router(query_router.router)
app.include_router(ingestion_router.router)
app.include_router(auth_router.router)


# ─── Request logging middleware ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
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
        "version": "0.9.2",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
