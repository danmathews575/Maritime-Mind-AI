"""
app/api/schemas.py
==================
Pydantic request/response models for the MaritimeMind AI REST API.
Decoupled from internal domain models to give the API a stable contract.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class CitationOut(BaseModel):
    """A single source citation attached to a response."""
    manual_name: str
    page_number: int
    section_title: str = ""
    chunk_id: str
    ship_id: str = ""


class ImageOut(BaseModel):
    """A retrieved diagram/image returned with a response."""
    image_id: str
    url: str                 # relative URL: /static/images/<manual>/<image_id>.png
    caption: str = ""
    page_number: int
    manual_name: str
    diagram_type: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Body for POST /api/v1/query."""
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    top_k: int = Field(10, ge=1, le=50, description="Max text chunks to retrieve")
    provider: Optional[str] = Field(None, description="Optional LLM provider (ollama, gemini, openai)")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")


class QueryResponse(BaseModel):
    """Response body for POST /api/v1/query."""
    answer: str
    citations: List[CitationOut] = Field(default_factory=list)
    images: List[ImageOut] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    intent: str = ""
    session_id: Optional[str] = None
    quality_passed: bool = True
    quality_notes: str = ""
    detected_language: str = ""


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

class SessionCreateResponse(BaseModel):
    """Response body for POST /api/v1/sessions."""
    session_id: str
    created_at: float
    message_count: int = 0


class ChatMessageOut(BaseModel):
    """A single message in a session history."""
    role: str
    content: str
    images: List[str] = Field(default_factory=list)


class SessionHistoryResponse(BaseModel):
    """Response body for GET /api/v1/sessions/{session_id}/history."""
    session_id: str
    messages: List[ChatMessageOut]


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Body for POST /api/v1/ingest."""
    pdf_path: Optional[str] = Field(None, description="Path to a single PDF file")
    pdf_dir: Optional[str] = Field(None, description="Path to a directory of PDFs")
    force_reingest: bool = Field(False, description="Re-ingest even if already processed")


class IngestResponse(BaseModel):
    """Response body for POST /api/v1/ingest."""
    status: str           # "started" | "completed" | "error"
    manual_name: str = ""
    chunk_count: int = 0
    image_count: int = 0
    errors: List[str] = Field(default_factory=list)


class IngestStatusItem(BaseModel):
    manual_name: str
    status: str
    chunk_count: int
    image_count: int
    processed_date: str


class IngestStatusResponse(BaseModel):
    """Response body for GET /api/v1/ingest/status."""
    total_manuals: int
    manuals: List[IngestStatusItem]


# ---------------------------------------------------------------------------
# Health / system endpoints
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response body for GET /api/v1/health."""
    status: str               # "healthy" | "degraded" | "unhealthy"
    ollama_available: bool
    vector_store_ready: bool
    bm25_index_ready: bool
    embedding_model_ready: bool
    llm_providers: Dict[str, bool] = Field(default_factory=dict, description="Per-provider availability")
    llm_fallback_order: List[str] = Field(default_factory=list, description="Configured fallback chain")


class StatsResponse(BaseModel):
    """Response body for GET /api/v1/stats."""
    text_chunk_count: int
    image_count: int
    ollama_model: str
    embedding_model: str
    bm25_index_path: str
    qdrant_path: str


# ---------------------------------------------------------------------------
# Generic error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
