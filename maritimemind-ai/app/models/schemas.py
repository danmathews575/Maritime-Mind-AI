from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone

# ---------------------------------------------------------
# Sub-Schemas & Metadata Components
# ---------------------------------------------------------

class BoundingBox(BaseModel):
    """Spatial coordinates for images extracted from pages."""
    x0: float
    y0: float
    x1: float
    y1: float

class RetrievalScores(BaseModel):
    """Detailed confidence scoring for a retrieved result."""
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    confidence_score: float = 0.0  # Normalized 0-1 confidence

# ---------------------------------------------------------
# Core Data Modalities
# ---------------------------------------------------------

class TextChunk(BaseModel):
    """Represents a hierarchical semantic chunk of a manual."""
    chunk_id: str
    manual_name: str
    department: str = "general"
    page_number: int
    section_title: str
    content: str
    keywords: List[str] = Field(default_factory=list)
    related_images: List[str] = Field(default_factory=list)
    
    # Hierarchy & Graph Routing
    hierarchy_path: List[str] = Field(default_factory=list)
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    
    # Metadata tracking
    embedding_model: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ImageMetadata(BaseModel):
    """Represents extracted diagrams, schematics, and figures."""
    image_id: str
    manual_name: str
    page_number: int
    image_path: str
    caption: str
    bbox: BoundingBox
    linked_chunks: List[str] = Field(default_factory=list)
    ocr_text: Optional[str] = None
    
    # Metadata tracking
    embedding_model: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ---------------------------------------------------------
# Retrieval & Query Envelopes
# ---------------------------------------------------------

class RetrievalResult(BaseModel):
    """A matched chunk or image bundled with scoring metadata."""
    chunk_id: str
    content: str
    manual_name: str
    page_number: int
    section_title: str
    hierarchy_path: List[str]
    related_images: List[str]
    
    # Tracking multiple scoring factors
    scores: RetrievalScores

class QueryRequest(BaseModel):
    """Input contract for the retrieval and routing layer."""
    query: str
    department_filter: Optional[str] = None
    manual_filter: Optional[str] = None
    top_k: int = 5
    include_images: bool = True
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Output contract for the retrieval layer."""
    query_id: str
    original_query: str
    results: List[RetrievalResult] = Field(default_factory=list)
    classified_intent: str = "UNKNOWN"
    processing_time_ms: float = 0.0

# ---------------------------------------------------------
# Pipeline Management
# ---------------------------------------------------------

class ProcessingManifest(BaseModel):
    """Tracks state and ingestion status of PDF files."""
    file_name: str
    status: str  # e.g., 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'
    processed_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    chunk_count: int = 0
    image_count: int = 0
    errors: List[str] = Field(default_factory=list)
