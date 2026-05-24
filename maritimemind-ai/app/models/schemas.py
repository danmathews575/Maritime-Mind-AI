from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)

class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

class QueryIntent(str, Enum):
    EXPLANATION = "EXPLANATION"
    PROCEDURE = "PROCEDURE"
    TROUBLESHOOTING = "TROUBLESHOOTING"
    DIAGRAM_REQUEST = "DIAGRAM_REQUEST"
    EMERGENCY = "EMERGENCY"
    SOP_LOOKUP = "SOP_LOOKUP"

class TextChunk(BaseModel):
    chunk_id: str
    manual_name: str
    ship_id: Optional[str] = None
    language: Optional[str] = None
    department: str
    subsystem: str = "general"
    document_type: str = "manual"
    page_number: int
    section_title: str
    content: str
    
    # Semantic Routing Metadata
    contains_procedure: bool = False
    contains_warning: bool = False
    contains_emergency_workflow: bool = False
    contains_diagram_reference: bool = False
    importance: str = "medium"  # high, medium, low
    applicable_intents: List[QueryIntent] = Field(default_factory=list)
    
    keywords: List[str] = Field(default_factory=list)
    related_image_ids: List[str] = Field(default_factory=list)
    diagram_references: List[str] = Field(default_factory=list)
    hierarchy_path: List[str] = Field(default_factory=list)
    
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    
    embedding_model: str
    created_at: datetime = Field(default_factory=_utcnow)

class ImageMetadata(BaseModel):
    image_id: str
    manual_name: str
    ship_id: Optional[str] = None
    language: Optional[str] = None
    page_number: int
    image_path: str
    section_title: str = ""
    caption: str = ""
    tags: List[str] = Field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    
    related_chunk_ids: List[str] = Field(default_factory=list)
    
    ocr_text: str = ""
    ocr_quality: float = 1.0
    diagram_confidence: float = 1.0
    
    embedding_model: str
    created_at: datetime = Field(default_factory=_utcnow)

class RetrievalScores(BaseModel):
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    confidence_score: float = 0.0

class ImageExplainability(BaseModel):
    retrieval_reason: List[str] = Field(default_factory=list)
    clip_score: float = 0.0
    association_score: float = 0.0
    subsystem_match: bool = False
    source_pdf: str = ""
    page: int = 0
    section_title: str = ""
    final_score: float = 0.0

class RetrievedImage(BaseModel):
    metadata: ImageMetadata
    explainability: ImageExplainability

class RetrievalResult(BaseModel):
    chunk: TextChunk
    scores: RetrievalScores
    images: List[RetrievedImage] = Field(default_factory=list)

class IngestionManifestEntry(BaseModel):
    status: str
    processed_date: datetime
    chunk_count: int
    image_count: int
    errors: List[str] = Field(default_factory=list)

class ChatMessage(BaseModel):
    role: str
    content: str
    images: List[str] = Field(default_factory=list)

class AgentState(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    current_intent: Optional[QueryIntent] = None
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list)
    final_answer: str = ""
    error: Optional[str] = None

class BenchmarkQuery(BaseModel):
    query_id: str
    query_text: str
    intent: QueryIntent
    expected_manual: str = ""
    expected_page: Optional[int] = None
    expected_chunk_ids: List[str] = Field(default_factory=list)
    expected_image_id: Optional[str] = None

class QueryEvalResult(BaseModel):
    query_id: str
    query_text: str
    intent: QueryIntent
    text_metrics: dict = Field(default_factory=dict)
    image_metrics: dict = Field(default_factory=dict)
    grounding_metrics: dict = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

class EvaluationReport(BaseModel):
    timestamp: datetime = Field(default_factory=_utcnow)
    benchmark_version: str = "v1.0"
    total_queries: int = 0
    metrics: dict = Field(default_factory=dict)
    per_query_results: List[QueryEvalResult] = Field(default_factory=list)
    failure_analysis: List[dict] = Field(default_factory=list)

