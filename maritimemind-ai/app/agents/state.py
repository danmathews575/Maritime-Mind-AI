from typing import TypedDict, List, Optional, Dict, Any
from app.models.schemas import QueryIntent, ChatMessage, RetrievalResult, ImageMetadata

class AgentState(TypedDict):
    """
    State schema for the LangGraph multi-agent orchestration.
    Shared across all agents in the graph.
    """
    # Input
    query: str
    conversation_history: List[ChatMessage]
    
    # Routing
    intent: Optional[QueryIntent]
    retrieval_strategy: str  # e.g., "text_only", "image_priority", "multimodal", "emergency"
    llm_provider: Optional[str]
    
    # Retrieval
    text_results: List[RetrievalResult]
    image_results: List[ImageMetadata]
    retrieval_confidence: float
    
    # Verification
    verification_passed: bool
    verification_notes: str
    
    # Memory tracking
    session_id: str
    
    # Synthesis
    response_text: str
    
    # Diagnostic state tracking
    active_fault_tree: str
    current_diagnosis_node: str
    citations: List[Dict[str, Any]]
    attached_images: List[str]  # file paths of attached images
    
    # Quality
    quality_passed: bool
    quality_notes: str
    
    # Control
    next_agent: Optional[str]
    retry_count: int
    max_retries: int
    error: Optional[str]
