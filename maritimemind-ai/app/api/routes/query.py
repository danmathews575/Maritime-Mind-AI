"""
app/api/routes/query.py
========================
Core query endpoint — the heart of the MaritimeMind AI API.

POST /api/v1/query
  - Accepts a natural-language question + optional session_id
  - Runs the LangGraph agent workflow (Phase 7)
  - Saves the exchange to conversation memory (Phase 8)
  - Returns a structured multimodal response (text + citations + image URLs)
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse, CitationOut, ImageOut
from app.configs.config import get_settings
from app.memory.conversation_memory import memory_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["query"])
settings = get_settings()

# Shared memory service
_memory = memory_service



def run_agent_workflow(query: str, history: list, provider: Optional[str] = None) -> dict:
    """Thin wrapper around the LangGraph workflow — isolated here so tests can patch it."""
    from app.orchestration.graph import run_agent_workflow as _run
    return _run(query=query, history=history, provider=provider)


def _build_image_url(image_path: str, manual_name: str) -> str:
    """Convert a local file path to a static HTTP URL."""
    import os
    filename = os.path.basename(image_path)
    # Streamlit/browsers will URL-encode spaces automatically
    return f"/static/images/{manual_name}/{filename}"


@router.post("/query", response_model=QueryResponse, summary="Submit a maritime question")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Main query endpoint.

    1. Loads conversation history from session (if session_id provided)
    2. Runs the LangGraph agent workflow
    3. Stores the Q&A exchange back in the session
    4. Returns structured response with text, citations, and image URLs
    """
    # --- Resolve session ---
    session_id: Optional[str] = request.session_id
    history = []
    if session_id:
        session = _memory.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        history = _memory.get_history(session_id, max_messages=10)

    # --- Run agent graph ---
    try:
        final_state = run_agent_workflow(query=request.query, history=history, provider=request.provider)
    except Exception as e:
        logger.exception(f"Agent workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(e)}")

    # --- Extract results from final state ---
    answer: str = final_state.get("response_text", "")
    if not answer:
        answer = "I was unable to generate a response. Please try rephrasing your question."

    # Build citations
    citations_raw = final_state.get("citations", [])
    citations = []
    for c in citations_raw:
        citations.append(CitationOut(
            manual_name=c.get("manual_name", ""),
            page_number=c.get("page_number", 0),
            section_title=c.get("section_title", ""),
            chunk_id=c.get("chunk_id", ""),
        ))

    # Build image URLs
    image_paths = final_state.get("attached_images", [])
    image_results = final_state.get("image_results", [])
    images = []
    for img_meta in image_results:
        images.append(ImageOut(
            image_id=img_meta.image_id,
            url=_build_image_url(img_meta.image_path, img_meta.manual_name),
            caption=img_meta.caption,
            page_number=img_meta.page_number,
            manual_name=img_meta.manual_name,
        ))

    confidence = final_state.get("retrieval_confidence", 0.0)
    intent_val = final_state.get("intent")
    intent_str = intent_val.value if intent_val else ""
    quality_passed = final_state.get("quality_passed", True)
    quality_notes = final_state.get("quality_notes", "")

    # --- Persist to memory ---
    if session_id:
        _memory.add_message(session_id, "user", request.query)
        _memory.add_message(
            session_id, "assistant", answer,
            images=[img.url for img in images],
        )

    return QueryResponse(
        answer=answer,
        citations=citations,
        images=images,
        confidence=confidence,
        intent=intent_str,
        session_id=session_id,
        quality_passed=quality_passed,
        quality_notes=quality_notes,
    )
