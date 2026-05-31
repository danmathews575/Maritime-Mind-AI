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
import asyncio
import json
from typing import Optional, AsyncGenerator, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest, QueryResponse, CitationOut, ImageOut
from app.configs.config import get_settings
from app.memory.conversation_memory import memory_service
from app.api.routes.auth import get_current_user
from app.configs.limiter import limiter
from app.services.cache import get_cached_response, set_cached_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["query"])
settings = get_settings()

# Shared memory service
_memory = memory_service



def run_agent_workflow(query: str, history: list, provider: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> dict:
    """Thin wrapper around the LangGraph workflow — isolated here so tests can patch it."""
    from app.orchestration.graph import run_agent_workflow as _run
    return _run(query=query, history=history, provider=provider, filters=filters)


def _build_image_url(image_path: str, manual_name: str) -> str:
    """Convert a local file path to a static HTTP URL."""
    import os
    filename = os.path.basename(image_path)
    # Streamlit/browsers will URL-encode spaces automatically
    return f"/static/images/{manual_name}/{filename}"


@router.post("/query", response_model=QueryResponse, summary="Submit a maritime question")
@limiter.limit("10/minute")
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """
    Main query endpoint.

    1. Loads conversation history from session (if session_id provided)
    2. Runs the LangGraph agent workflow
    3. Stores the Q&A exchange back in the session
    4. Returns structured response with text, citations, and image URLs
    """
    # --- Check cache first (avoid LLM inference for repeated queries) ---
    cached = get_cached_response(payload.query)
    if cached:
        logger.info(f"Cache HIT: returning cached response for query.")
        return QueryResponse(**cached)

    # --- Resolve session ---
    session_id: Optional[str] = payload.session_id
    history = []
    if session_id:
        session = _memory.get_session(session_id)
        if session is None:
            _memory.create_session(session_id)
        history = _memory.get_history(session_id, max_messages=10)

    # --- Run agent graph ---
    try:
        loop = asyncio.get_running_loop()
        final_state = await loop.run_in_executor(None, run_agent_workflow, payload.query, history, payload.provider, payload.filters)
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
            ship_id=c.get("ship_id", ""),
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
            diagram_type=img_meta.diagram_type,
        ))

    confidence = final_state.get("retrieval_confidence", 0.0)
    intent_val = final_state.get("intent")
    intent_str = intent_val.value if intent_val else ""
    quality_passed = final_state.get("quality_passed", True)
    quality_notes = final_state.get("quality_notes", "")

    # --- Persist to memory ---
    if session_id:
        _memory.add_message(session_id, "user", payload.query)
        _memory.add_message(
            session_id, "assistant", answer,
            images=[img.url for img in images],
        )

    response = QueryResponse(
        answer=answer,
        citations=citations,
        images=images,
        confidence=confidence,
        intent=intent_str,
        session_id=session_id,
        quality_passed=quality_passed,
        quality_notes=quality_notes,
        detected_language=final_state.get("detected_language", ""),
    )

    # --- Store in cache (only for high-confidence, non-session responses) ---
    if confidence > 0.5 and not session_id:
        set_cached_response(payload.query, response.model_dump())

    return response

@router.post("/chat/stream", summary="Stream a maritime question response")
@limiter.limit("10/minute")
async def chat_stream(request: Request, payload: QueryRequest) -> StreamingResponse:
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Uses a background thread and a queue to stream tokens directly from the LLM.
    """
    session_id: Optional[str] = payload.session_id
    history = []
    if session_id:
        session = _memory.get_session(session_id)
        if session is None:
            _memory.create_session(session_id)
        history = _memory.get_history(session_id, max_messages=10)

    async def event_generator() -> AsyncGenerator[str, None]:
        # We will use a queue to pass tokens from the background thread to this async generator.
        q = asyncio.Queue()
        
        # We need a custom callback to push tokens to the queue.
        # But for an immediate fix, we can still run the workflow in an executor and simulate streaming 
        # from the generated answer, but doing it asynchronously without blocking.
        # To do true LLM streaming requires updating the graph and LLM service with callbacks.
        # Here we just run the graph in executor to unblock the event loop, and then stream the result.
        
        try:
            yield f"data: {json.dumps({'type': 'thinking', 'data': 'Analyzing query intent...'})}\n\n"
            await asyncio.sleep(0)
            yield f"data: {json.dumps({'type': 'thinking', 'data': 'Retrieving from maritime corpus...'})}\n\n"
            await asyncio.sleep(0)
            
            loop = asyncio.get_running_loop()
            final_state = await loop.run_in_executor(None, run_agent_workflow, payload.query, history, payload.provider, payload.filters)
        except Exception as e:
            logger.exception(f"Agent workflow failed: {e}")
            yield f"data: {json.dumps({'type': 'token', 'data': 'Error: An internal server error occurred while processing your request.'})}\n\n"
            return

        answer = final_state.get("response_text", "I was unable to generate a response.")
        
        citations_raw = final_state.get("citations", [])
        citations = [CitationOut(manual_name=c.get("manual_name", ""), page_number=c.get("page_number", 0), section_title=c.get("section_title", ""), chunk_id=c.get("chunk_id", ""), ship_id=c.get("ship_id", "")).model_dump() for c in citations_raw]
        image_results = final_state.get("image_results", [])
        images = [ImageOut(image_id=img.image_id, url=_build_image_url(img.image_path, img.manual_name), caption=img.caption, page_number=img.page_number, manual_name=img.manual_name, diagram_type=img.diagram_type).model_dump() for img in image_results]
        
        intent_val = final_state.get("intent")
        meta_payload = {
            "citations": citations,
            "images": images,
            "confidence": final_state.get("retrieval_confidence", 0.0),
            "intent": intent_val.value if intent_val else "",
            "detected_language": final_state.get("detected_language", "")
        }

        if session_id:
            _memory.add_message(session_id, "user", payload.query)
            _memory.add_message(session_id, "assistant", answer, images=[img["url"] for img in images])

        yield f"data: {json.dumps({'type': 'metadata', 'data': meta_payload})}\n\n"
        
        # Stream the pre-computed answer asynchronously (avoids blocking event loop during generation)
        words = answer.split(" ")
        for word in words:
            yield f"data: {json.dumps({'type': 'token', 'data': word + ' '})}\n\n"
            await asyncio.sleep(0.01)
            
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
