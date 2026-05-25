"""
app/api/routes/sessions.py
===========================
Session management endpoints.

Depends on the ConversationMemoryService singleton injected via FastAPI DI.
"""
import logging
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    SessionCreateResponse,
    SessionHistoryResponse,
    ChatMessageOut,
)
from app.memory.conversation_memory import memory_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

# Singleton memory service
_memory = memory_service



@router.post("", response_model=SessionCreateResponse, status_code=201,
             summary="Create a new conversation session")
async def create_session() -> SessionCreateResponse:
    """Creates a new session and returns its ID."""
    session_id = _memory.create_session()
    session = _memory.get_session(session_id)
    return SessionCreateResponse(
        session_id=session_id,
        created_at=session.created_at,
        message_count=0,
    )


@router.get("/{session_id}/history", response_model=SessionHistoryResponse,
            summary="Retrieve conversation history for a session")
async def get_session_history(session_id: str, max_messages: int = 20) -> SessionHistoryResponse:
    """Returns the last `max_messages` messages for the given session."""
    session = _memory.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    messages = _memory.get_history(session_id, max_messages=max_messages)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[ChatMessageOut(role=m.role, content=m.content, images=m.images)
                  for m in messages],
    )


@router.delete("/{session_id}", status_code=204, summary="Clear a session's history")
async def clear_session(session_id: str) -> None:
    """Clears all messages from the given session (does not delete the session itself)."""
    session = _memory.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    _memory.clear_session(session_id)
