import uuid
import time
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from app.models.schemas import ChatMessage

class Session(BaseModel):
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    last_accessed_at: float = Field(default_factory=time.time)

class ConversationMemoryService:
    def __init__(self):
        # In-memory storage for single-user scenarios
        self.sessions: Dict[str, Session] = {}
        
    def create_session(self) -> str:
        """Create a new session and return the session ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(session_id=session_id)
        return session_id
        
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session.last_accessed_at = time.time()
        return session

    def add_message(self, session_id: str, role: str, content: str, images: List[str] = None):
        """Add a message to a session."""
        session = self.get_session(session_id)
        if not session:
            # Auto-create if not found (optional, but convenient)
            session = Session(session_id=session_id)
            self.sessions[session_id] = session
            
        message = ChatMessage(
            role=role,
            content=content,
            images=images or []
        )
        session.messages.append(message)
        session.last_accessed_at = time.time()
        
    def get_history(self, session_id: str, max_messages: int = 10) -> List[ChatMessage]:
        """Get the most recent messages for a session."""
        session = self.get_session(session_id)
        if not session:
            return []
        
        # Return the last 'max_messages'
        return session.messages[-max_messages:]
        
    def get_context_summary(self, session_id: str) -> str:
        """Get a text summary of recent context for query expansion."""
        history = self.get_history(session_id, max_messages=4)
        if not history:
            return ""
            
        summary = []
        for msg in history:
            summary.append(f"{msg.role.capitalize()}: {msg.content}")
        return "\n".join(summary)
        
    def clear_session(self, session_id: str):
        """Clear the history of a session."""
        if session_id in self.sessions:
            self.sessions[session_id].messages = []
            
    def cleanup_expired(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        expired_keys = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_accessed_at > max_age_seconds
        ]
        
        for key in expired_keys:
            del self.sessions[key]

# Singleton instance for the FastAPI app
memory_service = ConversationMemoryService()
