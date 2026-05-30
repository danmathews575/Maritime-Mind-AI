import uuid
import time
import json
from typing import List, Optional
import redis
from pydantic import BaseModel, Field

from app.models.schemas import ChatMessage
from app.configs.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.memory")

class Session(BaseModel):
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    last_accessed_at: float = Field(default_factory=time.time)

class ConversationMemoryService:
    def __init__(self):
        self.redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            self._use_redis = True
            logger.info(f"Connected to Redis at {self.redis_url}")
        except redis.ConnectionError as e:
            logger.warning(f"Could not connect to Redis: {e}. Falling back to in-memory dictionary.")
            self._use_redis = False
            self.sessions = {}
        
        self.ttl_seconds = 24 * 3600  # 24 hours

    def _get_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def _save_session(self, session: Session):
        if self._use_redis:
            try:
                self.redis_client.setex(
                    self._get_key(session.session_id),
                    self.ttl_seconds,
                    session.model_dump_json()
                )
            except Exception as e:
                logger.error(f"Error saving session to Redis: {e}")
        else:
            self.sessions[session.session_id] = session

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session and return the session ID."""
        if not session_id:
            session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        self._save_session(session)
        return session_id
        
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        if self._use_redis:
            try:
                data = self.redis_client.get(self._get_key(session_id))
                if data:
                    session = Session.model_validate_json(data)
                    session.last_accessed_at = time.time()
                    self._save_session(session) # Update TTL
                    return session
            except Exception as e:
                logger.error(f"Error retrieving session from Redis: {e}")
            return None
        else:
            session = self.sessions.get(session_id)
            if session:
                session.last_accessed_at = time.time()
            return session

    def add_message(self, session_id: str, role: str, content: str, images: List[str] = None):
        """Add a message to a session."""
        session = self.get_session(session_id)
        if not session:
            session = Session(session_id=session_id)
            
        message = ChatMessage(
            role=role,
            content=content,
            images=images or []
        )
        session.messages.append(message)
        session.last_accessed_at = time.time()
        self._save_session(session)
        
    def get_history(self, session_id: str, max_messages: int = 10) -> List[ChatMessage]:
        """Get the most recent messages for a session."""
        session = self.get_session(session_id)
        if not session:
            return []
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
        if self._use_redis:
            self.redis_client.delete(self._get_key(session_id))
        else:
            if session_id in self.sessions:
                self.sessions[session_id].messages = []
            
    def cleanup_expired(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours (Fallback only). Redis handles TTL natively."""
        if not self._use_redis:
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
