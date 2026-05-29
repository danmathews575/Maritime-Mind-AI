from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class DiagnosisSession(BaseModel):
    """Tracks the state of an ongoing diagnosis workflow."""
    session_id: str
    tree_id: str
    current_node_id: str
    history: list[dict] = Field(default_factory=list)  # List of {"node_id": str, "user_response": str}
    is_complete: bool = False
    
    def advance(self, next_node_id: str, user_response: str):
        """Advances the session to the next node."""
        self.history.append({
            "node_id": self.current_node_id,
            "user_response": user_response
        })
        self.current_node_id = next_node_id

class GuidedWorkflowManager:
    """Manages the state progression of a diagnostic session."""
    
    def __init__(self):
        # In-memory store for active diagnosis sessions (keyed by conversation session_id)
        # In a real production app, this would use Redis or the database.
        self.active_sessions: Dict[str, DiagnosisSession] = {}
        
    def get_session(self, conversation_id: str) -> Optional[DiagnosisSession]:
        return self.active_sessions.get(conversation_id)
        
    def start_session(self, conversation_id: str, tree_id: str, root_node_id: str) -> DiagnosisSession:
        session = DiagnosisSession(
            session_id=conversation_id,
            tree_id=tree_id,
            current_node_id=root_node_id
        )
        self.active_sessions[conversation_id] = session
        return session
        
    def clear_session(self, conversation_id: str):
        if conversation_id in self.active_sessions:
            del self.active_sessions[conversation_id]

# Singleton instance for the agent to use
workflow_manager = GuidedWorkflowManager()
