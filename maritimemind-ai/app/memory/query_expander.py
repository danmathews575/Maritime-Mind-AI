import re
import logging
from typing import List
from app.memory.conversation_memory import ChatMessage

logger = logging.getLogger(__name__)

class QueryExpander:
    """Expands queries using conversation context to resolve references."""
    
    def __init__(self):
        # Reference words that indicate the need for context
        self.reference_patterns = [
            r"\b(that)\b",
            r"\b(it)\b",
            r"\b(this)\b",
            r"\b(those)\b",
            r"\b(these)\b",
            r"\b(same)\b",
            r"\b(he)\b",
            r"\b(she)\b",
            r"\b(they)\b"
        ]
        self.ref_regex = re.compile("|".join(self.reference_patterns), re.IGNORECASE)

    def expand(self, query: str, history: List[ChatMessage]) -> str:
        """
        Expand the query if it contains reference words and history is available.
        """
        if not history:
            return query
            
        # Check if query contains references
        needs_expansion = bool(self.ref_regex.search(query))
        
        # Simple heuristic: if it's very short, it might be a fragment
        if len(query.split()) <= 3:
            needs_expansion = True
            
        if not needs_expansion:
            return query
            
        logger.info(f"Query expansion triggered for query: '{query}'")
            
        # Find the last substantive user query
        last_user_query = None
        for msg in reversed(history):
            if msg.role == "user" and len(msg.content.split()) > 3:
                last_user_query = msg.content
                break
                
        if last_user_query:
            expanded = f"{query} (Context: {last_user_query})"
            logger.info(f"Expanded query: '{expanded}'")
            return expanded
            
        return query
