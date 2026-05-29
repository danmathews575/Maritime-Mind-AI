import re
from typing import List
from app.models.schemas import ChatMessage
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.memory.query_expander")

class QueryExpander:
    """
    Expands queries using conversation context to resolve references.
    Hardened heuristic implementation:
    - Replaces vague pronouns with exact nouns from previous turns rather
      than naively appending "(Context: ...)".
    - Avoids calling LLMs to maintain low latency for the default path.
    """
    
    def __init__(self):
        # Pronouns/references indicating need for context
        self.ref_regex = re.compile(
            r"\b(that|it|this|those|these|same|he|she|they)\b", 
            re.IGNORECASE
        )
        
        # Simple stop words to strip when extracting subjects
        self.stop_words = {
            "what", "is", "the", "a", "an", "how", "to", "do", "i", 
            "can", "you", "tell", "me", "about", "show", "explain"
        }

    def expand(self, query: str, history: List[ChatMessage]) -> str:
        """
        Expand the query if it contains reference words and history is available.
        Uses heuristic noun extraction from history to replace pronouns.
        """
        if not history:
            return query
            
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
                
        if not last_user_query:
            return query

        # Heuristic entity extraction: grab the most "substantial" words from last query
        words = last_user_query.split()
        entities = []
        
        # Look for quoted phrases or capitalized terms (often subsystems/parts)
        quoted = re.findall(r'"([^"]*)"', last_user_query)
        if quoted:
            entities.extend(quoted)
            
        # Or just grab longer, non-stop words (likely nouns/subsystems)
        if not entities:
            candidates = [
                w.strip(".,?!\"'") for w in words 
                if w.lower() not in self.stop_words and len(w) > 4
            ]
            if candidates:
                # Take the last few substantive words (usually the object of the sentence)
                entities.append(" ".join(candidates[-2:]))
                
        if entities:
            # We found a likely subject from the previous turn
            subject = entities[-1]
            
            # Simple replacement: "how do I fix it" -> "how do I fix [subject]"
            # This is cleaner than appending context and doesn't confuse the embedder
            expanded = self.ref_regex.sub(subject, query, count=1)
            
            # If no replacement happened (e.g., it was just a short fragment without pronouns)
            if expanded == query:
                expanded = f"{query} {subject}"
                
            logger.info(f"Query expanded heuristically: '{query}' -> '{expanded}'")
            return expanded
            
        return query
