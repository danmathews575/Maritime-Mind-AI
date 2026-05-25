import logging
import re
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

MIN_RESPONSE_LENGTH = 50

def _check_hallucination_indicators(response_text: str, text_results: list) -> bool:
    """
    Simple heuristic to check if numeric values in the response 
    are actually present in the retrieved context.
    Returns True if suspicious (potential hallucination).
    """
    # Find all numbers in the response
    numbers_in_response = set(re.findall(r'\b\d+(?:\.\d+)?\b', response_text))
    
    # Collect all text from chunks
    context_text = " ".join([res.chunk.content for res in text_results])
    
    # If a number is in response but not in context, it might be hallucinated
    # (Excluding small common numbers like 1, 2, 3 which might be list items)
    suspicious_numbers = []
    for num_str in numbers_in_response:
        try:
            val = float(num_str)
            if val > 10 and num_str not in context_text:
                suspicious_numbers.append(num_str)
        except ValueError:
            pass
            
    if suspicious_numbers:
        logger.warning(f"Hallucination indicator: Response contains numbers not in context: {suspicious_numbers}")
        return True
        
    return False

def quality_review_agent(state: AgentState) -> AgentState:
    """
    Quality Review Agent: Validates the generated response for completeness,
    citations, and hallucination indicators.
    """
    response_text = state.get("response_text", "")
    citations = state.get("citations", [])
    text_results = state.get("text_results", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    logger.info("Quality Review Agent checking response.")
    
    passed = True
    notes = []
    
    # 1. Length check
    if len(response_text.strip()) < MIN_RESPONSE_LENGTH:
        passed = False
        notes.append("Response is too short.")
        
    # 2. Citation reference check (Response should mention "Source" or "Page" or "Manual")
    has_source_ref = bool(re.search(r'(source|page|manual)', response_text, re.IGNORECASE))
    
    # 3. Handling "I don't have"
    has_cant_answer = "insufficient information" in response_text.lower() or "i cannot find" in response_text.lower()
    
    if not has_cant_answer and not has_source_ref and text_results:
        passed = False
        notes.append("Response is substantive but lacks source references.")
        
    # 4. Empty citations with substantive response
    if not citations and not has_cant_answer and len(response_text) > MIN_RESPONSE_LENGTH:
        passed = False
        notes.append("Substantive response provided without any citations.")
        
    # 5. Hallucination heuristic
    if not has_cant_answer and _check_hallucination_indicators(response_text, text_results):
        passed = False
        notes.append("Potential hallucination detected (unverifiable numeric claims).")
        
    state["quality_passed"] = passed
    state["quality_notes"] = " | ".join(notes) if notes else "Quality checks passed."
    
    if not passed:
        logger.warning(f"Quality check failed: {state['quality_notes']}")
        if retry_count < max_retries:
            logger.info(f"Incrementing retry count for quality loop-back ({retry_count} -> {retry_count + 1})")
            state["retry_count"] = retry_count + 1
        else:
            logger.error("Max retries reached. Returning degraded response.")
    else:
        logger.info("Quality checks passed successfully.")
        
    return state
