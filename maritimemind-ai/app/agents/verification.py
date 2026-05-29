from app.utils.logger import setup_logger
from app.agents.state import AgentState
from app.models.schemas import QueryIntent
from app.configs.config import get_settings

logger = setup_logger("maritimemind.agents.verification")

def retrieval_verification_agent(state: AgentState) -> AgentState:
    """
    Retrieval Verification Agent: Validates the quality and completeness of retrieved context.
    Determines if the system should proceed to synthesis or attempt a retry.
    """
    settings = get_settings()
    
    text_results = state.get("text_results", [])
    image_results = state.get("image_results", [])
    intent = state.get("intent")
    retrieval_confidence = state.get("retrieval_confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    logger.info("Retrieval Verification Agent checking results.")
    
    passed = True
    notes = []
    
    # Rule 1 & 2: General confidence and existence
    if not text_results:
        passed = False
        notes.append("No text results retrieved.")
    elif retrieval_confidence < settings.CONFIDENCE_THRESHOLD:
        passed = False
        notes.append(f"Retrieval confidence ({retrieval_confidence:.2f}) is below threshold ({settings.CONFIDENCE_THRESHOLD}).")
        
    # Rule 3: Visual intent requirements
    if intent == QueryIntent.DIAGRAM_REQUEST and not image_results:
        passed = False
        notes.append("Diagram request intent, but no images retrieved.")
        
    # Rule 4: Emergency threshold
    if intent == QueryIntent.EMERGENCY and retrieval_confidence < 0.4:
        passed = False
        notes.append(f"Emergency intent requires high confidence, got {retrieval_confidence:.2f}.")

    state["verification_passed"] = passed
    state["verification_notes"] = " | ".join(notes) if notes else "All checks passed."
    
    if not passed:
        logger.warning(f"Verification failed: {state['verification_notes']}")
        if retry_count < max_retries:
            logger.info(f"Incrementing retry count ({retry_count} -> {retry_count + 1})")
            state["retry_count"] = retry_count + 1
            # The orchestrator will route back to retrieval or modify query
        else:
            logger.error("Max retries reached. Proceeding to synthesis with warning.")
    else:
        logger.info("Verification passed successfully.")
        
    return state
