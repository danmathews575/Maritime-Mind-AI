from app.utils.logger import setup_logger
from app.agents.state import AgentState
from app.models.schemas import QueryIntent
from app.retrieval.query_classifier import QueryClassifier

logger = setup_logger("maritimemind.agents.router")

# Initialize the classifier and expander once
_classifier = QueryClassifier()
from app.memory.query_expander import QueryExpander
_expander = QueryExpander()

def context_router_agent(state: AgentState) -> AgentState:
    """
    Router Agent: Determines the intent of the user's query and 
    assigns the appropriate retrieval strategy.
    
    Hardened routing logic:
    - PROCEDURE → multimodal (procedures reference diagrams)
    - TROUBLESHOOTING → multimodal + diagnosis routing
    - DIAGRAM_REQUEST → image_priority
    - EMERGENCY → emergency (fast-path, no retries)
    - SOP_LOOKUP → text_only
    - EXPLANATION → text_only
    
    Args:
        state: The current AgentState.
        
    Returns:
        Updated AgentState with intent, retrieval_strategy, and metadata hints set.
    """
    query = state.get("query", "")
    history = state.get("conversation_history", [])
    
    # Expand query using history
    expanded_query = _expander.expand(query, history)
    if expanded_query != query:
        logger.info(f"Router Agent analyzing expanded query: '{expanded_query}'")
        # Update state with expanded query so retrieval uses it
        state["query"] = expanded_query
        query = expanded_query
    else:
        logger.info(f"Router Agent analyzing query: '{query}'")
    
    # Classify with full metadata extraction
    classification = _classifier.classify(query)
    intent = classification.intent
    logger.info(
        f"Classified intent: {intent.name} | "
        f"Department: {classification.department_hint or 'none'} | "
        f"Subsystem: {classification.subsystem_hint or 'none'}"
    )
    
    # Determine retrieval strategy based on intent
    # HARDENED: Procedures are now multimodal because maintenance procedures
    # routinely reference "See Figure X" diagrams, piping schematics, etc.
    strategy = "text_only"

    if intent == QueryIntent.EXPLANATION:
        strategy = "text_only"
    elif intent == QueryIntent.PROCEDURE:
        strategy = "multimodal"  # Procedures often reference diagrams
    elif intent == QueryIntent.TROUBLESHOOTING:
        strategy = "multimodal"
        logger.info("Routing to structured diagnosis workflow.")
        state["next_agent"] = "diagnosis"
    elif intent == QueryIntent.DIAGRAM_REQUEST:
        strategy = "image_priority"
    elif intent == QueryIntent.EMERGENCY:
        strategy = "emergency"
    elif intent == QueryIntent.SOP_LOOKUP:
        strategy = "text_only"
    
    logger.info(f"Assigned retrieval strategy: {strategy}")
    
    # Update state with classification results
    state["intent"] = intent
    state["retrieval_strategy"] = strategy
    
    # Store metadata hints for downstream retrieval filtering
    if classification.department_hint:
        state["department_hint"] = classification.department_hint
    if classification.subsystem_hint:
        state["subsystem_hint"] = classification.subsystem_hint
    
    # Initialize some state fields if they aren't already
    if "text_results" not in state:
        state["text_results"] = []
    if "image_results" not in state:
        state["image_results"] = []
    if "retry_count" not in state:
        state["retry_count"] = 0
    if "max_retries" not in state:
        # Emergency queries get no retries — speed is critical
        state["max_retries"] = 0 if intent == QueryIntent.EMERGENCY else 2
        
    return state
