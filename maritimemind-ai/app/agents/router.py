import logging
from app.agents.state import AgentState
from app.models.schemas import QueryIntent
from app.retrieval.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)

# Initialize the classifier and expander once
_classifier = QueryClassifier()
from app.memory.query_expander import QueryExpander
_expander = QueryExpander()

def context_router_agent(state: AgentState) -> AgentState:
    """
    Router Agent: Determines the intent of the user's query and 
    assigns the appropriate retrieval strategy.
    
    Args:
        state: The current AgentState.
        
    Returns:
        Updated AgentState with intent and retrieval_strategy set.
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
    
    intent = _classifier.classify(query)
    logger.info(f"Classified intent: {intent.name}")
    
    # Determine retrieval strategy based on intent
    strategy = "text_only"
    if intent == QueryIntent.EXPLANATION:
        strategy = "text_only"
    elif intent == QueryIntent.PROCEDURE:
        strategy = "text_only"
    elif intent == QueryIntent.TROUBLESHOOTING:
        strategy = "multimodal"
    elif intent == QueryIntent.DIAGRAM_REQUEST:
        strategy = "image_priority"
    elif intent == QueryIntent.EMERGENCY:
        strategy = "emergency"
    elif intent == QueryIntent.SOP_LOOKUP:
        strategy = "text_only"
    
    logger.info(f"Assigned retrieval strategy: {strategy}")
    
    # Update state
    state["intent"] = intent
    state["retrieval_strategy"] = strategy
    
    # Initialize some state fields if they aren't already
    if "text_results" not in state:
        state["text_results"] = []
    if "image_results" not in state:
        state["image_results"] = []
    if "retry_count" not in state:
        state["retry_count"] = 0
    if "max_retries" not in state:
        state["max_retries"] = 2
        
    return state
