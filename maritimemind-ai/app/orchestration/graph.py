from app.utils.logger import setup_logger
from typing import Literal

from langgraph.graph import StateGraph, START, END
from app.agents.state import AgentState

# Agent imports
from app.agents.router import context_router_agent
from app.agents.visual_specialist import visual_specialist_agent
from app.agents.verification import retrieval_verification_agent
from app.agents.synthesizer import response_synthesis_agent
from app.agents.quality_reviewer import quality_review_agent
from app.agents.diagnosis_agent import diagnosis_agent

# Text Retrieval components
from app.retrieval.controller import RetrievalController
from app.services.llm_service import LLMService
from app.configs.config import get_settings
from app.models.schemas import QueryIntent

logger = setup_logger("maritimemind.orchestration.graph")
settings = get_settings()

# Initialize services
try:
    _controller = RetrievalController()
    _llm = LLMService()
except Exception as e:
    logger.error(f"Failed to initialize retrieval controller: {e}")

def text_retrieval_node(state: AgentState) -> AgentState:
    """
    Dedicated node for text retrieval orchestration.
    Implements hybrid query expansion on retry:
    - First attempt: direct search with metadata filters
    - Retry: LLM-based query rewriting to handle ambiguity/drift
    """
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    intent = state.get("intent")
    
    logger.info(f"Text Retrieval Node executing for query: '{query}' (Retry: {retry_count})")
    
    # ── HYBRID QUERY REWRITING ON RETRY ─────────────────────────────────────
    if retry_count > 0 and intent != QueryIntent.EMERGENCY:
        logger.info(f"Retry {retry_count} triggered. Using LLM for query rewriting.")
        
        rewrite_prompt = (
            "You are a maritime technical search assistant.\n"
            f"The original query was: '{query}'\n"
            "This query failed to retrieve relevant documents from the engineering manuals.\n"
            "Rewrite this query into a highly specific technical search query using alternative maritime terminology, "
            "expanding acronyms if necessary, and focusing on the core subsystem or procedure.\n"
            "Output ONLY the rewritten query string. Do not include quotes or conversational text."
        )
        
        try:
            rewritten_query = _llm.generate(rewrite_prompt, provider=state.get("llm_provider")).strip()
            # Remove any surrounding quotes the LLM might have added
            rewritten_query = rewritten_query.strip("'\"")
            
            if rewritten_query and rewritten_query != query:
                logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")
                query = rewritten_query
                state["query"] = query  # Update state so visual_specialist and synthesizer use it
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}. Falling back to original query.")
    
    # ── EXECUTE HARDENED RETRIEVAL ──────────────────────────────────────────
    # The controller now handles hybrid search, reranking, context expansion,
    # and absolute confidence scoring internally.
    filters = {}
    if state.get("department_hint"):
        filters["department"] = state.get("department_hint")
        
    results = _controller.retrieve(query, top_k=settings.TOP_K_RESULTS, filters=filters)
    
    state["text_results"] = results
    
    # Retrieval confidence is already set by the verification agent based on results,
    # but we can set a preliminary one here just in case.
    if results:
        state["retrieval_confidence"] = results[0].scores.confidence_score
    else:
        state["retrieval_confidence"] = 0.0
        
    logger.info(f"Text Retrieval Node found {len(results)} chunks.")
    return state


# ── CONDITIONAL ROUTING ──────────────────────────────────────────────────────

def route_from_router(state: AgentState) -> str:
    """Route after intent classification."""
    next_agent = state.get("next_agent")
    logger.info(f"route_from_router: next_agent is '{next_agent}'")
    if next_agent == "diagnosis":
        return "diagnosis"
        
    strategy = state.get("retrieval_strategy")
    if strategy == "image_priority":
        return "visual_specialist"
    return "text_retrieval"

def route_from_visual(state: AgentState) -> str:
    """Route after visual specialist (if it ran first)."""
    if not state.get("text_results"):
        return "text_retrieval"
    return "verification"

def route_from_text(state: AgentState) -> str:
    """Route after text retrieval."""
    strategy = state.get("retrieval_strategy")
    if strategy == "multimodal":
        return "visual_specialist"
    return "verification"

def route_from_verification(state: AgentState) -> str:
    """
    Route after retrieval verification.
    Emergency fast-path: emergency queries bypass retry loops to ensure speed.
    """
    passed = state.get("verification_passed", True)
    should_refuse = state.get("should_refuse", False)
    intent = state.get("intent")
    
    # Hard refusal -> go straight to synthesis for the refusal message
    if should_refuse:
        return "synthesizer"
        
    if passed:
        return "synthesizer"
    
    # Emergency bypass: proceed directly to synthesis even if verification failed (low confidence)
    if intent == QueryIntent.EMERGENCY:
        logger.info("Emergency fast-path: bypassing retrieval retries.")
        return "synthesizer"
        
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "text_retrieval"
    
    return "synthesizer"

def route_from_quality(state: AgentState) -> str:
    """
    Route after quality review.
    Emergency fast-path: emergency queries bypass quality retries to ensure speed.
    """
    passed = state.get("quality_passed", True)
    intent = state.get("intent")
    
    if passed:
        return END
        
    # Emergency bypass
    if intent == QueryIntent.EMERGENCY:
        logger.info("Emergency fast-path: bypassing quality retries.")
        return END
        
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        # We loop all the way back to verification to see if we should re-retrieve
        # Setting retry_count increments it, so verification will route to text_retrieval
        return "verification"
        
    return END

def create_graph() -> StateGraph:
    """Build and compile the LangGraph."""
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("context_router", context_router_agent)
    workflow.add_node("text_retrieval", text_retrieval_node)
    workflow.add_node("visual_specialist", visual_specialist_agent)
    workflow.add_node("verification", retrieval_verification_agent)
    workflow.add_node("synthesizer", response_synthesis_agent)
    workflow.add_node("quality_reviewer", quality_review_agent)
    workflow.add_node("diagnosis", diagnosis_agent)
    
    # Define Edges
    workflow.add_edge(START, "context_router")
    
    workflow.add_conditional_edges(
        "context_router",
        route_from_router,
        {
            "visual_specialist": "visual_specialist",
            "text_retrieval": "text_retrieval",
            "diagnosis": "diagnosis"
        }
    )
    
    workflow.add_conditional_edges(
        "visual_specialist",
        route_from_visual,
        {
            "text_retrieval": "text_retrieval",
            "verification": "verification"
        }
    )
    
    workflow.add_conditional_edges(
        "text_retrieval",
        route_from_text,
        {
            "visual_specialist": "visual_specialist",
            "verification": "verification"
        }
    )
    
    workflow.add_conditional_edges(
        "verification",
        route_from_verification,
        {
            "synthesizer": "synthesizer",
            "text_retrieval": "text_retrieval"
        }
    )
    
    workflow.add_edge("synthesizer", "quality_reviewer")
    workflow.add_edge("diagnosis", END)
    
    workflow.add_conditional_edges(
        "quality_reviewer",
        route_from_quality,
        {
            END: END,
            "verification": "verification"
        }
    )
    
    return workflow.compile()

# Instantiate the compiled graph
app_graph = create_graph()

def run_agent_workflow(query: str, history: list = None, provider: str = None) -> AgentState:
    """Convenience function to execute the graph."""
    if history is None:
        history = []
        
    initial_state = {
        "query": query,
        "conversation_history": history,
        "llm_provider": provider,
        "retry_count": 0,
        "max_retries": 2
    }
    
    return app_graph.invoke(initial_state)
