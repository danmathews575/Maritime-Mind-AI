import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END
from app.agents.state import AgentState

# Agent imports
from app.agents.router import context_router_agent
from app.agents.visual_specialist import visual_specialist_agent
from app.agents.verification import retrieval_verification_agent
from app.agents.synthesizer import response_synthesis_agent
from app.agents.quality_reviewer import quality_review_agent

# Text Retrieval components
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.reranker import RerankerService
from app.retrieval.scoring import ConfidenceScorer
from app.services.vector_store import VectorStoreService
from app.services.bm25_index import BM25IndexService
from app.services.embedding import TextEmbeddingService
from app.configs.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize text retrieval services for the graph node
try:
    _vs = VectorStoreService()
    _bm25 = BM25IndexService()
    if not _bm25.is_built:
        _bm25.load()
    _embedder = TextEmbeddingService()
    _hybrid = HybridSearchEngine(_vs, _bm25, _embedder)
    _reranker = RerankerService()
    _scorer = ConfidenceScorer(
        bm25_weight=0.3 if settings.RERANKING_ENABLED else 0.5,
        vector_weight=0.4 if settings.RERANKING_ENABLED else 0.5,
        rerank_weight=0.3 if settings.RERANKING_ENABLED else 0.0
    )
except Exception as e:
    logger.error(f"Failed to initialize text retrieval services: {e}")

def text_retrieval_node(state: AgentState) -> AgentState:
    """Dedicated node for text retrieval to allow graph orchestration."""
    query = state.get("query", "")
    logger.info(f"Text Retrieval Node executing for query: '{query}'")
    
    top_k = settings.TOP_K_RESULTS
    
    # Optional query expansion based on retry
    if state.get("retry_count", 0) > 0:
        logger.info("Retry detected, relaxing constraints (mock implementation).")
        # Could implement actual query rewriting here if needed
        
    results = _hybrid.search(query, top_k=top_k * 2)
    
    if results and settings.RERANKING_ENABLED:
        results = _reranker.rerank(query, results, top_n=top_k * 2)
        
    if results:
        results = _scorer.compute(results)
        results.sort(key=lambda r: r.scores.confidence_score, reverse=True)
        results = _scorer.apply_threshold(results, threshold=settings.CONFIDENCE_THRESHOLD)
        results = results[:top_k]
        state["retrieval_confidence"] = results[0].scores.confidence_score if results else 0.0
    else:
        state["retrieval_confidence"] = 0.0
        
    state["text_results"] = results
    logger.info(f"Text Retrieval Node found {len(results)} chunks.")
    return state


# Conditional Routing Functions
def route_from_router(state: AgentState) -> str:
    """Route after intent classification."""
    strategy = state.get("retrieval_strategy")
    if strategy == "image_priority":
        return "visual_specialist"
    return "text_retrieval"

def route_from_visual(state: AgentState) -> str:
    """Route after visual specialist (if it ran first)."""
    # If we haven't done text retrieval yet, do it.
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
    """Route after retrieval verification."""
    passed = state.get("verification_passed", True)
    if passed:
        return "synthesizer"
    
    # Failed verification
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "text_retrieval"  # Loop back
    return "synthesizer"  # Proceed with warning

def route_from_quality(state: AgentState) -> str:
    """Route after quality review."""
    passed = state.get("quality_passed", True)
    if passed:
        return END
        
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "verification"  # Loop back to verification (or could go to retrieval)
        
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
    
    # Define Edges
    workflow.add_edge(START, "context_router")
    
    workflow.add_conditional_edges(
        "context_router",
        route_from_router,
        {
            "visual_specialist": "visual_specialist",
            "text_retrieval": "text_retrieval"
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
