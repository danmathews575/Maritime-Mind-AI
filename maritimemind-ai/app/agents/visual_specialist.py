from app.utils.logger import setup_logger
from app.agents.state import AgentState
from app.services.vector_store import VectorStoreService
from app.services.clip_embedding import ImageEmbeddingService
from app.retrieval.image_retrieval import ImageRetrievalService

logger = setup_logger("maritimemind.agents.visual_specialist")

# Initialize services once to avoid reloading models
_vs = VectorStoreService()
_clip = ImageEmbeddingService()
_image_retrieval = ImageRetrievalService(_vs, _clip)

def visual_specialist_agent(state: AgentState) -> AgentState:
    """
    Visual Specialist Agent: Retrieves relevant diagrams and images
    using cross-modal CLIP search and text association.
    
    Args:
        state: The current AgentState.
        
    Returns:
        Updated AgentState with image_results and attached_images set.
    """
    query = state.get("query", "")
    text_results = state.get("text_results", [])
    
    logger.info(f"Visual Specialist Agent executing for query: '{query}'")
    
    # Check if we should skip based on intent
    if state.get("retrieval_strategy") == "text_only":
        logger.info("Skipping image retrieval due to text_only strategy.")
        state["image_results"] = []
        state["attached_images"] = []
        return state
    
    try:
        # Retrieve images
        retrieved_images = _image_retrieval.search(query, text_results=text_results, top_k=5)
        
        # Extract metadata and file paths
        image_results = [img.metadata for img in retrieved_images]
        attached_images = [img.metadata.image_path for img in retrieved_images]
        
        logger.info(f"Visual Specialist retrieved {len(image_results)} images.")
        
        state["image_results"] = image_results
        state["attached_images"] = attached_images
        
    except Exception as e:
        logger.error(f"Visual Specialist Agent failed: {e}")
        state["error"] = f"Image retrieval failed: {str(e)}"
        
    return state
