from app.utils.logger import setup_logger
from typing import List, Dict, Any
from app.agents.state import AgentState
from app.services.llm_service import LLMService

logger = setup_logger("maritimemind.agents.synthesizer")

# Initialize LLM Service once
_llm = LLMService()

def _format_context(text_results: List[Any]) -> str:
    """Format retrieval results into context blocks."""
    blocks = []
    for idx, res in enumerate(text_results):
        chunk = res.chunk
        block = f"Source [{idx+1}]: {chunk.manual_name}, Page: {chunk.page_number}\n"
        block += f"Content:\n{chunk.content}\n"
        blocks.append(block)
    return "\n".join(blocks)

def _extract_citations(text_results: List[Any]) -> List[Dict[str, Any]]:
    """Extract metadata for citations."""
    citations = []
    for res in text_results:
        chunk = res.chunk
        citations.append({
            "manual_name": chunk.manual_name,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id
        })
    return citations

def response_synthesis_agent(state: AgentState) -> AgentState:
    """
    Response Synthesis Agent: Constructs the final response using an LLM,
    grounded in the retrieved text and image context.
    """
    query = state.get("query", "")
    text_results = state.get("text_results", [])
    image_results = state.get("image_results", [])
    
    logger.info("Response Synthesis Agent generating answer.")
    
    # 1. Format Context
    context_blocks = _format_context(text_results)
    if not context_blocks:
        context_blocks = "No relevant context found."
        
    # Append image info if available
    if image_results:
        image_context = "\nAvailable Diagrams/Images:\n"
        for idx, img in enumerate(image_results):
            image_context += f"Image [{idx+1}]: {img.caption} (Source: {img.manual_name}, Page: {img.page_number})\n"
        context_blocks += image_context

    # 2. Construct Prompt
    system_prompt = (
        "You are MaritimeMind AI, a maritime technical assistant. Answer the user's question\n"
        "based ONLY on the following retrieved context. Do not introduce information not found\n"
        "in the context. If the context is insufficient, say so clearly."
    )
    
    history_context = ""
    history = state.get("conversation_history", [])
    if history:
        history_context = "## Conversation History:\n"
        for msg in history[-4:]:  # Include last 4 messages for context
            history_context += f"{msg.role.capitalize()}: {msg.content}\n"
        history_context += "\n"
    
    prompt = (
        f"## Retrieved Context:\n{context_blocks}\n\n"
        f"{history_context}"
        f"## User Question:\n{query}\n\n"
        "## Instructions:\n"
        "- Answer precisely and technically\n"
        "- Reference source manual and page numbers in your answer\n"
        "- If diagrams are available, mention them by their Image number, but DO NOT attempt to embed markdown images like ![]()\n"
        "- The system will automatically display the images to the user alongside your text\n"
        "- If you cannot find sufficient information, say: \"Insufficient information in \n"
        "  the available manuals to fully answer this question.\"\n"
        "- Format procedures as numbered steps\n"
        "- Highlight safety warnings prominently\n\n"
        "## Answer:\n"
    )

    # 3. Handle low confidence bypass
    if not state.get("verification_passed", True):
        # We failed verification but reached max retries. We should tell the LLM to be cautious.
        prompt = (
            "WARNING: The retrieved context has low confidence and may not fully answer the question.\n"
            "State this clearly before answering.\n\n"
        ) + prompt

    # 4. Generate response
    provider = state.get("llm_provider")
    logger.info(f"Generating response using provider: {provider or 'default'}")
    response_text = _llm.generate(prompt=prompt, system_prompt=system_prompt, provider=provider)
    
    # 5. Update citations and Images
    citations = _extract_citations(text_results)
    
    state["response_text"] = response_text
    state["citations"] = citations
    # attached_images is already set by visual_specialist, but ensure it's not lost
    if "attached_images" not in state:
        state["attached_images"] = []
        
    logger.info("Response synthesis complete.")
    return state
