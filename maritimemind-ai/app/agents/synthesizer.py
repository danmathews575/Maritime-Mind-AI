from app.utils.logger import setup_logger
from typing import List, Dict, Any
from app.agents.state import AgentState
from app.services.llm_service import LLMService
from app.utils.language import detect_language, get_language_name

logger = setup_logger("maritimemind.agents.synthesizer")

# Initialize LLM Service once
_llm = LLMService()

# ─── Hardened Maritime System Prompt ───────────────────────────────────────────

MARITIME_SYSTEM_PROMPT = """You are MaritimeMind AI, a professional maritime technical intelligence assistant deployed aboard vessels and in maritime operations centers.

ABSOLUTE RULES — VIOLATIONS ARE UNACCEPTABLE:
1. You MUST answer ONLY from the retrieved context provided below. Do not use prior knowledge or training data.
2. You MUST cite every factual claim with [Source N] where N matches the source number in the context.
3. If the retrieved context does not contain sufficient information, you MUST state:
   "The available manuals do not contain sufficient information to fully answer this question."
4. You MUST NOT invent procedures, operational steps, specifications, part numbers, pressure values, temperatures, or safety warnings.
5. You MUST NOT generate operational instructions that are not explicitly present in the retrieved context.
6. You MUST NOT fabricate subsystem relationships, component names, or diagnostic logic.

SAFETY RULES:
- If context contains WARNING, CAUTION, or DANGER content, display it prominently using ⚠️ markers BEFORE the related procedure step.
- NEVER omit safety warnings present in the source context.
- For emergency procedures, lead with IMMEDIATE ACTION items.

RESPONSE STRUCTURE:
- For PROCEDURES: Use numbered steps exactly as they appear in the source. Include ALL steps — do not summarize, merge, or skip any.
- For TROUBLESHOOTING: Present the diagnostic flow as given in the manual. Preserve the causal chain.
- For EXPLANATIONS: Provide the technical explanation with subsystem context and operating parameters from the source.
- For EMERGENCIES: Lead with immediate action items. Highlight time-critical steps. Include muster/notification requirements.
- For SOP/COMPLIANCE: Quote regulatory references exactly as stated in the source.

CITATION FORMAT:
- Inline: Place [Source N] after each factual claim or procedure step.
- At the end: List all referenced sources as:
  "Sources: [N] Manual Name, Page X, Section: Title"

DIAGRAMS AND IMAGES:
- If diagrams or images are listed in the context, reference them as "See Image [N]" where appropriate.
- Do NOT embed images using markdown syntax like ![](). The system displays them automatically.
- Explain what the diagram shows if its caption or context provides that information.

UNCERTAINTY AND PARTIAL INFORMATION:
- If context only partially addresses the question, explicitly state what IS covered and what requires additional manual consultation.
- Distinguish between "confirmed from manual" and "not found in available sources."
- Do NOT fill gaps with plausible-sounding technical content.

GROUNDING VERIFICATION:
Before including ANY fact in your answer, verify: Is this fact explicitly stated in the Sources above?
- If YES → Include it with a [Source N] citation.
- If NO → Do not include it. Do not guess or infer.
- If PARTIALLY → State what the source says and note what is uncertain."""


def _format_context(text_results: List[Any]) -> str:
    """
    Format retrieval results into enriched context blocks with metadata flags.
    Provides the LLM with structural signals to prioritize safety-critical content
    and distinguish procedures from explanations.
    """
    blocks = []
    for idx, res in enumerate(text_results):
        chunk = res.chunk

        # Build metadata flag string
        flags = []
        if chunk.contains_procedure:
            flags.append("PROCEDURE")
        if chunk.contains_warning:
            flags.append("⚠️ SAFETY WARNING")
        if chunk.contains_emergency_workflow:
            flags.append("🚨 EMERGENCY")
        if chunk.contains_diagram_reference:
            flags.append("📊 DIAGRAM REF")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        block = f"Source [{idx+1}]{flag_str}:\n"
        block += f"  Manual: {chunk.manual_name}\n"
        if chunk.ship_id:
            block += f"  Ship Context: {chunk.ship_id}\n"
        block += f"  Page: {chunk.page_number}\n"
        block += f"  Section: {chunk.section_title}\n"
        block += f"  Subsystem: {chunk.subsystem}\n"
        block += f"  Department: {chunk.department}\n"
        block += f"  Confidence: {res.scores.confidence_score:.2f}\n"
        block += f"  Content:\n{chunk.content}\n"
        blocks.append(block)

    return "\n---\n".join(blocks)


def _format_image_context(image_results: List[Any]) -> str:
    """Format image metadata into context blocks with enriched information."""
    if not image_results:
        return ""

    lines = ["\n## Available Diagrams/Images:\n"]
    for idx, img in enumerate(image_results):
        caption = img.caption if hasattr(img, "caption") else ""
        manual = img.manual_name if hasattr(img, "manual_name") else ""
        page = img.page_number if hasattr(img, "page_number") else ""
        section = img.section_title if hasattr(img, "section_title") else ""
        ocr = img.ocr_text if hasattr(img, "ocr_text") and img.ocr_text else ""
        tags = ", ".join(img.tags) if hasattr(img, "tags") and img.tags else ""

        line = f"Image [{idx+1}]: {caption}"
        line += f" (Source: {manual}, Page: {page}"
        if section:
            line += f", Section: {section}"
        line += ")"
        if tags:
            line += f"\n  Tags: {tags}"
        if ocr:
            line += f"\n  Diagram Labels: {ocr}"
        lines.append(line)

    return "\n".join(lines)


def _extract_citations(text_results: List[Any]) -> List[Dict[str, Any]]:
    """Extract metadata for citations."""
    citations = []
    for res in text_results:
        chunk = res.chunk
        citations.append({
            "manual_name": chunk.manual_name,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "section_title": chunk.section_title,
            "subsystem": chunk.subsystem,
            "ship_id": chunk.ship_id or "",
        })
    return citations


def response_synthesis_agent(state: AgentState) -> AgentState:
    """
    Response Synthesis Agent: Constructs the final response using an LLM,
    grounded in the retrieved text and image context.
    Uses hardened prompting for maritime operational intelligence.
    """
    query = state.get("query", "")
    text_results = state.get("text_results", [])
    image_results = state.get("image_results", [])
    
    logger.info("Response Synthesis Agent generating answer.")
    
    # ── Check for low-confidence refusal ──────────────────────────────────────
    if state.get("should_refuse", False):
        max_conf = 0.0
        searched_manuals = set()
        if text_results:
            max_conf = max(r.scores.confidence_score for r in text_results)
            searched_manuals = {r.chunk.manual_name for r in text_results[:5]}

        state["response_text"] = (
            "I was unable to find sufficiently relevant information in the "
            "available manuals to answer this question reliably.\n\n"
            f"**Query understood as**: {query}\n"
            f"**Manuals searched**: {', '.join(searched_manuals) if searched_manuals else 'N/A'}\n"
            f"**Best match confidence**: {max_conf:.0%}\n\n"
            "Please try rephrasing your question or specifying the subsystem/manual."
        )
        state["citations"] = []
        if "attached_images" not in state:
            state["attached_images"] = []
        logger.info("Low-confidence refusal issued.")
        return state

    # ── Format Context ────────────────────────────────────────────────────────
    context_blocks = _format_context(text_results)
    if not context_blocks:
        context_blocks = "No relevant context found in the available manuals."
        
    # Append image info if available
    image_context = _format_image_context(image_results)
    if image_context:
        context_blocks += image_context

    # ── Build Conversation History ────────────────────────────────────────────
    history_context = ""
    history = state.get("conversation_history", [])
    if history:
        history_context = "## Recent Conversation Context:\n"
        for msg in history[-4:]:  # Include last 4 messages for context
            history_context += f"{msg.role.capitalize()}: {msg.content}\n"
        history_context += "\n"
    
    # ── Construct User Prompt ─────────────────────────────────────────────────
    prompt = (
        f"## Retrieved Context:\n{context_blocks}\n\n"
        f"{history_context}"
        f"## User Question:\n{query}\n\n"
        "## Instructions:\n"
        "- Follow ALL rules from your system prompt above.\n"
        "- Cite every claim with [Source N].\n"
        "- If diagrams are available, reference them as Image [N].\n"
        "- The system will automatically display images — do NOT use markdown image syntax.\n"
        "- Format procedures as numbered steps preserving ALL original steps.\n"
        "- Highlight safety warnings with ⚠️ before the relevant step.\n"
        "- If information is insufficient, state so clearly rather than guessing.\n\n"
    )

    # ── Multilingual response instruction ─────────────────────────────────────
    detected_lang = state.get("detected_language", "en")
    if detected_lang and detected_lang != "en":
        lang_name = get_language_name(detected_lang)
        prompt += (
            f"## LANGUAGE RULE (MANDATORY):\n"
            f"The user's query is written in **{lang_name}** ({detected_lang}).\n"
            f"You MUST write your ENTIRE response in **{lang_name}**.\n"
            f"The retrieved context may be in English — translate your answer into {lang_name}.\n"
            f"Keep technical terms, part numbers, and [Source N] citations in their original form.\n\n"
        )
        logger.info(f"Multilingual response: answering in {lang_name} ({detected_lang})")

    prompt += "## Answer:\n"

    # ── Handle low confidence warning ─────────────────────────────────────────
    if not state.get("verification_passed", True):
        prompt = (
            "⚠️ SYSTEM NOTE: The retrieved context has LOW CONFIDENCE and may not "
            "fully answer the question. You MUST state this uncertainty clearly at "
            "the beginning of your response. Do NOT compensate by generating "
            "information not found in the context.\n\n"
        ) + prompt

    # ── Generate response ─────────────────────────────────────────────────────
    provider = state.get("llm_provider")
    logger.info(f"Generating response using provider: {provider or 'default'}")
    response_text = _llm.generate(
        prompt=prompt,
        system_prompt=MARITIME_SYSTEM_PROMPT,
        provider=provider,
    )
    
    # ── Update state ──────────────────────────────────────────────────────────
    citations = _extract_citations(text_results)
    
    state["response_text"] = response_text
    state["citations"] = citations
    # attached_images is already set by visual_specialist, but ensure it's not lost
    if "attached_images" not in state:
        state["attached_images"] = []
        
    logger.info("Response synthesis complete.")
    return state
