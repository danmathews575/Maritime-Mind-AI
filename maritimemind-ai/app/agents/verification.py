"""
Retrieval Verification Agent — Hardened Gating Logic

Validates retrieved context quality before allowing synthesis.
Key improvements:
- Absolute confidence gating (works with new absolute scoring)
- Hard refusal gate for genuinely irrelevant retrievals
- Emergency fast-path validation
- Visual intent enforcement
"""
from app.utils.logger import setup_logger
from app.agents.state import AgentState
from app.models.schemas import QueryIntent
from app.configs.config import get_settings

logger = setup_logger("maritimemind.agents.verification")

# Absolute confidence floor below which synthesis is refused
HARD_REFUSAL_THRESHOLD = 0.30


def retrieval_verification_agent(state: AgentState) -> AgentState:
    """
    Retrieval Verification Agent: Validates the quality and completeness
    of retrieved context. Determines if the system should:
    - Proceed to synthesis (passed)
    - Retry retrieval with modified parameters (retry)
    - Refuse to answer (hard gate)
    """
    settings = get_settings()

    text_results = state.get("text_results", [])
    image_results = state.get("image_results", [])
    intent = state.get("intent")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    logger.info("Retrieval Verification Agent checking results.")

    passed = True
    notes = []

    # ── Compute retrieval confidence from actual results ──────────────────
    if text_results:
        max_confidence = max(r.scores.confidence_score for r in text_results)
        avg_confidence = sum(r.scores.confidence_score for r in text_results) / len(text_results)
    else:
        max_confidence = 0.0
        avg_confidence = 0.0

    state["retrieval_confidence"] = max_confidence

    # ── Rule 1: No results at all ────────────────────────────────────────
    if not text_results:
        passed = False
        notes.append("No text results retrieved.")

    # ── Rule 2: Hard refusal gate (absolute confidence) ──────────────────
    elif max_confidence < HARD_REFUSAL_THRESHOLD:
        passed = False
        state["should_refuse"] = True
        notes.append(
            f"Retrieval confidence ({max_confidence:.2f}) below hard refusal "
            f"threshold ({HARD_REFUSAL_THRESHOLD}). Answer would be unreliable."
        )
        logger.warning(
            f"HARD REFUSAL GATE triggered: max_confidence={max_confidence:.2f}, "
            f"avg={avg_confidence:.2f}"
        )

    # ── Rule 3: Soft confidence check ────────────────────────────────────
    elif max_confidence < settings.CONFIDENCE_THRESHOLD:
        passed = False
        notes.append(
            f"Retrieval confidence ({max_confidence:.2f}) below threshold "
            f"({settings.CONFIDENCE_THRESHOLD}). Retry may improve results."
        )

    # ── Rule 4: Visual intent requirements ───────────────────────────────
    if intent == QueryIntent.DIAGRAM_REQUEST and not image_results:
        # Don't fail hard — still provide text context
        notes.append("Diagram request intent, but no images retrieved. Text context available.")
        logger.info("Diagram request with no images — proceeding with text context only.")

    # ── Rule 5: Emergency validation ─────────────────────────────────────
    if intent == QueryIntent.EMERGENCY:
        if max_confidence < 0.4:
            notes.append(
                f"Emergency intent with low confidence ({max_confidence:.2f}). "
                "Proceeding but flagging uncertainty."
            )
            # Don't block emergency responses — always provide what we have
            # but flag the uncertainty for the synthesizer
            state["verification_passed"] = True
            state["verification_notes"] = " | ".join(notes) if notes else "All checks passed."
            logger.warning("Emergency query with low confidence — proceeding with caution flag.")
            return state

    # ── Rule 6: Procedure completeness ───────────────────────────────────
    if intent == QueryIntent.PROCEDURE and text_results:
        procedure_chunks = [r for r in text_results if r.chunk.contains_procedure]
        if not procedure_chunks:
            notes.append(
                "Procedure intent but no procedure-flagged chunks retrieved. "
                "Results may not contain step-by-step instructions."
            )

    state["verification_passed"] = passed
    state["verification_notes"] = " | ".join(notes) if notes else "All checks passed."

    if not passed and not state.get("should_refuse", False):
        logger.warning(f"Verification failed: {state['verification_notes']}")
        if retry_count < max_retries:
            logger.info(
                f"Incrementing retry count ({retry_count} -> {retry_count + 1})"
            )
            state["retry_count"] = retry_count + 1
        else:
            logger.error("Max retries reached. Proceeding to synthesis with warning.")
            # After max retries, proceed but flag low confidence
            state["verification_passed"] = True
    else:
        logger.info("Verification passed successfully.")

    return state
