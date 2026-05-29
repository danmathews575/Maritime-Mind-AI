"""
Quality Review Agent — Hardened Hallucination Detection

Validates generated responses for:
1. Minimum length and structure
2. Citation presence
3. Numeric specification hallucination (values not in context)
4. Technical specification hallucination (units, pressures, temperatures)
5. Part number / model number hallucination
6. Procedure step count verification
7. Maritime-specific safety warning coverage
"""
from app.utils.logger import setup_logger
import re
from typing import List
from app.agents.state import AgentState

logger = setup_logger("maritimemind.agents.quality_reviewer")

MIN_RESPONSE_LENGTH = 50

# Pattern for maritime specifications with units
SPEC_PATTERN = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:bar|psi|°C|°F|rpm|kW|MW|kPa|MPa|m³|l/min|m/s|kg|'
    r'tonnes?|knots?|nm|nautical\s+miles?|litres?|liters?|gallons?|mm|cm|'
    r'meters?|metres?|amp(?:ere)?s?|volts?|watts?|hertz|Hz)\b',
    re.IGNORECASE
)

# Pattern for technical model/part numbers
MODEL_NUMBER_PATTERN = re.compile(
    r'\b[A-Z]{2,}[\s\-]?\d{2,}[A-Z0-9\-]*\b'
)

# Pattern for numbered procedure steps
STEP_PATTERN = re.compile(r'^\s*\d+[.)]\s', re.MULTILINE)


def _check_hallucination_indicators(response_text: str, text_results: list) -> List[str]:
    """
    Multi-layer hallucination detection for maritime responses.
    Returns a list of suspicious findings (empty = no issues detected).
    """
    if not text_results:
        return []

    context_text = " ".join([res.chunk.content for res in text_results])
    context_lower = context_text.lower()
    suspicious = []

    # ── 1. Numeric hallucination (original, improved) ──────────────────────
    numbers_in_response = set(re.findall(r'\b\d+(?:\.\d+)?\b', response_text))
    for num_str in numbers_in_response:
        try:
            val = float(num_str)
            # Skip small common numbers (list items, page refs, source citations)
            if val > 10 and num_str not in context_text:
                suspicious.append(f"Unverifiable number: {num_str}")
        except ValueError:
            pass

    # ── 2. Technical specification hallucination ──────────────────────────
    specs_in_response = SPEC_PATTERN.findall(response_text)
    for spec in specs_in_response:
        spec_normalized = spec.strip().lower()
        # Check if the spec value appears in context (fuzzy: allow slight spacing differences)
        if spec_normalized not in context_lower:
            # Try without spaces around the number
            spec_compact = re.sub(r'\s+', '', spec_normalized)
            context_compact = re.sub(r'\s+', '', context_lower)
            if spec_compact not in context_compact:
                suspicious.append(f"Unverifiable specification: {spec.strip()}")

    # ── 3. Part number / model number hallucination ──────────────────────
    models_in_response = MODEL_NUMBER_PATTERN.findall(response_text)
    for model in models_in_response:
        # Skip common patterns like "Source 1", "Page 24", "Step 3"
        if re.match(r'^(Source|Page|Step|Image|Figure|Table|Section)\s*\d+$', model, re.IGNORECASE):
            continue
        if model not in context_text:
            suspicious.append(f"Unverifiable model/part number: {model}")

    # ── 4. Procedure step count verification ──────────────────────────────
    response_steps = STEP_PATTERN.findall(response_text)
    context_steps = STEP_PATTERN.findall(context_text)
    if len(response_steps) > len(context_steps) + 2:
        suspicious.append(
            f"Response has {len(response_steps)} steps but context has "
            f"{len(context_steps)} — possible fabricated steps"
        )

    # ── 5. Safety warning coverage ────────────────────────────────────────
    # If context contains WARNING/CAUTION/DANGER but response doesn't mention them
    context_has_warning = bool(re.search(
        r'\b(WARNING|CAUTION|DANGER)\b', context_text, re.IGNORECASE
    ))
    response_has_warning = bool(re.search(
        r'(WARNING|CAUTION|DANGER|⚠️)', response_text, re.IGNORECASE
    ))
    if context_has_warning and not response_has_warning:
        suspicious.append("Context contains safety warnings but response omits them")

    if suspicious:
        logger.warning(
            f"Hallucination indicators detected ({len(suspicious)}): "
            f"{'; '.join(suspicious[:5])}"
        )

    return suspicious


def quality_review_agent(state: AgentState) -> AgentState:
    """
    Quality Review Agent: Validates the generated response for completeness,
    citations, and multi-layer hallucination indicators.
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

    # 2. Citation reference check
    has_source_ref = bool(re.search(
        r'(source\s*\[?\d|page\s*\d|manual|\[source)', response_text, re.IGNORECASE
    ))

    # 3. Handling "I don't have" / refusal responses
    has_cant_answer = any(
        phrase in response_text.lower()
        for phrase in [
            "insufficient information",
            "i cannot find",
            "not contain sufficient",
            "unable to find",
            "not found in available",
        ]
    )

    if not has_cant_answer and not has_source_ref and text_results:
        passed = False
        notes.append("Response is substantive but lacks source references.")

    # 4. Empty citations with substantive response
    if not citations and not has_cant_answer and len(response_text) > MIN_RESPONSE_LENGTH:
        passed = False
        notes.append("Substantive response provided without any citations.")

    # 5. Multi-layer hallucination check
    if not has_cant_answer and text_results:
        hallucination_findings = _check_hallucination_indicators(
            response_text, text_results
        )
        if hallucination_findings:
            passed = False
            # Include first 3 findings in quality notes
            finding_summary = "; ".join(hallucination_findings[:3])
            if len(hallucination_findings) > 3:
                finding_summary += f" (+{len(hallucination_findings) - 3} more)"
            notes.append(f"Potential hallucination: {finding_summary}")

    state["quality_passed"] = passed
    state["quality_notes"] = " | ".join(notes) if notes else "Quality checks passed."

    if not passed:
        logger.warning(f"Quality check failed: {state['quality_notes']}")
        if retry_count < max_retries:
            logger.info(
                f"Incrementing retry count for quality loop-back "
                f"({retry_count} -> {retry_count + 1})"
            )
            state["retry_count"] = retry_count + 1
        else:
            logger.error("Max retries reached. Returning degraded response.")
    else:
        logger.info("Quality checks passed successfully.")

    return state
