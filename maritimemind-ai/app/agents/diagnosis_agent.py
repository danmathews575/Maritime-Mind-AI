from app.utils.logger import setup_logger
from app.agents.state import AgentState
from app.diagnosis.symptom_extractor import SymptomExtractor
from app.diagnosis.knowledge_base import KnowledgeBase
from app.diagnosis.guided_workflow import workflow_manager
from app.services.llm_service import LLMService

logger = setup_logger("maritimemind.agents.diagnosis")
_llm = LLMService()

def diagnosis_agent(state: AgentState) -> AgentState:
    """
    LangGraph node for interactive fault diagnosis.
    Manages the state of the fault tree and queries the LLM to format responses.
    """
    logger.info("Executing DiagnosisAgent...")
    
    query = state.get("query", "")
    session_id = state.get("session_id", "default_session")
    
    # Check if there's an active diagnosis session
    diag_session = workflow_manager.get_session(session_id)
    
    if not diag_session:
        # 1. Start a new diagnosis session
        logger.info(f"No active diagnosis session for {session_id}. Attempting to start one.")
        extracted = SymptomExtractor.extract(query)
        tree = KnowledgeBase.find_best_tree(extracted["symptoms"], extracted["subsystems"], extracted["alarms"])
        
        if not tree:
            logger.info("No matching fault tree found. Falling back to general explanation.")
            # We could route this back or just handle it gracefully.
            # For now, we will handle it by providing a general troubleshooting response using standard retrieval.
            # We signal the graph to just proceed to synthesis (bypass structured diagnosis).
            # We don't have a direct edge to synthesis from here yet, but we can set a flag or just answer directly.
            state["response_text"] = "I couldn't identify a specific diagnostic workflow for this issue. Let me search the manuals for general troubleshooting advice."
            # Set intent to PROCEDURE to force standard retrieval on next step, or just let synthesizer handle it.
            # The router edge logic will need to handle this.
            return state
            
        logger.info(f"Matched fault tree: {tree.name}")
        diag_session = workflow_manager.start_session(session_id, tree.tree_id, tree.root_node_id)
        current_node = tree.get_node(diag_session.current_node_id)
        
        # We also want to retrieve context for this specific step. 
        # We will set the query in the state so the retrieval agent can find context if needed,
        # but diagnosis agent will format the immediate response.
        
        prompt = f"""You are a marine engineering diagnostic assistant aboard a vessel.
You are starting a structured troubleshooting workflow for: {tree.name}.

ABSOLUTE RULES:
1. Use ONLY the procedure steps defined in the fault tree. Do NOT add, modify, or invent diagnostic steps.
2. If any step involves safety-critical systems (fuel oil, high pressure, high temperature, electrical), include appropriate ⚠️ WARNING markers.
3. Do NOT suggest actions beyond what is stated in this step.
4. Do NOT speculate about the root cause — follow the structured diagnostic flow.

The first diagnostic step is:
"{current_node.instruction}"

Format this clearly for the user. Present ONLY this single step. Ask the question directly.
If possible answers are defined, list them for the user.
Possible responses: {list(current_node.possible_answers) if current_node.possible_answers else 'Open response'}
"""
        response = _llm.generate(prompt)
        state["response_text"] = response
        state["quality_passed"] = True  # Bypass quality reviewer for interactive steps
        
    else:
        # 2. Continue existing session
        logger.info(f"Continuing diagnosis session for {session_id}. Current node: {diag_session.current_node_id}")
        tree = KnowledgeBase.get_tree(diag_session.tree_id)
        current_node = tree.get_node(diag_session.current_node_id)
        
        # Use LLM to evaluate user's response against possible answers
        eval_prompt = f"""You are evaluating a marine engineer's response during a structured troubleshooting workflow aboard a vessel.
The diagnostic step asked: "{current_node.instruction}"
The engineer replied: "{query}"

The possible diagnostic branches are: {list(current_node.next_nodes.keys())}.

Rules:
1. Map the engineer's response to the MOST CLOSELY matching branch.
2. Reply with ONLY the exact name of the branch (one of: {list(current_node.next_nodes.keys())}).
3. If the response is clearly ambiguous or unrelated, reply with exactly "UNKNOWN".
4. Do NOT add explanation or commentary.
"""
        evaluation = _llm.generate(eval_prompt).strip()
        logger.info(f"LLM evaluated user response as: {evaluation}")
        
        if evaluation in current_node.next_nodes:
            # Advance session
            next_node_id = current_node.next_nodes[evaluation]
            diag_session.advance(next_node_id, query)
            next_node = tree.get_node(next_node_id)
            
            if next_node.action_type in ["resolution", "terminal_unresolved"]:
                diag_session.is_complete = True
                workflow_manager.clear_session(session_id)
                prompt = f"""You are a marine engineering diagnostic assistant aboard a vessel.
The diagnostic workflow is concluding with a {'resolution' if next_node.action_type == 'resolution' else 'recommendation for further investigation'}.

Final diagnostic conclusion:
"{next_node.instruction}"

ABSOLUTE RULES:
1. Present ONLY what is stated in the conclusion above. Do NOT invent additional repair steps.
2. If this involves safety-critical work, include appropriate ⚠️ WARNING markers.
3. Remind the user to log the fault and action taken in the engine room logbook.
4. If the action type is 'terminal_unresolved', clearly state that this requires further engineering team intervention and should not be attempted without proper authorization.

Format the conclusion clearly for the user.
"""
            else:
                prompt = f"""You are a marine engineering diagnostic assistant aboard a vessel.
The engineer responded '{evaluation}' to the previous step. Moving to the next diagnostic step.

Next diagnostic step:
"{next_node.instruction}"

ABSOLUTE RULES:
1. Present ONLY this single step. Do NOT add diagnostic steps not in the workflow.
2. If this step involves safety-critical systems, include appropriate ⚠️ WARNING markers.
3. Ask the question or instruction directly and clearly.
4. If possible answers are defined, list them for the user.
Possible responses: {list(next_node.possible_answers) if next_node.possible_answers else 'Open response'}
"""
            response = _llm.generate(prompt)
            state["response_text"] = response
            state["quality_passed"] = True
            
        else:
            # Could not map answer
            response = f"I didn't quite understand that. Please let me know based on the previous step: {current_node.instruction}"
            state["response_text"] = response
            state["quality_passed"] = True

    return state
