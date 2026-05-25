import pytest
from app.agents.state import AgentState
from app.agents.router import context_router_agent
from app.models.schemas import QueryIntent

def test_router_explanation():
    state: AgentState = {"query": "What is the function of the main engine?", "conversation_history": []}
    new_state = context_router_agent(state)
    assert new_state["intent"] == QueryIntent.EXPLANATION
    assert new_state["retrieval_strategy"] == "text_only"

def test_router_troubleshooting():
    state: AgentState = {"query": "Why is the exhaust temperature alarm going off?", "conversation_history": []}
    new_state = context_router_agent(state)
    assert new_state["intent"] == QueryIntent.TROUBLESHOOTING
    assert new_state["retrieval_strategy"] == "multimodal"

def test_router_diagram():
    state: AgentState = {"query": "Show me the diagram for the cooling system.", "conversation_history": []}
    new_state = context_router_agent(state)
    assert new_state["intent"] == QueryIntent.DIAGRAM_REQUEST
    assert new_state["retrieval_strategy"] == "image_priority"

def test_router_emergency():
    state: AgentState = {"query": "Fire in the engine room!", "conversation_history": []}
    new_state = context_router_agent(state)
    assert new_state["intent"] == QueryIntent.EMERGENCY
    assert new_state["retrieval_strategy"] == "emergency"
