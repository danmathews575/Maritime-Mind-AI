import pytest
from unittest.mock import patch
from app.agents.state import AgentState
from app.agents.synthesizer import response_synthesis_agent
from app.models.schemas import TextChunk, RetrievalResult, RetrievalScores

@pytest.fixture
def dummy_state():
    chunk = TextChunk(
        chunk_id="chunk-1",
        manual_name="Engine Manual",
        department="Engineering",
        section_title="Cooling",
        page_number=10,
        content="The cooling system uses sea water.",
        embedding_model="test"
    )
    result = RetrievalResult(
        chunk=chunk,
        scores=RetrievalScores(confidence_score=0.9)
    )
    
    return {
        "query": "What does the cooling system use?",
        "text_results": [result],
        "image_results": [],
        "verification_passed": True,
        "conversation_history": []
    }

@patch("app.agents.synthesizer._llm.generate")
def test_synthesizer_citations(mock_generate, dummy_state):
    mock_generate.return_value = "The cooling system uses sea water (Source [1])."
    
    new_state = response_synthesis_agent(dummy_state)
    
    assert new_state["response_text"] == "The cooling system uses sea water (Source [1])."
    assert len(new_state["citations"]) == 1
    assert new_state["citations"][0]["manual_name"] == "Engine Manual"
    assert new_state["citations"][0]["page_number"] == 10
