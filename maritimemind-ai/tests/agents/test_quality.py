import pytest
from app.agents.state import AgentState
from app.agents.quality_reviewer import quality_review_agent
from app.models.schemas import TextChunk, RetrievalResult, RetrievalScores

@pytest.fixture
def dummy_state():
    chunk = TextChunk(
        chunk_id="chunk-1",
        manual_name="Engine Manual",
        department="Engineering",
        section_title="Cooling",
        page_number=10,
        content="The cooling system uses sea water. Max temp is 95C.",
        embedding_model="test"
    )
    result = RetrievalResult(
        chunk=chunk,
        scores=RetrievalScores(confidence_score=0.9)
    )
    
    return {
        "response_text": "The cooling system uses sea water (Source: Engine Manual).",
        "citations": [{"manual_name": "Engine Manual"}],
        "text_results": [result],
        "retry_count": 0,
        "max_retries": 2
    }

def test_quality_pass(dummy_state):
    dummy_state["response_text"] = "The cooling system uses sea water, and its maximum temperature is 95C. This is according to the source manual."
    new_state = quality_review_agent(dummy_state)
    assert new_state["quality_passed"] is True

def test_quality_too_short(dummy_state):
    dummy_state["response_text"] = "It uses water."
    new_state = quality_review_agent(dummy_state)
    assert new_state["quality_passed"] is False
    assert "too short" in new_state["quality_notes"]

def test_quality_hallucination(dummy_state):
    # Number 150 is not in context
    dummy_state["response_text"] = "The cooling system uses sea water and the pressure is 150 bar. (Source: Engine Manual)."
    new_state = quality_review_agent(dummy_state)
    assert new_state["quality_passed"] is False
    assert "hallucination" in new_state["quality_notes"]
