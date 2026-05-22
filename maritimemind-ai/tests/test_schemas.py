import pytest
from pydantic import ValidationError
from datetime import datetime
from app.models.schemas import (
    BoundingBox,
    TextChunk,
    ImageMetadata,
    RetrievalScores,
    RetrievalResult,
    QueryIntent,
    IngestionManifestEntry,
    ChatMessage,
    AgentState
)

def test_bounding_box_valid():
    bbox = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0)
    assert bbox.x0 == 10.0
    assert bbox.y1 == 200.0

def test_text_chunk_valid():
    chunk = TextChunk(
        chunk_id="chunk1",
        manual_name="test_manual",
        department="deck",
        page_number=1,
        section_title="Introduction",
        content="This is a test chunk.",
        embedding_model="test-model"
    )
    assert chunk.chunk_id == "chunk1"
    assert isinstance(chunk.created_at, datetime)
    assert chunk.keywords == []

def test_text_chunk_missing_required():
    with pytest.raises(ValidationError):
        TextChunk(chunk_id="chunk1")  # Missing manual_name, etc.

def test_image_metadata_valid():
    img = ImageMetadata(
        image_id="img1",
        manual_name="test_manual",
        page_number=1,
        image_path="/data/img1.png",
        caption="Test image",
        embedding_model="test-model"
    )
    assert img.image_id == "img1"

def test_query_intent_enum():
    intent = QueryIntent.EXPLANATION
    assert intent.value == "EXPLANATION"

def test_agent_state_default():
    state = AgentState()
    assert len(state.messages) == 0
    assert state.current_intent is None
    assert len(state.retrieved_chunks) == 0
