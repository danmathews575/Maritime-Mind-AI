import pytest
from unittest.mock import Mock, patch
from app.retrieval.reranker import RerankerService
from app.models.schemas import RetrievalResult, RetrievalScores, TextChunk
from app.configs.config import settings

def _create_result(content: str):
    chunk = TextChunk(chunk_id="test", manual_name="test_manual", department="test", page_number=1, section_title="test", content=content, embedding_model="test")
    scores = RetrievalScores(bm25_score=1.0, vector_score=1.0, rerank_score=0.0)
    return RetrievalResult(chunk=chunk, scores=scores)

@patch("app.retrieval.reranker.CrossEncoder")
def test_reranking(mock_cross_encoder):
    mock_model = Mock()
    # Assume 1st is less relevant than 2nd
    mock_model.predict.return_value = [0.1, 0.9]
    mock_cross_encoder.return_value = mock_model
    
    settings.RERANKING_ENABLED = True
    reranker = RerankerService()
    
    results = [
        _create_result("irrelevant text"),
        _create_result("highly relevant text")
    ]
    
    reranked = reranker.rerank("some query", results)
    
    assert len(reranked) == 2
    # The second one should now be first
    assert reranked[0].chunk.content == "highly relevant text"
    assert reranked[0].scores.rerank_score == 0.9
    assert reranked[1].chunk.content == "irrelevant text"
    assert reranked[1].scores.rerank_score == 0.1

def test_reranking_disabled():
    settings.RERANKING_ENABLED = False
    reranker = RerankerService()
    
    results = [
        _create_result("irrelevant text"),
        _create_result("highly relevant text")
    ]
    
    reranked = reranker.rerank("some query", results)
    
    # Order should be unchanged
    assert reranked[0].chunk.content == "irrelevant text"
    assert reranked[1].chunk.content == "highly relevant text"
