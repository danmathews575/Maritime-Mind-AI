import pytest
from unittest.mock import Mock, patch
from app.retrieval.controller import RetrievalController
from app.models.schemas import RetrievalResult, RetrievalScores, TextChunk, QueryIntent
from app.configs.config import settings

def _create_result(cid: str, conf: float):
    chunk = TextChunk(chunk_id=cid, manual_name="test_manual", department="test", page_number=1, section_title="test", content="test", embedding_model="test")
    scores = RetrievalScores(confidence_score=conf)
    return RetrievalResult(chunk=chunk, scores=scores)

@patch("app.retrieval.controller.VectorStoreService")
@patch("app.retrieval.controller.BM25IndexService")
@patch("app.retrieval.controller.TextEmbeddingService")
def test_retrieve_pipeline(mock_emb, mock_bm25, mock_vs):
    # Setup mocks
    mock_bm25_instance = Mock()
    mock_bm25_instance.is_built = True
    mock_bm25.return_value = mock_bm25_instance

    controller = RetrievalController()
    
    # Mock the internal components
    controller.classifier = Mock()
    controller.classifier.classify.return_value = QueryIntent.EXPLANATION
    
    controller.hybrid_search = Mock()
    controller.hybrid_search.search.return_value = [
        _create_result("chunk1", 0.0),
        _create_result("chunk2", 0.0),
        _create_result("chunk3", 0.0)
    ]
    
    controller.reranker = Mock()
    controller.reranker.rerank.return_value = [
        _create_result("chunk2", 0.0),
        _create_result("chunk1", 0.0),
        _create_result("chunk3", 0.0)
    ]
    
    controller.scorer = Mock()
    scored_results = [
        _create_result("chunk2", 0.9),
        _create_result("chunk1", 0.8),
        _create_result("chunk3", 0.2)
    ]
    controller.scorer.compute.return_value = scored_results
    controller.scorer.apply_threshold.return_value = scored_results[:2] # mock threshold behavior
    
    settings.TOP_K_RESULTS = 5
    
    results = controller.retrieve("test query")
    
    # Verify calls
    controller.classifier.classify.assert_called_with("test query")
    controller.hybrid_search.search.assert_called_with("test query", top_k=10, filters=None)
    
    # Verify final results
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk2"
    assert results[0].scores.confidence_score == 0.9
    assert results[1].chunk.chunk_id == "chunk1"
    assert results[1].scores.confidence_score == 0.8
