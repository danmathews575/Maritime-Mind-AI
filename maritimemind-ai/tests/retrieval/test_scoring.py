import pytest
from app.retrieval.scoring import ConfidenceScorer
from app.models.schemas import RetrievalResult, RetrievalScores, TextChunk

def _create_result(bm25, vec, rerank=0.0):
    chunk = TextChunk(chunk_id="test", manual_name="test_manual", department="test", page_number=1, section_title="test", content="test", embedding_model="test")
    scores = RetrievalScores(bm25_score=bm25, vector_score=vec, rerank_score=rerank)
    return RetrievalResult(chunk=chunk, scores=scores)

def test_normalization():
    scorer = ConfidenceScorer()
    assert scorer._normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert scorer._normalize([5.0, 5.0]) == [1.0, 1.0]
    assert scorer._normalize([]) == []

def test_compute_scores():
    scorer = ConfidenceScorer(bm25_weight=0.3, vector_weight=0.4, rerank_weight=0.3)
    
    results = [
        _create_result(bm25=10.0, vec=0.1, rerank=0.8), # Best vector (smallest distance), highest bm25, highest rerank
        _create_result(bm25=5.0, vec=0.5, rerank=0.4),  # Worst vector (largest distance), lowest bm25, lowest rerank
    ]
    
    scored = scorer.compute(results)
    
    assert scored[0].scores.confidence_score == 1.0
    assert scored[1].scores.confidence_score == 0.0

def test_apply_threshold():
    scorer = ConfidenceScorer()
    results = [
        _create_result(0, 0),
        _create_result(0, 0)
    ]
    results[0].scores.confidence_score = 0.8
    results[1].scores.confidence_score = 0.3
    
    filtered = scorer.apply_threshold(results, 0.5)
    
    assert len(filtered) == 1
    assert filtered[0].scores.confidence_score == 0.8
