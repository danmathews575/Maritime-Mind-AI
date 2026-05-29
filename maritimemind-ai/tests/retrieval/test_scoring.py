import pytest
from app.retrieval.scoring import ConfidenceScorer
from app.models.schemas import RetrievalResult, RetrievalScores, TextChunk

def _create_result(bm25, vec, rerank=0.0):
    chunk = TextChunk(chunk_id="test", manual_name="test_manual", department="test", page_number=1, section_title="test", content="test", embedding_model="test")
    scores = RetrievalScores(bm25_score=bm25, vector_score=vec, rerank_score=rerank)
    return RetrievalResult(chunk=chunk, scores=scores)

def test_normalize_bm25():
    scorer = ConfidenceScorer()
    assert scorer._normalize_bm25([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert scorer._normalize_bm25([5.0, 5.0]) == [0.5, 0.5]
    assert scorer._normalize_bm25([]) == []

def test_compute_scores():
    scorer = ConfidenceScorer(bm25_weight=0.3, vector_weight=0.4, rerank_weight=0.3)
    
    results = [
        _create_result(bm25=10.0, vec=0.1, rerank=0.8), # Best
        _create_result(bm25=5.0, vec=0.5, rerank=-0.4), # Worse
    ]
    
    scored = scorer.compute(results)
    
    # Absolute scoring means scores are derived directly, not min-max scaled to [0, 1]
    assert scored[0].scores.confidence_score > scored[1].scores.confidence_score
    assert 0.0 <= scored[0].scores.confidence_score <= 1.0
    assert 0.0 <= scored[1].scores.confidence_score <= 1.0

def test_apply_threshold():
    scorer = ConfidenceScorer()
    results = [
        _create_result(0, 0.5), # Normal distance
        _create_result(0, 1.8)  # High distance (> 1.5 hard floor)
    ]
    results[0].scores.confidence_score = 0.8
    results[1].scores.confidence_score = 0.8 # Same confidence, but should be hard floored
    
    filtered = scorer.apply_threshold(results, 0.5)
    
    assert len(filtered) == 1
    assert filtered[0].scores.vector_score == 0.5
