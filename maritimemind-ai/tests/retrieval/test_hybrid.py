import pytest
from unittest.mock import Mock
from app.retrieval.hybrid_search import HybridSearchEngine
from app.models.schemas import TextChunk

def test_rrf_fusion():
    engine = HybridSearchEngine(Mock(), Mock(), Mock())
    
    bm25_results = [("chunk1", 2.5), ("chunk2", 1.8), ("chunk3", 1.0)]
    vector_results = [("chunk2", 0.1), ("chunk3", 0.2), ("chunk4", 0.5)]
    
    k = 60
    fused = engine._rrf_fusion(bm25_results, vector_results, k=k)
    
    assert "chunk1" in fused
    assert "chunk2" in fused
    assert "chunk3" in fused
    assert "chunk4" in fused
    
    # chunk2 is rank 2 in bm25 and rank 1 in vector
    expected_chunk2 = (1.0 / (k + 2)) + (1.0 / (k + 1))
    assert abs(fused["chunk2"] - expected_chunk2) < 1e-6
    
    # chunk1 is rank 1 in bm25 only
    expected_chunk1 = (1.0 / (k + 1))
    assert abs(fused["chunk1"] - expected_chunk1) < 1e-6

def test_rrf_fusion_empty():
    engine = HybridSearchEngine(Mock(), Mock(), Mock())
    fused = engine._rrf_fusion([], [], k=60)
    assert fused == {}

def test_build_retrieval_results():
    vs_mock = Mock()
    chunk = TextChunk(chunk_id="chunk1", manual_name="test_manual", department="deck", page_number=1, section_title="test", content="test content", embedding_model="test")
    vs_mock.get_text_chunks_by_ids.return_value = [chunk]
    
    engine = HybridSearchEngine(vs_mock, Mock(), Mock())
    
    fused_ids = ["chunk1"]
    fused_scores = {"chunk1": 0.05}
    bm25_results = [("chunk1", 2.0)]
    vector_results = [("chunk1", 0.1)]
    
    results = engine._build_retrieval_results(fused_ids, fused_scores, bm25_results, vector_results)
    
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk1"
    assert results[0].scores.bm25_score == 2.0
    assert results[0].scores.vector_score == 0.1
    assert results[0].scores.final_score == 0.05
