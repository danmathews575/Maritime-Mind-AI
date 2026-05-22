import pytest
from app.services.bm25_index import BM25IndexService, tokenize
from app.models.schemas import TextChunk

def test_tokenize():
    # Should lowercase, remove stopwords, split by punctuation but keep hyphens
    text = "The quick brown MAN-B&W pump is not working."
    tokens = tokenize(text)
    assert "man-b" in tokens
    assert "pump" in tokens
    assert "working" in tokens
    assert "the" not in tokens  # stopword
    assert "not" in tokens      # kept stopword

def test_bm25_build_and_search():
    svc = BM25IndexService()
    
    chunks = [
        TextChunk(chunk_id="c1", manual_name="m", content="Cooling pump maintenance", page_number=1, chunk_index=0, department="d", section_title="s", embedding_model="e"),
        TextChunk(chunk_id="c2", manual_name="m", content="Engine oil replacement", page_number=2, chunk_index=0, department="d", section_title="s", embedding_model="e"),
        TextChunk(chunk_id="c3", manual_name="m", content="Main cooling system failure", page_number=3, chunk_index=0, department="d", section_title="s", embedding_model="e"),
    ]
    
    svc.build_index(chunks)
    assert svc.is_built
    assert svc.corpus_size == 3
    
    # Search for cooling
    results = svc.search("cooling")
    assert len(results) == 2
    # c1 and c3 should match
    matched_ids = [r[0] for r in results]
    assert "c1" in matched_ids
    assert "c3" in matched_ids

def test_bm25_save_load(tmp_path):
    svc = BM25IndexService()
    path = str(tmp_path / "bm25.pkl")
    
    chunks = [
        TextChunk(chunk_id="c1", manual_name="m", content="Cooling pump maintenance", page_number=1, chunk_index=0, department="d", section_title="s", embedding_model="e"),
        TextChunk(chunk_id="c2", manual_name="m", content="Engine oil replacement", page_number=2, chunk_index=0, department="d", section_title="s", embedding_model="e"),
        TextChunk(chunk_id="c3", manual_name="m", content="Main cooling system failure", page_number=3, chunk_index=0, department="d", section_title="s", embedding_model="e"),
    ]
    svc.build_index(chunks)
    svc.save(path)
    
    svc2 = BM25IndexService()
    svc2.load(path)
    assert svc2.is_built
    assert svc2.corpus_size == 3
    
    res = svc2.search("pump")
    assert len(res) == 1
    assert res[0][0] == "c1"
