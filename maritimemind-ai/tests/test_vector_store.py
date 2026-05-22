import pytest
import datetime
from unittest.mock import patch
from app.services.vector_store import VectorStoreService
from app.models.schemas import TextChunk, ImageMetadata

@pytest.fixture
def temp_chroma(tmp_path):
    with patch("app.services.vector_store.settings.CHROMADB_PERSIST_DIR", str(tmp_path / "chromadb")):
        # Ensure singleton clears
        import app.services.vector_store as vs
        vs._client = None
        svc = vs.VectorStoreService()
        yield svc
        vs._client = None

def test_add_and_query_text(temp_chroma):
    chunk = TextChunk(
        chunk_id="t1",
        manual_name="m1",
        content="Engine oil replacement.",
        page_number=1,
        chunk_index=0,
        department="engineering",
        section_title="test",
        embedding_model="test-model"
    )
    # fake 384 dim
    emb = [0.1] * 384
    
    upserted = temp_chroma.add_text_chunks([chunk], [emb])
    assert upserted == 1
    
    results = temp_chroma.query_text([0.1] * 384, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "t1"
    assert results[0]["document"] == "Engine oil replacement."
    assert results[0]["metadata"]["manual_name"] == "m1"
    assert results[0]["metadata"]["department"] == "engineering"

def test_add_and_query_images(temp_chroma):
    img = ImageMetadata(
        image_id="i1",
        manual_name="m1",
        page_number=2,
        image_path="/tmp/i1.png",
        caption="Diagram 1",
        ocr_text="",
        embedding_model="test-clip",
        linked_chunks=[]
    )
    # fake 512 dim
    emb = [0.2] * 512
    
    upserted = temp_chroma.add_image_embeddings([img], [emb])
    assert upserted == 1
    
    results = temp_chroma.query_images([0.2] * 512, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "i1"
    assert "Diagram 1" in results[0]["document"]
    assert results[0]["metadata"]["manual_name"] == "m1"

def test_get_all_text_chunks(temp_chroma):
    chunk = TextChunk(
        chunk_id="t1",
        manual_name="m1",
        content="test content",
        page_number=1,
        chunk_index=0,
        department="test",
        section_title="test",
        embedding_model="test",
        created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    )
    emb = [0.1] * 384
    temp_chroma.add_text_chunks([chunk], [emb])
    
    chunks = temp_chroma.get_all_text_chunks()
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "t1"
    assert chunks[0].content == "test content"
    
def test_stats_and_delete(temp_chroma):
    chunk = TextChunk(
        chunk_id="t1",
        manual_name="m1",
        content="test",
        page_number=1,
        chunk_index=0,
        department="test",
        section_title="test",
        embedding_model="t"
    )
    temp_chroma.add_text_chunks([chunk], [[0.1]*384])
    
    stats = temp_chroma.get_collection_stats()
    assert stats["text_collection"]["count"] == 1
    assert stats["image_collection"]["count"] == 0
    
    temp_chroma.reset_all()
    stats = temp_chroma.get_collection_stats()
    assert stats["text_collection"]["count"] == 0
