import pytest
from unittest.mock import Mock
from app.retrieval.image_retrieval import ImageRetrievalService
from app.models.schemas import RetrievalResult, RetrievalScores, TextChunk, ImageMetadata

def test_image_retrieval_fusion():
    vs_mock = Mock()
    clip_mock = Mock()
    
    # Mock clip semantic search
    clip_mock.embed_text_for_image_search.return_value = [0.1] * 512
    # chroma returns {"id": "img1", "distance": 0.2} meaning similarity = 0.8
    vs_mock.query_images.return_value = [{"id": "img1", "distance": 0.2}]
    
    # Mock images returned by IDs
    img1 = ImageMetadata(image_id="img1", manual_name="test.pdf", page_number=1, image_path="path", section_title="", caption="cooling pump", embedding_model="clip")
    img2 = ImageMetadata(image_id="img2", manual_name="test.pdf", page_number=1, image_path="path", section_title="", caption="wiring diagram", embedding_model="clip")
    
    # get_images_by_ids will be called for path 1 (img1) and path 2 (img2)
    def mock_get_images_by_ids(ids):
        res = []
        for i in ids:
            if i == "img1": res.append(img1)
            elif i == "img2": res.append(img2)
        return res
    vs_mock.get_images_by_ids.side_effect = mock_get_images_by_ids
    
    # Mock text results (Path 2 triggers off related_image_ids)
    chunk = TextChunk(
        chunk_id="chunk1",
        manual_name="test.pdf",
        department="deck",
        page_number=1,
        section_title="test",
        content="test content",
        related_image_ids=["img2"],
        embedding_model="test"
    )
    text_results = [
        RetrievalResult(chunk=chunk, scores=RetrievalScores(confidence_score=0.9))
    ]
    
    service = ImageRetrievalService(vs_mock, clip_mock)
    retrieved = service.search("cooling pump diagram", text_results)
    
    assert len(retrieved) == 2
    
    # Check img1 (from Path 1)
    # clip_score = 0.8, association_score = 0
    # subsystem_boost = 1.0 ("cooling" in query and caption)
    # proximity_boost = 0.25 (same manual, same page 1)
    # raw_score = (0.45 * 0.8) + (0.30 * 0) + 0.25 + (0.05 * 1.0) + 0.10 (subsystem boost match) = 0.36 + 0.25 + 0.05 + 0.10 = 0.76
    # weight for "cooling pump" -> generic -> 0.4
    # final = 0.76 * 0.4 = 0.304
    
    # Check img2 (from Path 2)
    # clip_score = 0, association_score = 0.9
    # subsystem_boost = 0
    # proximity_boost = 0.25
    # raw_score = (0.45 * 0) + (0.30 * 0.9) + 0.25 + (0.05 * 1.0) = 0.27 + 0.25 + 0.05 = 0.57
    # weight for "wiring diagram" -> 1.0
    # final = 0.57 * 1.0 = 0.57
    
    # So img2 should be ranked higher than img1
    assert retrieved[0].metadata.image_id == "img2"
    assert retrieved[1].metadata.image_id == "img1"
    
    assert "Path 2: Associated with retrieved text" in retrieved[0].explainability.retrieval_reason
    assert "Path 1: CLIP Semantic Match" in retrieved[1].explainability.retrieval_reason
