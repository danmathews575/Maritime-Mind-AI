from unittest.mock import Mock
from app.retrieval.image_retrieval import ImageRetrievalService
from app.models.schemas import RetrievalResult, RetrievalScores, TextChunk, ImageMetadata

vs_mock = Mock()
clip_mock = Mock()

clip_mock.embed_text_for_image_search.return_value = [0.1] * 512
vs_mock.query_images.return_value = [{"id": "img1", "distance": 0.2}]

img1 = ImageMetadata(image_id="img1", manual_name="test.pdf", page_number=1, image_path="path", section_title="", caption="cooling pump", embedding_model="clip")
img2 = ImageMetadata(image_id="img2", manual_name="test.pdf", page_number=1, image_path="path", section_title="", caption="wiring diagram", embedding_model="clip")

def mock_get_images_by_ids(ids):
    res = []
    for i in ids:
        if i == "img1": res.append(img1)
        elif i == "img2": res.append(img2)
    return res

vs_mock.get_images_by_ids.side_effect = mock_get_images_by_ids

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

for r in retrieved:
    print(r.metadata.image_id, r.explainability.model_dump())
