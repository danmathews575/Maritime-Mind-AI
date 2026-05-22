import pytest
from app.services.embedding import TextEmbeddingService
from app.services.clip_embedding import ImageEmbeddingService

# Note: These tests will download the models on first run if not cached,
# which may take some time.

@pytest.fixture(scope="module")
def text_service():
    return TextEmbeddingService()

@pytest.fixture(scope="module")
def clip_service():
    return ImageEmbeddingService()

class TestTextEmbedding:
    def test_dimension(self, text_service):
        assert text_service.dimension == 384

    def test_embed_single_text(self, text_service):
        text = "This is a test of the text embedding service."
        vec = text_service.embed_text(text)
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch(self, text_service):
        texts = ["First test", "Second test", "Third test"]
        vecs = text_service.embed_batch(texts, show_progress=False)
        assert len(vecs) == 3
        for vec in vecs:
            assert len(vec) == 384

    def test_embed_query(self, text_service):
        query = "cooling pump"
        vec = text_service.embed_query(query)
        assert len(vec) == 384


class TestClipEmbedding:
    def test_dimension(self, clip_service):
        assert clip_service.dimension == 512

    def test_embed_text_for_image_search(self, clip_service):
        query = "A picture of a dog"
        vec = clip_service.embed_text_for_image_search(query)
        assert isinstance(vec, list)
        assert len(vec) == 512
        assert all(isinstance(v, float) for v in vec)

    def test_embed_missing_image_returns_zeros(self, clip_service):
        vec = clip_service.embed_image("/non/existent/path.png")
        assert len(vec) == 512
        assert all(v == 0.0 for v in vec)

    def test_embed_image(self, clip_service, tmp_path):
        from PIL import Image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        vec = clip_service.embed_image(str(img_path))
        assert len(vec) == 512
        # Normalised so some values must be non-zero
        assert any(v != 0.0 for v in vec)
