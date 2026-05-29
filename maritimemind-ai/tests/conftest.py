import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from app.models.schemas import TextChunk, ImageMetadata, RetrievalResult, RetrievalScores

# Ensure standard test env vars
os.environ["ENVIRONMENT"] = "test"
os.environ["LLM_PROVIDER"] = "ollama"

# Disable TensorFlow backend in HuggingFace transformers to prevent 
# protobuf incompatibilities and speed up model loading.
# SentenceTransformers relies exclusively on PyTorch.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for test file operations."""
    for subdir in ["raw_pdfs", "extracted_images", "extracted_text", "metadata"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)
    return tmp_path

@pytest.fixture
def sample_text_chunk():
    """A reusable TextChunk for testing."""
    return TextChunk(
        chunk_id="test_chunk_001",
        manual_name="test_manual",
        department="engineering",
        page_number=1,
        section_title="Test Section",
        content="This is a test chunk about cooling pump maintenance.",
        keywords=["cooling", "pump", "maintenance"],
        embedding_model="all-MiniLM-L6-v2",
    )

@pytest.fixture
def sample_image_metadata():
    """A reusable ImageMetadata for testing."""
    return ImageMetadata(
        image_id="test_img_001",
        manual_name="test_manual",
        page_number=5,
        image_path="/tmp/test_img.png",
        caption="Cooling pump diagram",
        embedding_model="ViT-B-32",
    )

@pytest.fixture
def sample_retrieval_result(sample_text_chunk):
    """A reusable RetrievalResult for testing."""
    return RetrievalResult(
        chunk=sample_text_chunk,
        scores=RetrievalScores(bm25_score=5.0, vector_score=0.3, confidence_score=0.8),
    )

@pytest.fixture
def mock_vector_store():
    """A mocked VectorStoreService."""
    return MagicMock()
