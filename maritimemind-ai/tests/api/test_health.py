"""
tests/api/test_health.py
=========================
Tests for GET /api/v1/health and GET /api/v1/stats.

Uses FastAPI's TestClient — no actual server required.
All heavy services (Ollama, ChromaDB, embedding model) are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestRootEndpoint:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "MaritimeMind AI"
        assert "docs" in data
        assert "health" in data


class TestHealthEndpoint:
    def test_health_returns_valid_structure(self):
        """Health endpoint must return expected fields regardless of backend status."""
        with (
            patch("app.api.routes.health.LLMService") as mock_llm_cls,
            patch("app.api.routes.health.VectorStoreService") as mock_vs_cls,
            patch("app.api.routes.health.TextEmbeddingService") as mock_emb_cls,
            patch("os.path.exists", return_value=True),
        ):
            # Mock healthy services
            mock_llm_cls.return_value.health_check.return_value = True
            mock_vs_cls.return_value.get_collection_stats.return_value = {"text_count": 100}
            mock_emb_cls.return_value.embed_query.return_value = [0.1] * 384

            response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ollama_available" in data
        assert "vector_store_ready" in data
        assert "bm25_index_ready" in data
        assert "embedding_model_ready" in data

    def test_health_status_healthy_when_all_pass(self):
        with (
            patch("app.api.routes.health.LLMService") as mock_llm_cls,
            patch("app.api.routes.health.VectorStoreService") as mock_vs_cls,
            patch("app.api.routes.health.TextEmbeddingService") as mock_emb_cls,
            patch("os.path.exists", return_value=True),
        ):
            mock_llm_cls.return_value.health_check.return_value = True
            mock_vs_cls.return_value.get_collection_stats.return_value = {"text_count": 50}
            mock_emb_cls.return_value.embed_query.return_value = [0.1] * 384

            response = client.get("/api/v1/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_degraded_when_ollama_down(self):
        """System should still be available (degraded) when Ollama is unreachable."""
        with (
            patch("app.api.routes.health.LLMService", side_effect=Exception("Connection refused")),
            patch("app.api.routes.health.VectorStoreService") as mock_vs_cls,
            patch("app.api.routes.health.TextEmbeddingService") as mock_emb_cls,
            patch("os.path.exists", return_value=True),
        ):
            mock_vs_cls.return_value.get_collection_stats.return_value = {"text_count": 50}
            mock_emb_cls.return_value.embed_query.return_value = [0.1] * 384

            response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["ollama_available"] is False


class TestStatsEndpoint:
    def test_stats_returns_valid_structure(self):
        with patch("app.api.routes.health.VectorStoreService") as mock_vs_cls:
            mock_vs_cls.return_value.get_collection_stats.return_value = {
                "text_count": 42,
                "image_count": 7,
            }
            response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert "text_chunk_count" in data
        assert "image_count" in data
        assert "ollama_model" in data
        assert "embedding_model" in data

    def test_stats_returns_zero_counts_on_vs_failure(self):
        with patch("app.api.routes.health.VectorStoreService",
                   side_effect=Exception("DB not found")):
            response = client.get("/api/v1/stats")
        assert response.status_code == 200
        assert response.json()["text_chunk_count"] == 0
