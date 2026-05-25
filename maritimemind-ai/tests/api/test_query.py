"""
tests/api/test_query.py
========================
Tests for POST /api/v1/query.

The agent workflow and all retrieval services are mocked so tests
run offline without Ollama, ChromaDB, or any ML models.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.api.main import app
from app.models.schemas import QueryIntent

client = TestClient(app, raise_server_exceptions=False)

# ─── helpers ─────────────────────────────────────────────────────────────────

def _mock_agent_state(
    answer="Cooling water is used to regulate engine temperature.",
    confidence=0.85,
    intent=QueryIntent.EXPLANATION,
    citations=None,
    images=None,
    quality_passed=True,
):
    """Build a minimal AgentState dict that query.py can consume."""
    if citations is None:
        citations = [{"manual_name": "Engine Manual", "page_number": 12,
                      "section_title": "Cooling", "chunk_id": "abc123"}]
    return {
        "response_text": answer,
        "retrieval_confidence": confidence,
        "intent": intent,
        "citations": citations,
        "image_results": images or [],
        "attached_images": [],
        "quality_passed": quality_passed,
        "quality_notes": "",
    }


# ─── tests ───────────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_query_returns_200(self):
        with patch("app.api.routes.query.run_agent_workflow",
                   return_value=_mock_agent_state()):
            response = client.post("/api/v1/query", json={"query": "What is cooling water?"})
        assert response.status_code == 200

    def test_query_response_has_expected_fields(self):
        with patch("app.api.routes.query.run_agent_workflow",
                   return_value=_mock_agent_state()):
            response = client.post("/api/v1/query", json={"query": "What is cooling water?"})
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "images" in data
        assert "confidence" in data
        assert "intent" in data
        assert "quality_passed" in data

    def test_query_returns_answer_text(self):
        expected = "The ballast system maintains vessel stability."
        with patch("app.api.routes.query.run_agent_workflow",
                   return_value=_mock_agent_state(answer=expected)):
            response = client.post("/api/v1/query", json={"query": "What is ballast?"})
        assert response.json()["answer"] == expected

    def test_query_confidence_in_range(self):
        with patch("app.api.routes.query.run_agent_workflow",
                   return_value=_mock_agent_state(confidence=0.72)):
            response = client.post("/api/v1/query", json={"query": "Explain ballast system"})
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_query_with_session_id(self):
        """Query with a valid session_id should store the exchange in memory."""
        session_id = client.post("/api/v1/sessions").json()["session_id"]
        with patch("app.api.routes.query.run_agent_workflow",
                   return_value=_mock_agent_state()):
            response = client.post("/api/v1/query",
                                   json={"query": "Pump pressure?", "session_id": session_id})
        assert response.status_code == 200
        assert response.json()["session_id"] == session_id

        # History should now have 2 messages (user + assistant)
        history = client.get(f"/api/v1/sessions/{session_id}/history").json()["messages"]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_query_invalid_session_returns_404(self):
        with patch("app.api.routes.query.run_agent_workflow",
                   return_value=_mock_agent_state()):
            response = client.post("/api/v1/query",
                                   json={"query": "Test?", "session_id": "bad-id"})
        assert response.status_code == 404

    def test_query_missing_query_returns_422(self):
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_query_agent_error_returns_500(self):
        with patch("app.api.routes.query.run_agent_workflow",
                   side_effect=RuntimeError("LLM unavailable")):
            response = client.post("/api/v1/query",
                                   json={"query": "What is the pump?"})
        assert response.status_code == 500

    def test_query_fallback_answer_when_empty_response(self):
        """If agent returns empty response_text we should still return a message."""
        with patch("app.api.routes.query.run_agent_workflow",
                   return_value=_mock_agent_state(answer="")):
            response = client.post("/api/v1/query", json={"query": "Empty test"})
        assert response.status_code == 200
        assert len(response.json()["answer"]) > 0
