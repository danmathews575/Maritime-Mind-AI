"""
tests/api/test_sessions.py
===========================
Tests for session management endpoints.

POST   /api/v1/sessions
GET    /api/v1/sessions/{id}/history
DELETE /api/v1/sessions/{id}
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestCreateSession:
    def test_create_session_returns_201(self):
        response = client.post("/api/v1/sessions")
        assert response.status_code == 201

    def test_create_session_returns_session_id(self):
        response = client.post("/api/v1/sessions")
        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0

    def test_each_session_gets_unique_id(self):
        r1 = client.post("/api/v1/sessions").json()["session_id"]
        r2 = client.post("/api/v1/sessions").json()["session_id"]
        assert r1 != r2


class TestSessionHistory:
    def test_history_empty_on_new_session(self):
        session_id = client.post("/api/v1/sessions").json()["session_id"]
        response = client.get(f"/api/v1/sessions/{session_id}/history")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["messages"] == []

    def test_history_404_for_unknown_session(self):
        response = client.get("/api/v1/sessions/nonexistent-id/history")
        assert response.status_code == 404


class TestClearSession:
    def test_clear_session_returns_204(self):
        session_id = client.post("/api/v1/sessions").json()["session_id"]
        response = client.delete(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 204

    def test_clear_session_404_for_unknown(self):
        response = client.delete("/api/v1/sessions/no-such-session")
        assert response.status_code == 404
