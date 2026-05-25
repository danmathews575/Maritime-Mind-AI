import requests
from typing import Dict, Any, Optional

API_BASE = "http://localhost:8000/api/v1"

def check_health() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_stats() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{API_BASE}/stats", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {}

def create_session() -> Optional[str]:
    try:
        resp = requests.post(f"{API_BASE}/sessions", timeout=5)
        resp.raise_for_status()
        return resp.json().get("session_id")
    except Exception as e:
        print(f"Error creating session: {e}")
        return None

def clear_session(session_id: str) -> bool:
    try:
        resp = requests.delete(f"{API_BASE}/sessions/{session_id}", timeout=5)
        return resp.status_code == 204
    except:
        return False

def get_ingestion_status() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{API_BASE}/ingest/status", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

def trigger_ingestion(pdf_path: str) -> Dict[str, Any]:
    try:
        resp = requests.post(f"{API_BASE}/ingest", json={"pdf_path": pdf_path}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

def query_agent(query: str, session_id: Optional[str] = None, provider: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "query": query,
        "session_id": session_id,
        "top_k": 5,
        "provider": provider
    }
    try:
        resp = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout:
        return {"error": "Query timed out. The model took too long to respond."}
    except Exception as e:
        return {"error": f"API Error: {str(e)}"}
