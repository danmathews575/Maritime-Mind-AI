import pytest
from unittest.mock import patch, MagicMock

# Mock the heavy dependencies before importing the graph
with patch("app.orchestration.graph.RetrievalController"), \
     patch("app.orchestration.graph.LLMService"):
    from app.orchestration.graph import create_graph, app_graph

def test_graph_compiles():
    """Verify that the LangGraph state graph compiles without errors."""
    assert app_graph is not None

def test_graph_nodes_exist():
    """Verify that all required nodes are present in the graph."""
    nodes = app_graph.nodes
    assert "context_router" in nodes
    assert "text_retrieval" in nodes
    assert "visual_specialist" in nodes
    assert "verification" in nodes
    assert "synthesizer" in nodes
    assert "quality_reviewer" in nodes
