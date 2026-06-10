import pytest
from unittest.mock import patch, MagicMock

# Mock the heavy dependencies before importing the graph
with patch("app.orchestration.graph.RetrievalController"), \
     patch("app.orchestration.graph.LLMService"):
    from app.orchestration.graph import create_graph

def test_graph_compiles():
    """Verify that the LangGraph state graph compiles without errors."""
    graph = create_graph()
    assert graph is not None

def test_graph_nodes_exist():
    """Verify that all required nodes are present in the graph."""
    graph = create_graph()
    nodes = graph.nodes
    assert "context_router" in nodes
    assert "text_retrieval" in nodes
    assert "visual_specialist" in nodes
    assert "verification" in nodes
    assert "synthesizer" in nodes
    assert "quality_reviewer" in nodes

