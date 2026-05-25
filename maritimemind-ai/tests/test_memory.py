import pytest
from app.memory.conversation_memory import ConversationMemoryService
from app.memory.query_expander import QueryExpander
from app.models.schemas import ChatMessage

def test_session_creation():
    memory = ConversationMemoryService()
    session_id = memory.create_session()
    assert session_id is not None
    assert memory.get_session(session_id) is not None

def test_add_and_get_message():
    memory = ConversationMemoryService()
    session_id = memory.create_session()
    memory.add_message(session_id, "user", "What is the cooling pump?")
    
    history = memory.get_history(session_id)
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "What is the cooling pump?"

def test_history_limit():
    memory = ConversationMemoryService()
    session_id = memory.create_session()
    
    for i in range(15):
        memory.add_message(session_id, "user", f"Message {i}")
        
    history = memory.get_history(session_id, max_messages=10)
    assert len(history) == 10
    assert history[-1].content == "Message 14"
    assert history[0].content == "Message 5"

def test_query_expansion():
    expander = QueryExpander()
    history = [
        ChatMessage(role="user", content="What is the main engine cooling pump?"),
        ChatMessage(role="assistant", content="The main engine cooling pump circulates water...")
    ]
    
    # Should expand because of 'it'
    expanded = expander.expand("Show me a diagram of it", history)
    assert expanded != "Show me a diagram of it"
    assert "main engine cooling pump" in expanded
    
    # Should not expand
    not_expanded = expander.expand("What is the ballast system?", history)
    assert not_expanded == "What is the ballast system?"
