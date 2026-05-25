import streamlit as st
from app.ui.api_client import check_health, get_stats, create_session, clear_session

def render_sidebar():
    with st.sidebar:
        st.title("⚓ MaritimeMind AI")
        st.markdown("Your intelligent maritime technical assistant.")
        
        st.divider()
        
        # Session Management
        st.subheader("Conversation")
        if st.button("Start New Session", use_container_width=True):
            new_id = create_session()
            if new_id:
                st.session_state["session_id"] = new_id
                st.session_state["messages"] = []
                st.rerun()
            else:
                st.error("Failed to start new session.")
                
        if st.button("Clear History", use_container_width=True):
            if "session_id" in st.session_state:
                clear_session(st.session_state["session_id"])
                st.session_state["messages"] = []
                st.rerun()
                
        st.divider()
        
        # Model Selection
        st.subheader("Model Selection")
        providers = ["ollama", "gemini", "openai"]
        
        # Initialize default in session state if not present
        if "llm_provider" not in st.session_state:
            st.session_state["llm_provider"] = "ollama"
            
        current_idx = providers.index(st.session_state["llm_provider"]) if st.session_state["llm_provider"] in providers else 0
        
        selected_provider = st.selectbox(
            "Language Model",
            options=providers,
            index=current_idx,
            format_func=lambda x: x.capitalize()
        )
        st.session_state["llm_provider"] = selected_provider

        st.divider()
        
        # System Status
        st.subheader("System Status")
        health = check_health()
        
        if health.get("status") == "healthy":
            st.success("🟢 System Online")
        elif health.get("status") == "degraded":
            st.warning("🟡 System Degraded")
        else:
            st.error("🔴 System Offline")
            
        if health.get("status") != "error":
            checks = health.get("checks", {})
            ollama = checks.get("ollama", False)
            vs = checks.get("vector_store", False)
            bm25 = checks.get("bm25", False)
            
            st.caption(f"LLM (Ollama): {'🟢' if ollama else '🔴'}")
            st.caption(f"Vector Store: {'🟢' if vs else '🔴'}")
            st.caption(f"BM25 Index: {'🟢' if bm25 else '🔴'}")
            
        st.divider()
        
        # Knowledge Base Stats
        st.subheader("Knowledge Base")
        stats = get_stats()
        if stats:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            st.metric("Total Images", stats.get("total_images", 0))
        else:
            st.caption("Stats unavailable")
            
        st.divider()
        st.caption("v1.0.0 | Phase 10")
