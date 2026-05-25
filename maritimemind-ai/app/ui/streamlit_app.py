import streamlit as st
import os
import sys

# Ensure the root directory is in the path so imports work correctly
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.ui.sidebar import render_sidebar
from app.ui.chat_page import render_chat_page
from app.ui.ingestion_page import render_ingestion_page

# Page Configuration
st.set_page_config(
    page_title="MaritimeMind AI",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Let Streamlit's config.toml handle the background colors for Dark/Light mode natively */
    
    /* Enhanced Chat Message Styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Differentiation for Assistant vs User */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
    }
    
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: transparent;
        border-left: 4px solid #64748b;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize basic state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Chat"
        
    render_sidebar()
    
    # Simple navigation via radio buttons in the sidebar
    with st.sidebar:
        st.divider()
        st.session_state["current_page"] = st.radio(
            "Navigation", 
            ["Chat", "Knowledge Base"],
            index=0 if st.session_state["current_page"] == "Chat" else 1
        )
        
    # Route to the selected page
    if st.session_state["current_page"] == "Chat":
        render_chat_page()
    else:
        render_ingestion_page()

if __name__ == "__main__":
    main()
