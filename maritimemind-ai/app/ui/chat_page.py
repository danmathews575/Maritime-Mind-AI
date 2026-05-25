import streamlit as st
from app.ui.api_client import query_agent

def _get_confidence_color(score: float) -> str:
    if score >= 0.8:
        return "green"
    elif score >= 0.5:
        return "orange"
    return "red"

def render_chat_page():
    st.title("💬 Chat")
    
    # Ensure session exists in state
    if "session_id" not in st.session_state:
        from app.ui.api_client import create_session
        new_id = create_session()
        if new_id:
            st.session_state["session_id"] = new_id
        else:
            st.error("Backend unreachable. Could not start session.")
            return

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat messages from history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display inline images if any were returned
            if "images" in msg and msg["images"]:
                for img_data in msg["images"]:
                    url = img_data.get("url")
                    # Streamlit can render from local URLs if we prefix with backend address
                    if url and url.startswith("/"):
                        full_url = f"http://localhost:8000{url}"
                        st.image(full_url, caption=img_data.get("caption", ""))
            
            # Display citations
            if "citations" in msg and msg["citations"]:
                with st.expander("Sources"):
                    for c in msg["citations"]:
                        st.markdown(f"- **{c.get('manual_name')}** (Page {c.get('page_number')})")
                        
            # Display confidence
            if "confidence" in msg and msg["role"] == "assistant":
                conf = msg["confidence"]
                color = _get_confidence_color(conf)
                st.markdown(f"<span style='color:{color}; font-size:0.8em'>Retrieval Confidence: {conf:.2f}</span>", unsafe_allow_html=True)

    # Accept user input
    if prompt := st.chat_input("Ask a technical question..."):
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("*(Thinking...)*")
            
            with st.spinner("Analyzing manuals..."):
                provider = st.session_state.get("llm_provider", "ollama")
                response_data = query_agent(prompt, st.session_state["session_id"], provider)
                
            if "error" in response_data:
                message_placeholder.markdown(f"**Error:** {response_data['error']}")
                st.session_state["messages"].append({"role": "assistant", "content": f"**Error:** {response_data['error']}"})
            else:
                answer = response_data.get("answer", "")
                citations = response_data.get("citations", [])
                images = response_data.get("images", [])
                confidence = response_data.get("confidence", 0.0)
                
                message_placeholder.markdown(answer)
                
                # Render images
                for img_data in images:
                    url = img_data.get("url")
                    if url and url.startswith("/"):
                        full_url = f"http://localhost:8000{url}"
                        st.image(full_url, caption=img_data.get("caption", ""))
                        
                # Render citations
                if citations:
                    with st.expander("Sources"):
                        for c in citations:
                            st.markdown(f"- **{c.get('manual_name')}** (Page {c.get('page_number')})")
                            
                color = _get_confidence_color(confidence)
                st.markdown(f"<span style='color:{color}; font-size:0.8em'>Retrieval Confidence: {confidence:.2f}</span>", unsafe_allow_html=True)
                
                # Save to session state
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": answer,
                    "citations": citations,
                    "images": images,
                    "confidence": confidence
                })
