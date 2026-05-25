import streamlit as st
import os
from app.ui.api_client import get_ingestion_status, trigger_ingestion

def render_ingestion_page():
    st.title("📚 Knowledge Base Management")
    st.markdown("Manage the technical manuals indexed by MaritimeMind AI.")
    
    status_data = get_ingestion_status()
    
    st.subheader("Currently Indexed Manuals")
    if status_data.get("status") == "error":
        st.error(f"Failed to fetch ingestion status: {status_data.get('error')}")
    else:
        manifest = status_data.get("manifest", {})
        if not manifest:
            st.info("No manuals are currently indexed.")
        else:
            for manual, meta in manifest.items():
                with st.expander(f"📖 {manual}"):
                    st.write(f"**Indexed on:** {meta.get('ingested_at')}")
                    st.write(f"**Chunks:** {meta.get('chunk_count')}")
                    st.write(f"**Images:** {meta.get('image_count')}")
                    
    st.divider()
    
    st.subheader("Ingest New Manual")
    st.markdown("Enter the absolute path to a PDF file on the server to ingest it.")
    
    pdf_path = st.text_input("PDF File Path", placeholder="/path/to/manual.pdf")
    
    if st.button("Start Ingestion", type="primary"):
        if not pdf_path:
            st.warning("Please enter a file path.")
        elif not os.path.exists(pdf_path):
            st.error(f"File not found on the server: {pdf_path}")
        elif not pdf_path.lower().endswith(".pdf"):
            st.error("Only PDF files are supported.")
        else:
            with st.spinner("Ingesting manual... This may take several minutes."):
                res = trigger_ingestion(pdf_path)
                if res.get("status") == "success":
                    st.success(f"Successfully ingested {res.get('manual_name')}")
                    st.write(f"Created {res.get('chunk_count')} chunks and extracted {res.get('image_count')} images.")
                    st.rerun() # Refresh the manifest list
                else:
                    st.error(f"Ingestion failed: {res.get('error')}")
