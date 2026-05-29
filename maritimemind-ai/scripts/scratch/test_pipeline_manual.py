import os
import sys

# Ensure Python path is set
sys.path.insert(0, os.path.abspath("."))

from app.ingestion.pipeline import IngestionPipeline
from app.services.vector_store import VectorStoreService

def main():
    print("Resetting Vector Store to wipe old schemas...")
    vs = VectorStoreService()
    vs.reset_all()

    pipeline = IngestionPipeline()
    test_pdf = "data/raw_pdfs/engineering/ship cooling system.pdf"
    
    print(f"\nRunning final retrieval-aware ingestion on {test_pdf}...")
    result = pipeline.run(test_pdf, force=True)
    
    print("\n--- INGESTION RESULT ---")
    print(result)
        
    print("\n--- VECTOR STORE STATS ---")
    print(vs.get_collection_stats())

if __name__ == "__main__":
    main()
