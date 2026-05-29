import os
import sys

# Ensure the app module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def download_models():
    print("Downloading SentenceTransformer model...")
    from app.services.embedding import TextEmbeddingService
    TextEmbeddingService() # Initializes the model in its constructor

    print("Downloading OpenCLIP model...")
    from app.services.clip_embedding import ImageEmbeddingService
    ImageEmbeddingService() # Initializes the model in its constructor

    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
