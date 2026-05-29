import os

def setup():
    dirs = [
        "data/raw_pdfs",
        "data/extracted_text",
        "data/extracted_images",
        "data/processed_chunks",
        "data/metadata",
        "vector_store/chromadb",
        "logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Ensured directory exists: {d}")

    print("Setup complete. Directories are ready.")

if __name__ == "__main__":
    setup()
