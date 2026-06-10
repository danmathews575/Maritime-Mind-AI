import os
from pathlib import Path
from PIL import Image, ImageStat
from app.configs.config import settings
from app.services.vector_store import VectorStoreService
from app.services.image_extractor import ImageExtractorService

def cleanup_images():
    base_dir = Path(settings.EXTRACTED_IMAGES_DIR)
    vs = VectorStoreService()
    
    useless_ids = []
    useless_paths = []
    
    for manual_dir in base_dir.iterdir():
        if not manual_dir.is_dir(): continue
        for img_path in manual_dir.glob("*.png"):
            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                
                # Aspect ratio and Size filter
                if not ImageExtractorService._passes_size_filter(w, h):
                    useless_ids.append(img_path.stem)
                    useless_paths.append(img_path)
                    continue
                
                # Variance / Confidence filter
                conf = ImageExtractorService._compute_diagram_confidence(img)
                if conf < 0.2:
                    useless_ids.append(img_path.stem)
                    useless_paths.append(img_path)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    print(f"Found {len(useless_ids)} images to delete.")
    
    # Delete from Vector Store
    if useless_ids:
        print(f"Deleting {len(useless_ids)} records from Qdrant...")
        try:
            import uuid
            from qdrant_client.http import models
            qdrant_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, i)) for i in useless_ids]
            vs._client.delete(
                collection_name=vs._image_collection_name,
                points_selector=models.PointIdsList(points=qdrant_ids)
            )
            print("Successfully deleted from Qdrant.")
        except Exception as e:
            print(f"Error deleting from Qdrant: {e}")
            
    # Delete from Disk
    for p in useless_paths:
        try:
            p.unlink()
        except Exception as e:
            print(f"Error deleting file {p}: {e}")
            
    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup_images()
