import os
from pathlib import Path
from PIL import Image, ImageStat
import chromadb
from app.configs.config import settings

def find_useless_images():
    base_dir = Path(settings.EXTRACTED_IMAGES_DIR)
    useless = []
    
    for manual_dir in base_dir.iterdir():
        if not manual_dir.is_dir(): continue
        for img_path in manual_dir.glob("*.png"):
            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                
                # Aspect ratio
                aspect = w / h if h > 0 else 0
                
                # Variance
                stat = ImageStat.Stat(img)
                avg_stddev = sum(stat.stddev) / len(stat.stddev)
                
                # Check for useless heuristics
                # 1. Very small images
                if w * h < 80000:
                    useless.append((img_path, "too small"))
                # 2. Extreme aspect ratios (e.g. headers/footers)
                elif aspect > 5.0 or aspect < 0.2:
                    useless.append((img_path, "extreme aspect ratio"))
                # 3. Very low variance (almost solid colors)
                elif avg_stddev < 15.0:
                    useless.append((img_path, "low variance"))
            except Exception as e:
                useless.append((img_path, f"error: {e}"))
                
    print(f"Found {len(useless)} potentially useless images out of {sum([len(list(d.glob('*.png'))) for d in base_dir.iterdir() if d.is_dir()])}.")
    # Print a sample
    for path, reason in useless[:20]:
        print(f"{path.name} - {reason}")

if __name__ == "__main__":
    find_useless_images()
