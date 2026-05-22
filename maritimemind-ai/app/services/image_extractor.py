"""
Image Extractor Service — Final Retrieval-Aware Architecture
Saves high-quality diagrams/schematics to disk, deduplicates, filters noise,
and produces typed ImageMetadata objects.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import io

from PIL import Image, ImageStat
from app.configs.config import settings
from app.models.schemas import BoundingBox, ImageMetadata
from app.services.pdf_parser import ParsedDocument
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.image_extractor")


class ImageExtractorService:
    """
    Saves parsed images to disk and produces ImageMetadata records.
    Filters out decorative, small, or low-variance images.
    """

    def __init__(self) -> None:
        self._base_dir = Path(settings.EXTRACTED_IMAGES_DIR)

    def extract_and_save(
        self,
        parsed_doc: ParsedDocument,
        ocr_service=None
    ) -> List[ImageMetadata]:
        manual_dir = self._base_dir / parsed_doc.manual_name
        manual_dir.mkdir(parents=True, exist_ok=True)

        seen_hashes: Dict[str, str] = {}
        results: List[ImageMetadata] = []

        for page in parsed_doc.pages:
            for raw_img in page.images:
                image_id = self._generate_image_id(raw_img.image_bytes, raw_img.page_number)

                # Deduplicate
                if self._is_duplicate(image_id, seen_hashes):
                    continue

                # Size & Aspect Ratio Filter
                if not self._passes_size_filter(raw_img.width, raw_img.height):
                    continue

                # Variance / Diagram Confidence Check
                try:
                    img = Image.open(io.BytesIO(raw_img.image_bytes)).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to open image {image_id}: {e}")
                    continue

                diagram_confidence = self._compute_diagram_confidence(img)
                if diagram_confidence < 0.2:
                    logger.debug(f"Skipping low-confidence image {image_id}")
                    continue

                # Save to disk
                out_path = manual_dir / f"{image_id}.png"
                if not out_path.exists():
                    img.save(out_path, format="PNG")

                # OCR
                ocr_text = ""
                ocr_quality = 1.0
                if ocr_service:
                    try:
                        ocr_text = ocr_service.extract(str(out_path))
                        if ocr_text:
                            alnum = sum(c.isalnum() for c in ocr_text)
                            ocr_quality = alnum / len(ocr_text) if len(ocr_text) > 0 else 0.0
                    except Exception as e:
                        logger.warning(f"OCR failed for {image_id}: {e}")
                        ocr_quality = 0.0

                x0, y0, x1, y1 = raw_img.bbox
                bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

                metadata = ImageMetadata(
                    image_id=image_id,
                    manual_name=parsed_doc.manual_name,
                    page_number=raw_img.page_number,
                    image_path=str(out_path),
                    caption="",
                    bbox=bbox,
                    linked_chunks=[],
                    ocr_text=ocr_text,
                    ocr_quality=ocr_quality,
                    diagram_confidence=diagram_confidence,
                    embedding_model=settings.CLIP_MODEL_NAME,
                )

                seen_hashes[image_id] = image_id
                results.append(metadata)

        logger.info(f"Extracted {len(results)} high-quality diagrams from '{parsed_doc.manual_name}'")
        return results

    @staticmethod
    def _generate_image_id(image_bytes: bytes, page_number: int) -> str:
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        hasher.update(str(page_number).encode())
        return hasher.hexdigest()[:16]

    @staticmethod
    def _passes_size_filter(width: int, height: int) -> bool:
        """Enforce strict min dimensions (250px) and sane aspect ratios to remove page borders."""
        min_w = max(settings.MIN_IMAGE_WIDTH, 250)
        min_h = max(settings.MIN_IMAGE_HEIGHT, 250)
        
        if width < min_w or height < min_h:
            return False
            
        aspect = width / height if height > 0 else 0
        if aspect > 8.0 or aspect < 0.125:
            return False
            
        return True

    @staticmethod
    def _is_duplicate(image_id: str, seen: Dict[str, str]) -> bool:
        return image_id in seen

    @staticmethod
    def _compute_diagram_confidence(img: Image.Image) -> float:
        """
        Computes a confidence score based on image variance. 
        Filters out blank pages, solid color blocks, and extreme noise.
        """
        stat = ImageStat.Stat(img)
        avg_stddev = sum(stat.stddev) / len(stat.stddev)
        
        # Solid colors / practically blank have very low stddev
        if avg_stddev < 5.0:
            return 0.1
            
        confidence = 0.9
        
        # Slight penalty for smaller images
        area = img.width * img.height
        if area < 100000:
            confidence -= 0.1
            
        return min(max(confidence, 0.0), 1.0)
