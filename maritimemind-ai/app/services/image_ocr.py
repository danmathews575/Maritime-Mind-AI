"""
app/services/image_ocr.py
==========================
EasyOCR-based text extraction from engineering diagrams and technical images.

Purpose: Make diagram labels, callout text, figure titles, and part numbers
searchable so that keyword queries can find relevant diagrams even when
CLIP (which is trained on natural images) fails on engineering schematics.

Why EasyOCR over Tesseract:
- No system-level install required (pure pip)
- GPU-optional (works on CPU for demo use)
- Strong performance on printed technical text
- Handles rotated/skewed text better

Diagram Type Classification:
A lightweight keyword-based classifier assigns a diagram_type label to each
image, which is displayed as a badge in the frontend UI — making the routing
system visually impressive.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.configs.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.image_ocr")

# Diagram type keywords for classification
_DIAGRAM_TYPES = {
    "SCHEMATIC": [
        "schematic", "p&id", "pid", "piping", "instrumentation",
        "flow diagram", "process flow", "hydraulic", "pneumatic",
    ],
    "WIRING": [
        "wiring", "electrical", "circuit", "single line", "switchboard",
        "relay", "terminal", "cable", "distribution", "power diagram",
    ],
    "FLOWCHART": [
        "flowchart", "flow chart", "procedure", "decision", "start", "end",
        "yes", "no", "process", "step", "sequence",
    ],
    "CROSS-SECTION": [
        "cross section", "cross-section", "section view", "cutaway",
        "longitudinal", "transverse", "exploded view",
    ],
    "TABLE": [
        "table", "specification", "data sheet", "parameters", "values",
        "limits", "tolerances",
    ],
    "PHOTO": [],  # Fallback: high resolution, complex image
}


class ImageOCRService:
    """
    Extracts text from technical images using EasyOCR.

    Usage:
        ocr = ImageOCRService()
        result = ocr.extract_text("/path/to/diagram.png")
        print(result.text)       # OCR text
        print(result.diagram_type)  # "SCHEMATIC", "WIRING", etc.
    """

    def __init__(self):
        self._reader = None  # Lazy-load to avoid slow startup

    def _get_reader(self):
        """Lazy-init EasyOCR reader (downloads model on first use)."""
        if self._reader is None:
            try:
                import easyocr
                langs = getattr(settings, "EASYOCR_LANGUAGES", ["en"])
                gpu = settings.DEVICE == "cuda"
                self._reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
                logger.info(f"EasyOCR initialized (GPU={gpu}, langs={langs})")
            except ImportError:
                logger.warning("EasyOCR not installed. Run: pip install easyocr")
                return None
        return self._reader

    def extract_text(self, image_path: str, min_confidence: float = 0.4) -> "OCRResult":
        """
        Extract text from an image file.

        Args:
            image_path: Absolute path to the image file.
            min_confidence: Minimum EasyOCR confidence to include a text block.

        Returns:
            OCRResult with .text and .diagram_type fields.
        """
        reader = self._get_reader()
        if reader is None:
            return OCRResult(text="", diagram_type="UNKNOWN")

        try:
            path = Path(image_path)
            if not path.exists():
                logger.debug(f"Image not found: {image_path}")
                return OCRResult(text="", diagram_type="UNKNOWN")

            results = reader.readtext(str(path), detail=1, paragraph=False)
            # Filter by confidence and join text blocks
            texts = [
                text for (_, text, conf) in results
                if conf >= min_confidence and text.strip()
            ]
            ocr_text = " ".join(texts)
            diagram_type = _classify_diagram(ocr_text)

            if ocr_text:
                logger.debug(
                    f"OCR: '{path.name}' → {len(texts)} text blocks, "
                    f"type={diagram_type}"
                )
            return OCRResult(text=ocr_text, diagram_type=diagram_type)

        except Exception as e:
            logger.warning(f"OCR failed for {image_path}: {e}")
            return OCRResult(text="", diagram_type="UNKNOWN")

    def is_available(self) -> bool:
        """Check if EasyOCR is installed and loadable."""
        return self._get_reader() is not None


class OCRResult:
    """Container for OCR extraction results."""

    def __init__(self, text: str, diagram_type: str):
        self.text = text.strip()
        self.diagram_type = diagram_type

    def __repr__(self) -> str:
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"OCRResult(type={self.diagram_type!r}, text={preview!r})"


def _classify_diagram(ocr_text: str) -> str:
    """
    Classify a diagram type from its OCR-extracted text.

    Strategy: keyword matching against known maritime engineering diagram types.
    Falls back to 'DIAGRAM' if no strong signal found.
    """
    text_lower = ocr_text.lower()
    scores: dict[str, int] = {}

    for dtype, keywords in _DIAGRAM_TYPES.items():
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[dtype] = score

    if not scores:
        # Check for typical figure/diagram indicators in the text
        if any(kw in text_lower for kw in ["fig.", "figure", "drawing", "dwg", "dia."]):
            return "DIAGRAM"
        return "PHOTO"

    return max(scores, key=scores.get)


# Module-level singleton
_ocr_service: Optional[ImageOCRService] = None


def get_ocr_service() -> ImageOCRService:
    """Get the module-level EasyOCR service singleton."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = ImageOCRService()
    return _ocr_service
