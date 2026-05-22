"""
OCR Service — Phase 2.6 (Stub)
Interface stub for optical character recognition on extracted images.
Currently returns empty string. Future integration: Tesseract or LLaVA vision model.
"""
from __future__ import annotations

from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.ocr")


class OcrService:
    """
    OCR service interface stub.

    Current behavior: returns empty string (no-op).

    Future integration options (Phase 15):
    - Tesseract (offline, lightweight): `pytesseract.image_to_string(Image.open(path))`
    - LLaVA / MiniCPM-V (local vision LLM via Ollama): structured caption + text extraction
    - EasyOCR (GPU-accelerated, multi-language support)

    This stub ensures that all downstream components (image extractor, pipeline)
    are already wired for OCR output without requiring OCR to be active.
    """

    def extract(self, image_path: str) -> str:
        """
        Extracts text from an image file.

        Args:
            image_path: Absolute path to a PNG/JPEG image file.

        Returns:
            Extracted text string, or empty string if OCR is not active.
        """
        logger.debug(f"OCR stub called for: {image_path} (returning empty)")
        return ""

    def is_available(self) -> bool:
        """Returns False — OCR engine not yet installed."""
        return False
