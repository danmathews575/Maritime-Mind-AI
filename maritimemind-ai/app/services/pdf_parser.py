"""
PDF Parser Service — Phase 2.1
Extracts text blocks (with font metadata), tables (via pdfplumber → Markdown),
and raw images from maritime PDF manuals using PyMuPDF as the primary engine.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import base64

import fitz  # PyMuPDF
import pdfplumber

from app.configs.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.pdf_parser")


# ──────────────────────────────────────────────
#  Internal data models (parser-private)
# ──────────────────────────────────────────────

@dataclass
class TextBlock:
    text: str
    font_size: float
    font_flags: int   # PyMuPDF flags: bold=0x10, italic=0x2
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_number: int


@dataclass
class RawImage:
    image_bytes: bytes
    bbox: Tuple[float, float, float, float]
    page_number: int
    xref: int           # PyMuPDF internal reference
    width: int
    height: int


@dataclass
class ParsedPage:
    page_number: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)   # Each entry is Markdown
    images: List[RawImage] = field(default_factory=list)


@dataclass
class ParsedDocument:
    manual_name: str
    pdf_path: str
    pages: List[ParsedPage] = field(default_factory=list)
    total_images_extracted: int = 0
    total_tables_extracted: int = 0


# ──────────────────────────────────────────────
#  PDF Parser Service
# ──────────────────────────────────────────────

class PdfParserService:
    """
    Parses maritime PDFs into structured ParsedDocument objects.

    Uses PyMuPDF as primary engine for text and image extraction.
    Uses pdfplumber as secondary engine for complex table detection.
    """

    # Font size thresholds for heading classification.
    # These are tunable defaults; adjust per-manual if fonts differ.
    HEADING_FONT_SIZE_THRESHOLD = 11.0   # >= this is likely a heading
    BOLD_FLAG = 16                        # PyMuPDF font flag bit for bold (0x10)
    ITALIC_FLAG = 2                       # PyMuPDF font flag bit for italic (0x02)

    def parse_pdf(self, pdf_path: str) -> ParsedDocument:
        """
        Main entry point. Parses a single PDF file into a ParsedDocument.
        Processes page-by-page to avoid loading entire document into memory.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        manual_name = path.stem
        logger.info(f"Parsing PDF: {path.name}")

        doc = ParsedDocument(manual_name=manual_name, pdf_path=str(path))

        try:
            fitz_doc = fitz.open(pdf_path)
            with pdfplumber.open(pdf_path) as plumber_doc:
                for page_num in range(len(fitz_doc)):
                    fitz_page = fitz_doc[page_num]
                    plumber_page = plumber_doc.pages[page_num]
                    parsed_page = self._parse_page(fitz_page, plumber_page, page_num + 1)
                    doc.pages.append(parsed_page)
                    doc.total_images_extracted += len(parsed_page.images)
                    doc.total_tables_extracted += len(parsed_page.tables)

            fitz_doc.close()

        except Exception as e:
            logger.error(f"Failed to parse {path.name}: {e}", exc_info=True)
            raise

        logger.info(
            f"Parsed {path.name}: {len(doc.pages)} pages, "
            f"{doc.total_images_extracted} images, "
            f"{doc.total_tables_extracted} tables"
        )
        return doc

    # ──────────────────────────────────────────
    #  Per-page parsing
    # ──────────────────────────────────────────

    def _parse_page(
        self,
        fitz_page: fitz.Page,
        plumber_page,
        page_number: int
    ) -> ParsedPage:
        """Orchestrates extraction of text blocks, tables, and images for one page."""
        parsed = ParsedPage(page_number=page_number)

        # Step 1: Extract tables first (pdfplumber), collect their bboxes
        table_bboxes, tables_md = self._extract_tables(plumber_page, page_number)
        parsed.tables = tables_md

        # Step 2: Extract text blocks (PyMuPDF), skip regions covered by tables
        parsed.text_blocks = self._extract_text_blocks(fitz_page, page_number, table_bboxes)

        # Fallback Vision OCR for scanned pages (Phase 12)
        if not parsed.text_blocks and settings.OCR_ENABLED:
            logger.info(f"Page {page_number} has no text blocks. Checking for blank page before Vision OCR...")
            try:
                from app.services.vision_ocr import NvidiaVisionService
                if not hasattr(self, '_vision_service'):
                    self._vision_service = NvidiaVisionService()
                
                # Render page to image (150 DPI is usually sufficient for OCR)
                pix = fitz_page.get_pixmap(dpi=150)
                img_data = pix.tobytes("jpeg")
                
                # Fast Blank Page Detection (StdDev check)
                import io
                from PIL import Image, ImageStat
                pil_img = Image.open(io.BytesIO(img_data)).convert('L')
                stat = ImageStat.Stat(pil_img)
                if stat.stddev[0] < 15.0:
                    logger.info(f"Page {page_number} appears blank (stddev={stat.stddev[0]:.2f}). Skipping OCR.")
                else:
                    b64_img = base64.b64encode(img_data).decode("utf-8")
                    
                    ocr_text = self._vision_service.extract_text_from_image(b64_img)
                    if ocr_text:
                        rect = fitz_page.rect
                        tb = TextBlock(
                            text=ocr_text,
                            font_size=12.0,
                            font_flags=0,
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            page_number=page_number
                        )
                        parsed.text_blocks.append(tb)
            except Exception as e:
                logger.warning(f"Vision OCR fallback failed for page {page_number}: {e}")

        # Step 3: Extract images (PyMuPDF)
        parsed.images = self._extract_images(fitz_page, page_number)

        return parsed

    def _extract_text_blocks(
        self,
        page: fitz.Page,
        page_number: int,
        table_bboxes: List[Tuple]
    ) -> List[TextBlock]:
        """
        Extracts text blocks from a page with font size and flag metadata.
        Skips regions already covered by detected tables.
        """
        blocks = []
        raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in raw_blocks:
            if block.get("type") != 0:  # 0 = text block, 1 = image block
                continue

            block_bbox = tuple(block["bbox"])

            # Skip this block if it lies within a known table region
            if self._overlaps_table(block_bbox, table_bboxes):
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    tb = TextBlock(
                        text=text,
                        font_size=round(span.get("size", 0.0), 2),
                        font_flags=span.get("flags", 0),
                        bbox=tuple(span["bbox"]),
                        page_number=page_number
                    )
                    blocks.append(tb)

        return blocks

    def _extract_tables(
        self,
        plumber_page,
        page_number: int
    ) -> Tuple[List[Tuple], List[str]]:
        """
        Detects and extracts tables using pdfplumber.
        Returns (table_bboxes, [markdown_strings]).
        """
        table_bboxes: List[Tuple] = []
        tables_md: List[str] = []

        try:
            tables = plumber_page.extract_tables()
            plumber_tables_with_bbox = plumber_page.find_tables()

            for i, table in enumerate(tables):
                if not table:
                    continue

                md = self._table_to_markdown(table)
                if md:
                    tables_md.append(md)

                # Collect bounding box of this table region
                if i < len(plumber_tables_with_bbox):
                    bbox = plumber_tables_with_bbox[i].bbox
                    table_bboxes.append(bbox)

        except Exception as e:
            logger.warning(f"Table extraction failed on page {page_number}: {e}")

        return table_bboxes, tables_md

    def _extract_images(
        self,
        page: fitz.Page,
        page_number: int
    ) -> List[RawImage]:
        """
        Extracts embedded raster images from a page via PyMuPDF.
        Applies minimum size filter (MIN_IMAGE_WIDTH / MIN_IMAGE_HEIGHT).
        """
        images: List[RawImage] = []
        doc = page.parent

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                w = base_image.get("width", 0)
                h = base_image.get("height", 0)

                if w < settings.MIN_IMAGE_WIDTH or h < settings.MIN_IMAGE_HEIGHT:
                    logger.debug(f"Skipping small image xref={xref} ({w}x{h})")
                    continue

                # Get bounding box of image on the page
                img_rects = page.get_image_rects(xref)
                bbox = tuple(img_rects[0]) if img_rects else (0, 0, 0, 0)

                raw_img = RawImage(
                    image_bytes=base_image["image"],
                    bbox=bbox,
                    page_number=page_number,
                    xref=xref,
                    width=w,
                    height=h
                )
                images.append(raw_img)

            except Exception as e:
                logger.warning(f"Failed to extract image xref={xref} on page {page_number}: {e}")

        return images

    def detect_headings(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Classifies text blocks as headings based on font-size thresholds and bold flag.
        Returns list of text blocks that are likely headings.
        """
        headings = []
        for tb in text_blocks:
            is_large = tb.font_size >= self.HEADING_FONT_SIZE_THRESHOLD
            is_bold = bool(tb.font_flags & self.BOLD_FLAG)
            is_short = len(tb.text) < 120  # Headings are rarely long sentences

            if (is_large or is_bold) and is_short:
                headings.append(tb)

        return headings

    def find_caption(
        self,
        image_bbox: Tuple[float, float, float, float],
        text_blocks: List[TextBlock],
        max_distance: float = 50.0
    ) -> Optional[str]:
        """
        Searches for a text caption near an image using spatial proximity.
        Looks for text blocks directly below the image within max_distance pixels.
        Also looks for blocks starting with 'Figure', 'Fig.', 'Diagram', 'Table'.
        """
        img_x0, img_y0, img_x1, img_y1 = image_bbox
        caption_patterns = re.compile(
            r"^(Figure|Fig\.?|Diagram|Schematic|Drawing|Chart|Illustration|Table)\s+[\d\w]",
            re.IGNORECASE
        )

        best_candidate: Optional[str] = None
        best_distance = float("inf")

        for tb in text_blocks:
            tx0, ty0, tx1, ty1 = tb.bbox

            # Check if block is directly below image and horizontally overlapping
            below = ty0 >= img_y1
            horizontally_overlaps = not (tx1 < img_x0 or tx0 > img_x1)
            distance = ty0 - img_y1

            if below and horizontally_overlaps and distance < max_distance:
                if distance < best_distance:
                    best_distance = distance
                    best_candidate = tb.text

            # Also match explicit caption labels anywhere on the same page
            if caption_patterns.match(tb.text.strip()):
                if best_candidate is None:
                    best_candidate = tb.text

        return best_candidate

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _table_to_markdown(table: List[List]) -> str:
        """Converts a pdfplumber table (list of rows) to a Markdown table string."""
        if not table:
            return ""

        rows = []
        for i, row in enumerate(table):
            # Replace None cells with empty string
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            rows.append("| " + " | ".join(cells) + " |")
            # Insert separator after header row
            if i == 0:
                separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                rows.append(separator)

        return "\n".join(rows)

    @staticmethod
    def _overlaps_table(
        block_bbox: Tuple,
        table_bboxes: List[Tuple],
        overlap_threshold: float = 0.5
    ) -> bool:
        """Returns True if a text block significantly overlaps with any table region."""
        bx0, by0, bx1, by1 = block_bbox
        block_area = max((bx1 - bx0) * (by1 - by0), 1)

        for tx0, ty0, tx1, ty1 in table_bboxes:
            ix0 = max(bx0, tx0)
            iy0 = max(by0, ty0)
            ix1 = min(bx1, tx1)
            iy1 = min(by1, ty1)

            if ix1 > ix0 and iy1 > iy0:
                intersection = (ix1 - ix0) * (iy1 - iy0)
                if intersection / block_area >= overlap_threshold:
                    return True

        return False
