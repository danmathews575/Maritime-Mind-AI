"""
tests/test_ingestion.py — Phase 2 Integration Tests

Tests the full Phase 2 ingestion pipeline including:
- PDF parser (with synthetic in-memory PDF)
- Chunker (with synthetic ParsedDocument)
- Image extractor (with synthetic image bytes)
- Association engine
- Manifest tracking
- Pipeline orchestrator (end-to-end with real or synthetic PDF)
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal synthetic PDF builder (no PyMuPDF dependency for unit tests)
# ---------------------------------------------------------------------------

def _make_minimal_pdf(text: str = "Chapter 1\nThis is a test document.") -> bytes:
    """
    Generates a valid 1-page PDF with embedded text.
    Uses only stdlib — no PyMuPDF required.
    """
    # This creates a bare-minimum but spec-valid PDF
    objects = []
    
    def obj(n: int, s: str) -> str:
        return f"{n} 0 obj\n{s}\nendobj\n"
    
    stream_content = f"BT /F1 12 Tf 50 750 Td ({text[:200]}) Tj ET"
    stream_bytes = stream_content.encode("latin-1")
    
    o1 = obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
    o2 = obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    o3 = obj(3, (
        "<< /Type /Page /Parent 2 0 R "
        "/MediaBox [0 0 612 792] "
        "/Contents 4 0 R "
        "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>"
    ))
    o4 = obj(4, f"<< /Length {len(stream_bytes)} >>\nstream\n{stream_content}\nendstream")
    
    body = o1 + o2 + o3 + o4
    header = "%PDF-1.4\n"
    xref_offset = len(header) + len(body)
    
    pdf = (
        header + body +
        "xref\n0 5\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000115 00000 n \n"
        "0000000266 00000 n \n"
        f"trailer\n<< /Size 5 /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    )
    return pdf.encode("latin-1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_pdf(tmp_path) -> Path:
    """Creates a temporary minimal PDF file for testing."""
    pdf_bytes = _make_minimal_pdf("Chapter 1\nEngine Room Maintenance\nStep 1: Check oil level.\nStep 2: Inspect filters.")
    pdf_file = tmp_path / "test_manual.pdf"
    pdf_file.write_bytes(pdf_bytes)
    return pdf_file


@pytest.fixture
def tmp_pdf_dir(tmp_path) -> Path:
    """Creates a directory with two minimal PDFs."""
    for name in ["manual_a.pdf", "manual_b.pdf"]:
        pdf_bytes = _make_minimal_pdf(f"Content of {name}")
        (tmp_path / name).write_bytes(pdf_bytes)
    return tmp_path


@pytest.fixture
def synthetic_parsed_doc():
    """Returns a synthetic ParsedDocument for testing the chunker without PDF I/O."""
    from app.services.pdf_parser import (
        ParsedDocument, ParsedPage, TextBlock, RawImage
    )
    page1 = ParsedPage(
        page_number=1,
        text_blocks=[
            TextBlock(
                text="Chapter 1: Engine Room Maintenance",
                font_size=16.0,
                font_flags=16,  # bold
                bbox=(50.0, 700.0, 400.0, 720.0),
                page_number=1,
            ),
            TextBlock(
                text="The engine room requires regular inspection. "
                     "Check oil levels daily. Inspect all fluid connections weekly.",
                font_size=10.0,
                font_flags=0,
                bbox=(50.0, 650.0, 550.0, 680.0),
                page_number=1,
            ),
        ],
        tables=["| Parameter | Value |\n|---|---|\n| Oil Pressure | 4.5 bar |"],
        images=[
            RawImage(
                image_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
                bbox=(300.0, 400.0, 550.0, 600.0),
                page_number=1,
                xref=1,
                width=300,
                height=300,
            )
        ],
    )
    return ParsedDocument(
        manual_name="test_manual",
        pdf_path="test_manual.pdf",
        pages=[page1],
        total_images_extracted=1,
        total_tables_extracted=1,
    )


# ---------------------------------------------------------------------------
# Tests: Manifest
# ---------------------------------------------------------------------------

class TestIngestionManifest:
    """Tests for the IngestionManifest service."""

    def test_empty_manifest_on_first_load(self, tmp_path):
        from app.services.manifest import IngestionManifest
        with patch("app.services.manifest.settings") as mock_settings:
            mock_settings.METADATA_DIR = str(tmp_path)
            manifest = IngestionManifest()
            data = manifest.load()
        assert data == {}

    def test_update_and_is_processed(self, tmp_path):
        from app.services.manifest import IngestionManifest
        with patch("app.services.manifest.settings") as mock_settings:
            mock_settings.METADATA_DIR = str(tmp_path)
            manifest = IngestionManifest()
            assert not manifest.is_processed("my_manual")
            manifest.update(
                manual_name="my_manual",
                status="COMPLETED",
                chunk_count=42,
                image_count=7,
                errors=[],
            )
            assert manifest.is_processed("my_manual")

    def test_manifest_persists_to_json(self, tmp_path):
        from app.services.manifest import IngestionManifest
        with patch("app.services.manifest.settings") as mock_settings:
            mock_settings.METADATA_DIR = str(tmp_path)
            manifest = IngestionManifest()
            manifest.update("doc_a", "COMPLETED", 10, 3, [])
            # Re-instantiate to verify persistence
            manifest2 = IngestionManifest()
            data = manifest2.load()
        assert "doc_a" in data
        assert data["doc_a"]["status"] == "COMPLETED"
        assert data["doc_a"]["chunk_count"] == 10

    def test_failed_status_not_marked_processed(self, tmp_path):
        from app.services.manifest import IngestionManifest
        with patch("app.services.manifest.settings") as mock_settings:
            mock_settings.METADATA_DIR = str(tmp_path)
            manifest = IngestionManifest()
            manifest.update("bad_doc", "FAILED", 0, 0, ["parse error"])
            assert not manifest.is_processed("bad_doc")


# ---------------------------------------------------------------------------
# Tests: Chunker
# ---------------------------------------------------------------------------

class TestChunkerService:
    """Tests for the ChunkerService with synthetic ParsedDocuments."""

    def test_chunks_produced(self, synthetic_parsed_doc):
        from app.services.chunker import SemanticChunkerService
        chunker = SemanticChunkerService()
        chunks = chunker.chunk_document(synthetic_parsed_doc)
        assert len(chunks) > 0, "Expected at least one chunk"

    def test_chunk_ids_are_unique(self, synthetic_parsed_doc):
        from app.services.chunker import SemanticChunkerService
        chunker = SemanticChunkerService()
        chunks = chunker.chunk_document(synthetic_parsed_doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_table_preserved_as_single_chunk(self, synthetic_parsed_doc):
        from app.services.chunker import SemanticChunkerService
        chunker = SemanticChunkerService()
        chunks = chunker.chunk_document(synthetic_parsed_doc)
        # Find the table chunk — it should contain the pipe characters
        table_chunks = [c for c in chunks if "|" in c.content and "Parameter" in c.content]
        assert len(table_chunks) >= 1, "Table should produce at least one chunk"
        # The table content should not be split mid-row
        for tc in table_chunks:
            assert "Oil Pressure" in tc.content or "Parameter" in tc.content

    def test_chunks_have_page_numbers(self, synthetic_parsed_doc):
        from app.services.chunker import SemanticChunkerService
        chunker = SemanticChunkerService()
        chunks = chunker.chunk_document(synthetic_parsed_doc)
        for chunk in chunks:
            assert chunk.page_number >= 1

    def test_chunks_linked_previous_next(self, synthetic_parsed_doc):
        from app.services.chunker import SemanticChunkerService
        chunker = SemanticChunkerService()
        chunks = chunker.chunk_document(synthetic_parsed_doc)
        if len(chunks) > 1:
            # First chunk should have no previous
            assert chunks[0].previous_chunk_id is None
            # Last chunk should have no next
            assert chunks[-1].next_chunk_id is None
            # Middle chunks should be linked
            for i in range(1, len(chunks) - 1):
                assert chunks[i].previous_chunk_id is not None
                assert chunks[i].next_chunk_id is not None

    def test_hierarchy_path_set_from_heading(self, synthetic_parsed_doc):
        from app.services.chunker import SemanticChunkerService
        chunker = SemanticChunkerService()
        chunks = chunker.chunk_document(synthetic_parsed_doc)
        # Body text chunks should reference the heading
        body_chunks = [c for c in chunks if ("engine" in c.content.lower() or "oil" in c.content.lower()) and not c.content.startswith("[TABLE")]
        for bc in body_chunks:
            assert bc.hierarchy_path is not None and len(bc.hierarchy_path) > 0


# ---------------------------------------------------------------------------
# Tests: Association Engine
# ---------------------------------------------------------------------------

class TestAssociationEngine:
    """Tests for the text-image bidirectional association engine."""

    def _make_chunk(self, chunk_id: str, page: int, content: str):
        from app.models.schemas import TextChunk
        return TextChunk(
            chunk_id=chunk_id,
            manual_name="test_manual",
            content=content,
            page_number=page,
            department="engineering",
            section_title="Test",
            hierarchy_path=["Chapter 1"],
            related_image_ids=[],
            embedding_model="test-model",
        )

    def _make_image(self, image_id: str, page: int):
        from app.models.schemas import ImageMetadata
        return ImageMetadata(
            image_id=image_id,
            manual_name="test_manual",
            page_number=page,
            image_path=f"/tmp/{image_id}.png",
            caption="",
            bbox=None,
            related_chunk_ids=[],
            embedding_model="clip-test",
        )

    def test_spatial_association_same_page(self):
        from app.services.association import AssociationEngine
        engine = AssociationEngine()
        chunks = [self._make_chunk("c1", 1, "Engine inspection procedure.")]
        images = [self._make_image("img1", 1)]
        linked_chunks, linked_images = engine.associate(chunks, images)
        assert "img1" in linked_chunks[0].related_image_ids
        assert "c1" in linked_images[0].related_chunk_ids

    def test_no_association_different_page(self):
        from app.services.association import AssociationEngine
        engine = AssociationEngine()
        chunks = [self._make_chunk("c1", 1, "Text on page 1.")]
        images = [self._make_image("img1", 5)]
        linked_chunks, linked_images = engine.associate(chunks, images)
        assert "img1" not in linked_chunks[0].related_image_ids

    def test_textual_reference_association(self):
        from app.services.association import AssociationEngine
        engine = AssociationEngine()
        # Chunk on page 1 references Figure 3; image is on page 2 (within ±1 page of page 1 is required)
        chunks = [self._make_chunk("c1", 1, "As shown in Figure 3, the valve assembly is critical.")]
        images = [self._make_image("img3", 2)]
        images[0].caption = "Figure 3: Valve Assembly"
        linked_chunks, linked_images = engine.associate(chunks, images)
        # Textual reference should create a link even across pages
        assert "img3" in linked_chunks[0].related_image_ids

    def test_bidirectional_consistency(self):
        from app.services.association import AssociationEngine
        engine = AssociationEngine()
        chunks = [
            self._make_chunk("c1", 2, "See Diagram 1 for wiring details."),
            self._make_chunk("c2", 2, "Additional notes on page 2."),
        ]
        images = [self._make_image("img_d1", 2)]
        linked_chunks, linked_images = engine.associate(chunks, images)
        # Image should reference all chunks that link to it
        for chunk in linked_chunks:
            if "img_d1" in chunk.related_image_ids:
                assert chunk.chunk_id in linked_images[0].related_chunk_ids


# ---------------------------------------------------------------------------
# Tests: Image Extractor
# ---------------------------------------------------------------------------

class TestImageExtractorService:
    """Tests for the ImageExtractorService."""

    def test_filter_small_image(self, tmp_path, synthetic_parsed_doc):
        from app.services.image_extractor import ImageExtractorService
        with patch("app.services.image_extractor.settings") as mock_settings:
            mock_settings.MIN_IMAGE_WIDTH = 50
            mock_settings.MIN_IMAGE_HEIGHT = 50
            mock_settings.EXTRACTED_IMAGES_DIR = str(tmp_path / "images")

            extractor = ImageExtractorService()

            # Small images should be filtered
            result = extractor._passes_size_filter(5, 5)
            assert result is False  # False = does not pass = should filter

    def test_image_id_is_sha256(self):
        from app.services.image_extractor import ImageExtractorService
        with patch("app.services.image_extractor.settings"):
            extractor = ImageExtractorService()
            data = b"fake_image_bytes"
            expected = hashlib.sha256(data + b"1").hexdigest()[:16]
            assert extractor._generate_image_id(data, 1) == expected


# ---------------------------------------------------------------------------
# Tests: Pipeline Integration (with mocked parser)
# ---------------------------------------------------------------------------

class TestIngestionPipeline:
    """Integration tests for the full IngestionPipeline orchestrator."""

    def _make_pipeline_with_mocks(self, tmp_path, synthetic_parsed_doc):
        """Create a pipeline with parser mocked to return synthetic_parsed_doc."""
        from app.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline.settings = MagicMock()
        pipeline.settings.DATA_DIR = str(tmp_path)

        pipeline.parser = MagicMock()
        pipeline.parser.parse_pdf.return_value = synthetic_parsed_doc

        pipeline.chunker = MagicMock()
        from app.models.schemas import TextChunk
        fake_chunk = TextChunk(
            chunk_id="abc123",
            manual_name="test_manual",
            content="Engine room maintenance procedure.",
            page_number=1,
            department="test",
            section_title="Chapter 1",
            hierarchy_path=["Chapter 1"],
            related_image_ids=[],
            embedding_model="test-model",
        )
        pipeline.chunker.chunk_document.return_value = [fake_chunk]

        pipeline.image_extractor = MagicMock()
        from app.models.schemas import ImageMetadata
        fake_image = ImageMetadata(
            image_id="imgabc",
            manual_name="test_manual",
            page_number=1,
            image_path=str(tmp_path / "img.png"),
            caption="",
            bbox=None,
            related_chunk_ids=[],
            embedding_model="clip-test",
        )
        pipeline.image_extractor.extract_and_save.return_value = [fake_image]

        pipeline.association = MagicMock()
        fake_chunk.related_image_ids = ["imgabc"]
        fake_image.related_chunk_ids = ["abc123"]
        pipeline.association.associate.return_value = ([fake_chunk], [fake_image])

        pipeline.text_embedder = MagicMock()
        pipeline.image_embedder = MagicMock()
        pipeline.vector_store = MagicMock()
        pipeline.bm25_index = MagicMock()
        pipeline.eval_data = []

        # Real manifest backed by tmp_path
        with patch("app.services.manifest.settings") as ms:
            ms.METADATA_DIR = str(tmp_path)
            from app.services.manifest import IngestionManifest
            pipeline.manifest = IngestionManifest()

        return pipeline

    def test_successful_run(self, tmp_path, synthetic_parsed_doc):
        """Full pipeline run with mocked parser returns IngestionResult(success=True)."""
        from app.ingestion.pipeline import IngestionPipeline
        import time

        pipeline = self._make_pipeline_with_mocks(tmp_path, synthetic_parsed_doc)

        # Create a dummy PDF file
        pdf_path = tmp_path / "test_manual.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        result = pipeline._run_pipeline("test_manual", str(pdf_path), "test", time.perf_counter())
        assert result.success is True
        assert result.chunk_count == 1
        assert result.image_count == 1

    def test_idempotency_skips_completed(self, tmp_path, synthetic_parsed_doc):
        """Re-running on an already-completed manual should return skipped=True."""
        pipeline = self._make_pipeline_with_mocks(tmp_path, synthetic_parsed_doc)

        # Pre-mark as completed
        pipeline.manifest.update("test_manual", "COMPLETED", 10, 3, [])

        pdf_path = tmp_path / "test_manual.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        result = pipeline.run(str(pdf_path), force=False)
        assert result.skipped is True
        assert result.success is True
        pipeline.parser.parse_pdf.assert_not_called()

    def test_force_flag_bypasses_skip(self, tmp_path, synthetic_parsed_doc):
        """force=True should re-run even if manifest says COMPLETED."""
        import time
        pipeline = self._make_pipeline_with_mocks(tmp_path, synthetic_parsed_doc)
        pipeline.manifest.update("test_manual", "COMPLETED", 10, 3, [])

        pdf_path = tmp_path / "test_manual.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        result = pipeline.run(str(pdf_path), force=True)
        assert result.skipped is False
        pipeline.parser.parse_pdf.assert_called_once()

    def test_missing_pdf_returns_failure(self, tmp_path, synthetic_parsed_doc):
        """run() on nonexistent file should return success=False without raising."""
        pipeline = self._make_pipeline_with_mocks(tmp_path, synthetic_parsed_doc)
        result = pipeline.run(str(tmp_path / "nonexistent.pdf"))
        assert result.success is False
        assert result.error is not None

    def test_run_directory_empty(self, tmp_path, synthetic_parsed_doc):
        """Empty directory should return empty list without error."""
        pipeline = self._make_pipeline_with_mocks(tmp_path, synthetic_parsed_doc)
        results = pipeline.run_directory(str(tmp_path))
        assert results == []
