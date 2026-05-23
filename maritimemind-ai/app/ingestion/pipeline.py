"""
Ingestion Pipeline Orchestrator — Final Retrieval-Aware Architecture
Ties together all semantic services into a single coherent pipeline:
    PDF Parser → Semantic Chunker → Image Extractor → Association Engine → Manifest
"""
from __future__ import annotations

import os
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.configs.config import get_settings
from app.services.pdf_parser import PdfParserService, ParsedDocument
from app.services.chunker import SemanticChunkerService
from app.services.image_extractor import ImageExtractorService
from app.services.association import AssociationEngine
from app.services.manifest import IngestionManifest
from app.services.embedding import TextEmbeddingService
from app.services.clip_embedding import ImageEmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.bm25_index import BM25IndexService
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.ingestion.pipeline")

BATCH_SIZE = 50 # Process 50 pages at a time to prevent OOM on large manuals

@dataclass
class IngestionResult:
    manual_name: str
    pdf_path: str
    success: bool
    chunk_count: int = 0
    image_count: int = 0
    page_count: int = 0
    table_count: int = 0
    text_embedded_count: int = 0
    images_embedded_count: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    skipped: bool = False
    chunk_ids: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.skipped: return f"[SKIPPED] {self.manual_name}"
        if not self.success: return f"[FAILED]  {self.manual_name} — {self.error}"
        return f"[OK] {self.manual_name} | {self.page_count} pages | {self.chunk_count} chunks | {self.image_count} images"


class IngestionPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.parser = PdfParserService()
        self.chunker = SemanticChunkerService()
        self.image_extractor = ImageExtractorService()
        self.association = AssociationEngine()
        self.text_embedder = TextEmbeddingService()
        self.image_embedder = ImageEmbeddingService()
        self.vector_store = VectorStoreService()
        self.bm25_index = BM25IndexService()
        self.manifest = IngestionManifest()
        self.eval_data = []
        logger.info("IngestionPipeline initialised — all semantic services ready")

    def run(self, pdf_path: str, force: bool = False) -> IngestionResult:
        pdf_path = str(Path(pdf_path).resolve())
        manual_name = Path(pdf_path).stem
        # Derive department from parent directory
        parent_dir = Path(pdf_path).parent.name
        department = parent_dir if parent_dir in ["deck", "engineering", "navigation", "safety"] else "general"

        if not force and self.manifest.is_processed(manual_name):
            logger.info(f"Skipping already-ingested manual: {manual_name}")
            return IngestionResult(manual_name=manual_name, pdf_path=pdf_path, success=True, skipped=True)

        if not os.path.isfile(pdf_path):
            err = f"PDF not found: {pdf_path}"
            logger.error(err)
            self.manifest.update(manual_name=manual_name, status="FAILED", chunk_count=0, image_count=0, errors=[err])
            return IngestionResult(manual_name=manual_name, pdf_path=pdf_path, success=False, error=err)

        logger.info(f"Starting ingestion: {manual_name} ({pdf_path})")
        t_start = time.perf_counter()

        try:
            return self._run_pipeline(manual_name, pdf_path, department, t_start)
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            err_msg = str(exc)
            err_tb = traceback.format_exc()
            logger.error(f"Pipeline failed for {manual_name}: {err_msg}\n{err_tb}")
            self.manifest.update(manual_name=manual_name, status="FAILED", chunk_count=0, image_count=0, errors=[err_msg])
            return IngestionResult(manual_name=manual_name, pdf_path=pdf_path, success=False, elapsed_seconds=elapsed, error=err_msg, error_traceback=err_tb)

    def run_directory(self, pdf_dir: str, force: bool = False, extensions: tuple[str, ...] = (".pdf",)) -> List[IngestionResult]:
        pdf_dir_path = Path(pdf_dir).resolve()
        if not pdf_dir_path.is_dir():
            logger.error(f"Directory not found: {pdf_dir_path}")
            return []

        # Recursively find PDFs (fixes bug where subdirs were ignored)
        pdf_files = sorted(
            p for p in pdf_dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        if not pdf_files:
            logger.warning(f"No PDF files found in: {pdf_dir_path}")
            return []

        logger.info(f"Batch ingestion: {len(pdf_files)} file(s) in {pdf_dir_path}")
        results: List[IngestionResult] = []

        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            result = self.run(str(pdf_file), force=force)
            results.append(result)
            logger.info(str(result))

        # Rebuild BM25 globally
        logger.info("Rebuilding BM25 index from full vector store corpus...")
        all_chunks = self.vector_store.get_all_text_chunks()
        if all_chunks:
            self.bm25_index.build_index(all_chunks)
            self.bm25_index.save()
            logger.info("BM25 index rebuild complete.")
            
        # Save Eval Dataset
        if self.eval_data:
            eval_path = Path(self.settings.DATA_DIR) / "eval_dataset.json"
            eval_path.parent.mkdir(exist_ok=True)
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(self.eval_data, f, indent=2)
            logger.info(f"Saved {len(self.eval_data)} synthetic evaluation queries to {eval_path}")

        ok = sum(1 for r in results if r.success and not r.skipped)
        skipped = sum(1 for r in results if r.skipped)
        failed = sum(1 for r in results if not r.success)
        logger.info(f"Batch complete — OK:{ok} SKIPPED:{skipped} FAILED:{failed}")
        return results

    def _run_pipeline(self, manual_name: str, pdf_path: str, department: str, t_start: float) -> IngestionResult:
        logger.info(f"[{manual_name}] Parsing PDF...")
        parsed_doc = self.parser.parse_pdf(pdf_path)
        
        total_pages = len(parsed_doc.pages)
        all_chunks = []
        all_images = []
        upserted_texts = 0
        upserted_images = 0
        
        # Incremental Batching for large documents
        for i in range(0, total_pages, BATCH_SIZE):
            batch_pages = parsed_doc.pages[i:i+BATCH_SIZE]
            logger.info(f"[{manual_name}] Processing batch {i}-{min(i+BATCH_SIZE, total_pages)} of {total_pages} pages")
            
            batch_doc = ParsedDocument(
                manual_name=manual_name,
                pdf_path=pdf_path,
                pages=batch_pages, 
                total_images_extracted=0, 
                total_tables_extracted=0
            )
            
            chunks = self.chunker.chunk_document(batch_doc, department=department)
            all_chunks.extend(chunks)
            
            images = self.image_extractor.extract_and_save(batch_doc)
            all_images.extend(images)
            
            # Incrementally commit text chunks to ChromaDB (before association, 
            # links will be back-filled by upsert after full-doc association)
            if chunks:
                text_embeddings = self.text_embedder.embed_batch([c.content for c in chunks])
                upserted_texts += self.vector_store.add_text_chunks(chunks, text_embeddings)
                
            if images:
                image_embeddings = self.image_embedder.embed_batch([img.image_path for img in images])
                upserted_images += self.vector_store.add_image_embeddings(images, image_embeddings)

        # ── Full-document association (after all batches) ──────────────────
        # CRITICAL: Must run on ALL chunks + images together, not per-batch,
        # to correctly link diagrams from page 80 to text from page 75.
        logger.info(f"[{manual_name}] Running full-document text-image association...")
        all_chunks, all_images = self.association.associate(all_chunks, all_images)
        linked_chunk_count = sum(1 for c in all_chunks if c.related_image_ids)
        logger.info(f"[{manual_name}] Association complete: {linked_chunk_count} chunks linked to images")

        # Re-upsert chunks with updated related_image_ids metadata
        if linked_chunk_count > 0:
            logger.info(f"[{manual_name}] Re-upserting {linked_chunk_count} linked chunks with image references...")
            linked_chunks = [c for c in all_chunks if c.related_image_ids]
            link_embeddings = self.text_embedder.embed_batch([c.content for c in linked_chunks])
            self.vector_store.add_text_chunks(linked_chunks, link_embeddings)

        # Re-upsert images with updated related_chunk_ids metadata
        linked_images_list = [img for img in all_images if img.related_chunk_ids]
        if linked_images_list:
            link_img_embeddings = self.image_embedder.embed_batch([img.image_path for img in linked_images_list])
            self.vector_store.add_image_embeddings(linked_images_list, link_img_embeddings)

        # Synthetic QA for Eval Dataset
        for c in all_chunks:
            if c.importance == "high" and len(c.content) > 100:
                self.eval_data.append({
                    "query": f"How do I troubleshoot or operate {c.subsystem} ({c.section_title})?",
                    "ground_truth_chunk_id": c.chunk_id,
                    "department": c.department
                })
                
        elapsed = time.perf_counter() - t_start
        self.manifest.update(
            manual_name=manual_name,
            status="COMPLETED",
            chunk_count=len(all_chunks),
            image_count=len(all_images),
            errors=[],
        )
        logger.info(f"[{manual_name}] Ingestion complete in {elapsed:.1f}s")

        return IngestionResult(
            manual_name=manual_name,
            pdf_path=pdf_path,
            success=True,
            chunk_count=len(all_chunks),
            image_count=len(all_images),
            page_count=total_pages,
            table_count=parsed_doc.total_tables_extracted,
            text_embedded_count=upserted_texts,
            images_embedded_count=upserted_images,
            elapsed_seconds=elapsed,
            chunk_ids=[c.chunk_id for c in all_chunks],
            image_ids=[img.image_id for img in all_images],
        )
