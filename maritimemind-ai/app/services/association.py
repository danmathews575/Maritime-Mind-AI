"""
Text-Image Association Engine — Final Retrieval-Aware Architecture
Bidirectionally links TextChunks and ImageMetadata objects using:
  1. Spatial proximity rule
  2. Textual reference rule
  3. Semantic tagging of images based on linked chunk keywords
"""
from __future__ import annotations

import re
from typing import List, Tuple

from app.models.schemas import ImageMetadata, TextChunk
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.association")

FIGURE_PATTERNS = [
    re.compile(r"(?:Figure|Fig\.?|Diagram|Schematic|Drawing|Chart|Illustration)\s*[\d\w][\d\w\-\.]*", re.IGNORECASE),
    re.compile(r"(?:see|refer\s+to|as\s+shown\s+in)\s+(?:Figure|Fig\.?|Diagram)\s*[\d\w][\d\w\-\.]*", re.IGNORECASE),
    re.compile(r"(?:Table)\s*[\d\w][\d\w\-\.]*", re.IGNORECASE),
]

class AssociationEngine:
    def associate(
        self, chunks: List[TextChunk], images: List[ImageMetadata]
    ) -> Tuple[List[TextChunk], List[ImageMetadata]]:
        if not chunks or not images:
            return chunks, images

        chunks_by_page = self._index_by_page(chunks)
        images_by_page = self._index_by_page(images)

        for page_num, page_images in images_by_page.items():
            # Use ±1 page window — many technical manuals place diagrams on
            # image-only pages with caption text on the preceding/following page.
            candidate_page_nums = [page_num - 1, page_num, page_num + 1]
            page_chunks = []
            for pn in candidate_page_nums:
                page_chunks.extend(chunks_by_page.get(pn, []))

            for img in page_images:
                for chunk in page_chunks:
                    self._link(chunk, img)
                
                # 2. Heuristic Captioning & Semantic Tagging
                if not img.caption and page_chunks:
                    # Look for explicit figure headers on the same page
                    for chunk in page_chunks:
                        if re.search(r"^(?:Figure|Fig\.?)\s+\d+", chunk.content, re.IGNORECASE):
                            img.caption = chunk.content[:200]  # first 200 chars as caption
                            break
                    if not img.caption:
                        # Fallback to the title of the first chunk on the page
                        img.caption = page_chunks[0].section_title
                        
                # Tagging based on chunk keywords
                tags = set(img.tags)
                for chunk in page_chunks:
                    if chunk.keywords:
                        tags.update(chunk.keywords)
                    if chunk.subsystem:
                        tags.add(chunk.subsystem)
                img.tags = list(tags)
                img.section_title = page_chunks[0].section_title if page_chunks else ""

        # 3. Textual reference rule — explicit mentions
        for chunk in chunks:
            referenced_labels = self._extract_figure_labels(chunk.content)
            if not referenced_labels:
                continue

            chunk.diagram_references.extend(referenced_labels)

            candidate_pages = {chunk.page_number - 1, chunk.page_number, chunk.page_number + 1}
            for page_num in candidate_pages:
                for img in images_by_page.get(page_num, []):
                    if img.caption:
                        for label in referenced_labels:
                            if label.lower() in img.caption.lower():
                                self._link(chunk, img)

        linked_chunks = sum(1 for c in chunks if c.related_image_ids)
        linked_images = sum(1 for i in images if i.related_chunk_ids)
        logger.info(f"Association complete: {linked_chunks} chunks linked, {linked_images} images linked.")

        return chunks, images

    @staticmethod
    def _link(chunk: TextChunk, image: ImageMetadata) -> None:
        if image.image_id not in chunk.related_image_ids:
            chunk.related_image_ids.append(image.image_id)
        if chunk.chunk_id not in image.related_chunk_ids:
            image.related_chunk_ids.append(chunk.chunk_id)

    @staticmethod
    def _index_by_page(items) -> dict:
        index = {}
        for item in items:
            page = item.page_number
            index.setdefault(page, []).append(item)
        return index

    @staticmethod
    def _extract_figure_labels(text: str) -> List[str]:
        labels = []
        for pattern in FIGURE_PATTERNS:
            labels.extend(pattern.findall(text))
        return list(set(labels))
