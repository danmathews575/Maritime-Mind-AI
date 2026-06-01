"""
Image Retrieval Service — Hardened Multimodal Retrieval

Dual-path image retrieval with:
- Path 1: CLIP semantic search (query-preprocessed for technical diagrams)
- Path 2: Text-association expansion (from retrieved text chunks)

Hardened with:
- Rebalanced scoring (text-association > CLIP for technical diagrams)
- Compound maritime term matching for subsystem boosting
- CLIP query preprocessing for better cross-modal precision
- Minimum relevance threshold to eliminate low-quality images
- Diagram-type weighting for engineering schematics
"""
import re
from typing import Dict, List, Tuple

from app.models.schemas import (
    ImageExplainability,
    ImageMetadata,
    RetrievalResult,
    RetrievedImage,
    TextChunk,
)
from app.services.clip_embedding import ImageEmbeddingService
from app.services.vector_store import VectorStoreService
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.image_retrieval")

# Minimum final score for an image to be returned
MIN_IMAGE_RELEVANCE_SCORE = 0.15

# Maritime compound terms for subsystem matching
COMPOUND_MARITIME_TERMS = [
    "cooling water", "lube oil", "lubricating oil", "fuel oil",
    "sea water", "fresh water", "exhaust gas", "main engine",
    "auxiliary engine", "steering gear", "fire fighting",
    "cargo hold", "ballast water", "bilge system", "deck machinery",
    "engine room", "bridge equipment", "life saving", "fire protection",
    "starting air", "control air", "compressed air", "hydraulic system",
    "shore connection", "sewage treatment", "oily water separator",
    "incinerator system", "stern tube", "propeller shaft",
]

# Words to strip from queries before CLIP encoding
CLIP_STRIP_WORDS = frozenset({
    "show", "explain", "display", "provide", "what", "how", "where",
    "when", "why", "me", "the", "a", "an", "please", "can", "you",
    "tell", "describe", "give", "find", "get", "look", "see",
    "is", "are", "was", "were", "do", "does", "did",
})


class ImageRetrievalService:
    def __init__(
        self,
        vector_store: VectorStoreService,
        clip_embedder: ImageEmbeddingService,
    ):
        self.vs = vector_store
        self.clip = clip_embedder

    def search(
        self,
        query: str,
        text_results: List[RetrievalResult],
        top_k: int = 5,
        filters: dict = None,
    ) -> List[RetrievedImage]:
        """
        Dual-path image retrieval with hardened scoring.

        Path 1: CLIP semantic search with preprocessed query
        Path 2: Association expansion from text retrieval results
        """
        # Path 1: Direct CLIP Semantic Search (with preprocessed query)
        clip_query = self._simplify_for_clip(query)
        clip_embedding = self.clip.embed_text_for_image_search(clip_query)
        clip_hits = self.vs.query_images(clip_embedding, top_k=top_k * 2)

        # Path 2: Association Expansion
        associated_image_ids = set()
        chunk_confidence_map: Dict[str, float] = {}
        for res in text_results:
            chunk = res.chunk
            conf = res.scores.confidence_score
            for img_id in chunk.related_image_ids:
                associated_image_ids.add(img_id)
                chunk_confidence_map[img_id] = max(
                    chunk_confidence_map.get(img_id, 0.0), conf
                )

        associated_images = self.vs.get_images_by_ids(list(associated_image_ids))

        # Path 3: Keyword payload search (OCR/Caption)
        query_lower = query.lower()
        search_terms = []
        for term in COMPOUND_MARITIME_TERMS:
            if term in query_lower:
                search_terms.append(term)
        if not search_terms:
            words = [w for w in query_lower.split() if w not in CLIP_STRIP_WORDS and len(w) > 4]
            if words:
                search_terms.append(" ".join(words))

        keyword_image_ids = set()
        keyword_images = []
        for term in search_terms:
            k_hits = self.vs.search_images_by_keyword(term, limit=top_k)
            keyword_images.extend(k_hits)
            for img in k_hits:
                keyword_image_ids.add(img.image_id)

        # Merge results and deduplicate
        merged_metadata: Dict[str, ImageMetadata] = {}
        clip_scores: Dict[str, float] = {}

        # Fetch proper schemas for clip hits
        clip_hit_ids = [hit["id"] for hit in clip_hits]
        clip_images = self.vs.get_images_by_ids(clip_hit_ids)

        # Map raw chroma distances to clip scores (similarity)
        for hit in clip_hits:
            clip_scores[hit["id"]] = max(0.0, 1.0 - hit["distance"])

        for img in clip_images:
            merged_metadata[img.image_id] = img
        for img in associated_images:
            merged_metadata[img.image_id] = img
        for img in keyword_images:
            merged_metadata[img.image_id] = img

        # Score and build results
        retrieved_images: List[RetrievedImage] = []

        for img_id, img_meta in merged_metadata.items():
            reasons = []
            c_score = clip_scores.get(img_id, 0.0)
            a_score = chunk_confidence_map.get(img_id, 0.0)
            k_score = 0.60 if img_id in keyword_image_ids else 0.0

            if img_id in clip_scores:
                reasons.append("Path 1: CLIP Semantic Match")
            if img_id in associated_image_ids:
                reasons.append("Path 2: Associated with retrieved text")
            if img_id in keyword_image_ids:
                reasons.append("Path 3: Keyword Match (OCR/Caption)")

            # Diagram Type Weighting
            diagram_weight = self._get_diagram_weight(img_meta)

            # Subsystem Boosting (compound-term-aware)
            subsystem_boost = self._calculate_subsystem_boost(
                query, img_meta, text_results
            )
            if subsystem_boost > 0:
                reasons.append("Subsystem Match Boost")

            # Page Proximity Boosting
            proximity_boost = self._calculate_proximity_boost(
                img_meta, text_results
            )
            if proximity_boost > 0:
                reasons.append(f"Page Proximity Boost (+{proximity_boost:.2f})")

            # ── Rebalanced Scoring Formula ────────────────────────────────
            # Keyword and Text-association weights are dominant because
            # technical terminology inside diagrams is highly diagnostic.
            # CLIP weight is lower since ViT-B-32 was trained on natural images.
            raw_score = (
                (0.25 * c_score)
                + (0.35 * a_score)
                + (0.40 * k_score)
                + proximity_boost
                + (0.10 * img_meta.diagram_confidence)
                + subsystem_boost
            )

            final_score = raw_score * diagram_weight

            expl = ImageExplainability(
                retrieval_reason=reasons,
                clip_score=c_score,
                association_score=a_score,
                subsystem_match=(subsystem_boost > 0),
                source_pdf=img_meta.manual_name,
                page=img_meta.page_number,
                section_title=img_meta.section_title,
                final_score=final_score,
            )

            retrieved_images.append(
                RetrievedImage(metadata=img_meta, explainability=expl)
            )

        # Sort by final score descending
        retrieved_images.sort(
            key=lambda x: x.explainability.final_score, reverse=True
        )

        # Apply metadata filters (e.g. ship_id)
        if filters and "ship_id" in filters:
            ship_id = filters["ship_id"]
            retrieved_images = [
                img for img in retrieved_images
                if img.metadata.ship_id == ship_id
            ]

        # Apply minimum relevance threshold
        retrieved_images = [
            img
            for img in retrieved_images
            if img.explainability.final_score >= MIN_IMAGE_RELEVANCE_SCORE
        ]

        if len(retrieved_images) > top_k:
            retrieved_images = retrieved_images[:top_k]

        logger.info(
            f"Image retrieval: {len(retrieved_images)} images returned "
            f"(threshold={MIN_IMAGE_RELEVANCE_SCORE})"
        )

        return retrieved_images

    def _simplify_for_clip(self, query: str) -> str:
        """
        Convert complex maritime queries to CLIP-friendly short descriptions.
        CLIP was trained on short image captions ("a photo of X"), not complex
        technical questions. Simplifying the query improves cross-modal precision.
        """
        # Remove common question/action words
        words = query.lower().split()
        filtered = [w for w in words if w not in CLIP_STRIP_WORDS]

        simplified = " ".join(filtered).strip()

        # If we stripped everything, use original
        if len(simplified) < 3:
            simplified = query

        # Prefix with technical context for engineering diagrams
        simplified = f"technical diagram of {simplified}"

        return simplified

    def _get_diagram_weight(self, img: ImageMetadata) -> float:
        """
        Weight multiplier based on diagram type. Engineering schematics
        receive higher weight than generic figures.
        """
        text = f"{img.caption} {' '.join(img.tags)} {img.ocr_text}".lower()

        if "piping" in text or "pipe" in text:
            return 1.2
        if "wiring" in text or "electrical" in text or "circuit" in text:
            return 1.2
        if "flow chart" in text or "flowchart" in text or "flow" in text:
            return 1.15
        if "layout" in text or "map" in text or "schematic" in text:
            return 1.15
        if "cross section" in text or "cross-section" in text:
            return 1.1
        if "radar" in text or "ecdis" in text or "interface" in text:
            return 1.1
        if "diagram" in text or "figure" in text or "fig" in text:
            return 1.05
        return 1.0

    def _calculate_subsystem_boost(
        self,
        query: str,
        img: ImageMetadata,
        text_results: List[RetrievalResult],
    ) -> float:
        """
        Subsystem matching with compound maritime term awareness.
        Uses both single-word and multi-word term matching to avoid
        missing compound subsystem names like "cooling water" or "lube oil".
        """
        img_text = (
            f"{img.caption} {' '.join(img.tags)} "
            f"{img.section_title} {img.ocr_text}"
        ).lower()
        query_lower = query.lower()

        # 1. Check compound maritime terms in both query and image metadata
        for term in COMPOUND_MARITIME_TERMS:
            if term in query_lower and term in img_text:
                return 0.15

        # 2. Check against text results subsystem/department
        for res in text_results:
            if res.chunk.subsystem and res.chunk.subsystem != "general":
                if res.chunk.subsystem.lower() in img_text:
                    return 0.15
            if res.chunk.department and res.chunk.department != "general":
                if res.chunk.department.lower() in img_text:
                    return 0.10

        # 3. Specific keyword overlap (>5 chars to avoid false positives)
        stop_words = {
            "show", "explain", "please", "provide", "diagram",
            "figure", "picture", "display", "system", "equipment",
        }
        query_words = [
            w.lower() for w in query.split() if len(w) > 5 and w.lower() not in stop_words
        ]

        for word in query_words:
            if word in img_text:
                return 0.10

        return 0.0

    def _calculate_proximity_boost(
        self,
        img: ImageMetadata,
        text_results: List[RetrievalResult],
    ) -> float:
        """Page proximity boost between image and retrieved text chunks."""
        best_boost = 0.0
        for res in text_results:
            if res.chunk.manual_name == img.manual_name:
                diff = abs(res.chunk.page_number - img.page_number)
                if diff == 0:
                    best_boost = max(best_boost, 0.25)
                elif diff == 1:
                    best_boost = max(best_boost, 0.15)
                elif diff == 2:
                    best_boost = max(best_boost, 0.05)
        return best_boost
