import re
from typing import Dict, List, Tuple
from app.models.schemas import RetrievedImage, ImageMetadata, ImageExplainability, RetrievalResult, TextChunk
from app.services.vector_store import VectorStoreService
from app.services.clip_embedding import ImageEmbeddingService
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.image_retrieval")

class ImageRetrievalService:
    def __init__(self, vector_store: VectorStoreService, clip_embedder: ImageEmbeddingService):
        self.vs = vector_store
        self.clip = clip_embedder

    def search(self, query: str, text_results: List[RetrievalResult], top_k: int = 5) -> List[RetrievedImage]:
        # Path 1: Direct CLIP Semantic Search
        clip_embedding = self.clip.embed_text_for_image_search(query)
        # Vector search returns distances (smaller is better). Convert to similarity.
        clip_hits = self.vs.query_images(clip_embedding, top_k=top_k * 2)
        
        # Path 2: Association Expansion
        associated_image_ids = set()
        chunk_confidence_map = {}
        for res in text_results:
            chunk = res.chunk
            conf = res.scores.confidence_score
            for img_id in chunk.related_image_ids:
                associated_image_ids.add(img_id)
                chunk_confidence_map[img_id] = max(chunk_confidence_map.get(img_id, 0.0), conf)
        
        associated_images = self.vs.get_images_by_ids(list(associated_image_ids))

        # Merge results and deduplicate
        merged_metadata: Dict[str, ImageMetadata] = {}
        clip_scores: Dict[str, float] = {}
        

        # Fetch proper schemas for clip hits
        clip_hit_ids = [hit["id"] for hit in clip_hits]
        clip_images = self.vs.get_images_by_ids(clip_hit_ids)
        
        # Map raw chroma distances to clip scores (similarity)
        for hit in clip_hits:
            # chroma cosine distance: similarity = 1 - distance
            clip_scores[hit["id"]] = max(0.0, 1.0 - hit["distance"])

        for img in clip_images:
            merged_metadata[img.image_id] = img
        for img in associated_images:
            merged_metadata[img.image_id] = img

        retrieved_images: List[RetrievedImage] = []

        for img_id, img_meta in merged_metadata.items():
            reasons = []
            c_score = clip_scores.get(img_id, 0.0)
            a_score = chunk_confidence_map.get(img_id, 0.0)
            
            if img_id in clip_scores:
                reasons.append("Path 1: CLIP Semantic Match")
            if img_id in associated_image_ids:
                reasons.append("Path 2: Associated with retrieved text")

            # Diagram Type Weighting
            diagram_weight = self._get_diagram_weight(img_meta)

            # Subsystem Boosting (Check if image subsystem/department matches any highly retrieved text)
            subsystem_boost = self._calculate_subsystem_boost(query, img_meta, text_results)
            if subsystem_boost > 0:
                reasons.append("Subsystem Match Boost")

            # Page Proximity Boosting
            proximity_boost = self._calculate_proximity_boost(img_meta, text_results)
            if proximity_boost > 0:
                reasons.append(f"Page Proximity Boost (+{proximity_boost})")

            # Base Formula
            # (0.45 * clip_score) + (0.30 * association) + (0.10 * subsystem) + proximity_boost + (0.05 * diagram_conf)
            raw_score = (
                (0.45 * c_score) +
                (0.30 * a_score) +
                proximity_boost + 
                (0.05 * img_meta.diagram_confidence)
            )
            
            # Subsystem boost logic
            if subsystem_boost > 0:
                raw_score += subsystem_boost

            final_score = raw_score * diagram_weight

            expl = ImageExplainability(
                retrieval_reason=reasons,
                clip_score=c_score,
                association_score=a_score,
                subsystem_match=(subsystem_boost > 0),
                source_pdf=img_meta.manual_name,
                page=img_meta.page_number,
                section_title=img_meta.section_title,
                final_score=final_score
            )

            retrieved_images.append(RetrievedImage(metadata=img_meta, explainability=expl))

        # Sort by final score descending
        retrieved_images.sort(key=lambda x: x.explainability.final_score, reverse=True)
        
        return retrieved_images[:top_k]

    def _get_diagram_weight(self, img: ImageMetadata) -> float:
        text = f"{img.caption} {' '.join(img.tags)} {img.ocr_text}".lower()
        if "piping" in text or "pipe" in text: return 1.2
        if "wiring" in text or "electrical" in text or "circuit" in text: return 1.2
        if "flow chart" in text or "flowchart" in text or "flow" in text: return 1.15
        if "layout" in text or "map" in text or "schematic" in text: return 1.15
        if "cross section" in text or "cross-section" in text: return 1.1
        if "radar" in text or "ecdis" in text or "interface" in text: return 1.1
        if "diagram" in text or "figure" in text or "fig" in text: return 1.05
        return 1.0

    def _calculate_subsystem_boost(self, query: str, img: ImageMetadata, text_results: List[RetrievalResult]) -> float:
        """Generic keyword overlap between query/text results and image metadata."""
        img_text = f"{img.caption} {' '.join(img.tags)} {img.section_title} {img.ocr_text}".lower()

        # Check against text results subsystem/department
        for res in text_results:
            if res.chunk.subsystem and res.chunk.subsystem != "general":
                if res.chunk.subsystem.lower() in img_text:
                    return 0.15
            if res.chunk.department and res.chunk.department != "general":
                if res.chunk.department.lower() in img_text:
                    return 0.1

        # Generic keyword overlap: only match highly specific/long words to avoid false positives
        query_words = [w.lower() for w in query.split() if len(w) > 5]
        stop_words = {"show", "explain", "please", "provide", "diagram", "figure", "picture"}
        
        for word in query_words:
            if word not in stop_words and word in img_text:
                return 0.1

        return 0.0

    def _calculate_proximity_boost(self, img: ImageMetadata, text_results: List[RetrievalResult]) -> float:
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
