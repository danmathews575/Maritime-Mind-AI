"""
Context Expander — Retrieval Hardening

Expands retrieved results by fetching adjacent chunks to preserve
procedural continuity. When a procedure is split across multiple chunks,
this ensures the full procedure reaches the LLM.

Also distributes retrieved images to their most-relevant text results
instead of attaching all images to results[0].
"""
from typing import Dict, List, Set, Tuple

from app.models.schemas import RetrievalResult, RetrievedImage, ImageMetadata
from app.services.vector_store import VectorStoreService
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.context_expander")


class ContextExpander:
    """
    Post-retrieval expansion service that:
    1. Fetches adjacent chunks for procedural results to maintain step continuity
    2. Distributes images to their most-relevant text results
    """

    def __init__(self, vector_store: VectorStoreService):
        self.vs = vector_store

    def expand_context(
        self, results: List[RetrievalResult], max_adjacent: int = 1
    ) -> List[RetrievalResult]:
        """
        For each retrieved chunk that contains a procedure, fetch adjacent
        chunks (previous/next) to ensure procedural continuity.

        Args:
            results: Scored and ranked retrieval results.
            max_adjacent: Maximum adjacent chunks to fetch per side (default: 1).

        Returns:
            Expanded list of RetrievalResult with adjacent chunks appended.
        """
        if not results:
            return results

        existing_ids: Set[str] = {r.chunk.chunk_id for r in results}
        adjacent_ids_to_fetch: Set[str] = set()

        # Identify adjacent chunks needed
        for res in results:
            chunk = res.chunk
            should_expand = (
                chunk.contains_procedure
                or chunk.contains_emergency_workflow
            )

            if not should_expand:
                continue

            # Collect adjacent IDs
            for _ in range(max_adjacent):
                if chunk.previous_chunk_id and chunk.previous_chunk_id not in existing_ids:
                    adjacent_ids_to_fetch.add(chunk.previous_chunk_id)
                if chunk.next_chunk_id and chunk.next_chunk_id not in existing_ids:
                    adjacent_ids_to_fetch.add(chunk.next_chunk_id)

        if not adjacent_ids_to_fetch:
            return results

        # Fetch adjacent chunks from vector store
        logger.info(
            f"Expanding context: fetching {len(adjacent_ids_to_fetch)} adjacent chunks "
            f"for {sum(1 for r in results if r.chunk.contains_procedure)} procedural results."
        )

        adjacent_chunks = self.vs.get_text_chunks_by_ids(list(adjacent_ids_to_fetch))

        if not adjacent_chunks:
            return results

        # Build expanded results — adjacent chunks get a reduced confidence score
        # to indicate they're contextual support, not primary matches
        from app.models.schemas import RetrievalScores

        for adj_chunk in adjacent_chunks:
            if adj_chunk.chunk_id in existing_ids:
                continue

            # Find the parent result this is adjacent to and inherit a reduced score
            parent_score = 0.0
            for res in results:
                if (
                    res.chunk.next_chunk_id == adj_chunk.chunk_id
                    or res.chunk.previous_chunk_id == adj_chunk.chunk_id
                ):
                    parent_score = res.scores.confidence_score
                    break

            adj_scores = RetrievalScores(
                bm25_score=0.0,
                vector_score=0.0,
                rerank_score=0.0,
                final_score=0.0,
                confidence_score=parent_score * 0.6,  # Reduced confidence for context
            )

            results.append(RetrievalResult(chunk=adj_chunk, scores=adj_scores))
            existing_ids.add(adj_chunk.chunk_id)

        logger.info(
            f"Context expansion complete: {len(adjacent_chunks)} adjacent chunks added."
        )

        return results

    def distribute_images(
        self,
        results: List[RetrievalResult],
        retrieved_images: List[RetrievedImage],
    ) -> Tuple[List[RetrievalResult], List[RetrievedImage]]:
        """
        Distribute retrieved images to their most-relevant text results
        instead of attaching all images to results[0].

        Alignment scoring:
        - Same manual name: +0.3
        - Same page: +0.25, ±1 page: +0.15, ±2 page: +0.05
        - Image ID in chunk's related_image_ids: +0.4
        - Subsystem keyword overlap: +0.1

        Args:
            results: Text retrieval results.
            retrieved_images: Images from the image retrieval service.

        Returns:
            Tuple of (updated results, unmatched images).
        """
        if not results or not retrieved_images:
            return results, retrieved_images if not results else []

        # Clear any existing image assignments
        for res in results:
            res.images = []

        unmatched_images: List[RetrievedImage] = []

        for img in retrieved_images:
            best_score = -1.0
            best_idx = -1

            for idx, res in enumerate(results):
                score = self._alignment_score(res, img.metadata)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_score >= 0.15 and best_idx >= 0:
                results[best_idx].images.append(img)
            else:
                unmatched_images.append(img)

        # Log distribution
        attached_count = sum(len(r.images) for r in results)
        logger.info(
            f"Image distribution: {attached_count} images attached to results, "
            f"{len(unmatched_images)} unmatched."
        )

        return results, unmatched_images

    def _alignment_score(
        self, result: RetrievalResult, img_meta: ImageMetadata
    ) -> float:
        """Compute alignment score between a text result and an image."""
        score = 0.0
        chunk = result.chunk

        # Manual name match
        if chunk.manual_name == img_meta.manual_name:
            score += 0.3

        # Page proximity
        page_diff = abs(chunk.page_number - img_meta.page_number)
        if page_diff == 0:
            score += 0.25
        elif page_diff == 1:
            score += 0.15
        elif page_diff == 2:
            score += 0.05

        # Direct association link
        if img_meta.image_id in chunk.related_image_ids:
            score += 0.4

        # Subsystem keyword overlap
        if chunk.subsystem and chunk.subsystem != "general":
            img_text = f"{img_meta.caption} {' '.join(img_meta.tags)} {img_meta.section_title}".lower()
            if chunk.subsystem.lower() in img_text:
                score += 0.1

        return score
