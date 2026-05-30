import functools
import time
from typing import Any, Dict, List

from app.configs.config import settings
from app.models.schemas import RetrievalResult
from app.retrieval.query_classifier import QueryClassifier
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.reranker import RerankerService
from app.retrieval.scoring import ConfidenceScorer
from app.retrieval.context_expander import ContextExpander
from app.services.vector_store import VectorStoreService
from app.services.bm25_index import BM25IndexService
from app.services.embedding import TextEmbeddingService
from app.services.clip_embedding import ImageEmbeddingService
from app.retrieval.image_retrieval import ImageRetrievalService
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.controller")

class RetrievalController:
    """
    Top-level orchestrator for the Hybrid Retrieval Engine.
    Coordinates intent classification, hybrid search, reranking,
    context expansion, confidence scoring, and multimodal retrieval.

    Pipeline:
    1. Classify Intent (+ extract metadata hints)
    2. Hybrid Search (Vector + BM25 + RRF) with optional metadata filters
    3. Cross-Encoder Reranking
    4. Context Expansion (adjacent chunk retrieval for procedures)
    5. Absolute Confidence Scoring with intent-aligned boosting
    6. Threshold Filtering
    7. Multimodal Image Retrieval
    8. Image-to-Result Distribution
    """

    def __init__(self):
        self.classifier = QueryClassifier()
        
        # Instantiate base services
        self.vs = VectorStoreService()
        self.bm25 = BM25IndexService()
        self.embedder = TextEmbeddingService()

        # Load BM25 index from disk if not built
        if not self.bm25.is_built:
            try:
                self.bm25.load()
            except Exception as e:
                logger.warning(f"Could not load BM25 index: {e}")

        self.hybrid_search = HybridSearchEngine(self.vs, self.bm25, self.embedder)
        self.reranker = RerankerService()
        self.clip = ImageEmbeddingService()
        self.image_retrieval = ImageRetrievalService(self.vs, self.clip)
        self.context_expander = ContextExpander(self.vs)
        
        # Confidence scorer with absolute scoring weights
        rerank_weight = 0.40 if settings.RERANKING_ENABLED else 0.0
        bm25_weight = 0.25 if settings.RERANKING_ENABLED else 0.40
        vec_weight = 0.35 if settings.RERANKING_ENABLED else 0.60
        
        self.scorer = ConfidenceScorer(
            bm25_weight=bm25_weight, 
            vector_weight=vec_weight, 
            rerank_weight=rerank_weight
        )

    def retrieve(self, query: str, top_k: int = None, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Executes the full retrieval pipeline with hardened scoring and
        multimodal alignment.
        """
        top_k = top_k or settings.TOP_K_RESULTS
        filters_tuple = tuple(sorted(filters.items())) if filters else None
        return self._cached_retrieve(query, top_k, filters_tuple)

    def _cached_retrieve(self, query: str, top_k: int, filters_tuple: tuple) -> List[RetrievalResult]:
        start_time = time.perf_counter()
        
        # Convert tuple back to dict if present
        filters = dict(filters_tuple) if filters_tuple else None

        # 1. Intent Classification with metadata hints
        classification = self.classifier.classify(query)
        intent = classification.intent
        logger.info(f"Query: '{query}' classified as Intent: {intent.name}")

        # Build metadata filters from classification hints
        effective_filters = dict(filters) if filters else {}
        if classification.department_hint and "department" not in effective_filters:
            effective_filters["department"] = classification.department_hint

        # 2. Hybrid Search (retrieve extra for reranking and expansion)
        results = self.hybrid_search.search(
            query, top_k=top_k * 2, filters=effective_filters or None
        )

        # Fallback: if filtered search yields too few results, retry without filters
        if len(results) < 3 and effective_filters:
            logger.info(
                f"Filtered search returned only {len(results)} results. "
                f"Retrying without metadata filters."
            )
            results = self.hybrid_search.search(query, top_k=top_k * 2, filters=None)
        
        # 3. Reranking
        if results:
            results = self.reranker.rerank(query, results, top_n=top_k * 2)

        # 4. Context Expansion (adjacent chunks for procedures)
        if results:
            results = self.context_expander.expand_context(results, max_adjacent=1)

        # 5. Confidence Scoring (absolute, with intent boosting)
        if results:
            results = self.scorer.compute(results, intent=intent)
            # Re-sort by final confidence score
            results.sort(key=lambda r: r.scores.confidence_score, reverse=True)

            # 6. Apply threshold and limit to top_k
            results = self.scorer.apply_threshold(
                results, threshold=settings.CONFIDENCE_THRESHOLD
            )
            results = results[:top_k]

        # 7. Multimodal Image Retrieval
        retrieved_images = self.image_retrieval.search(
            query, text_results=results, top_k=5
        )
        
        # 8. Distribute images to their most-relevant text results
        if results and retrieved_images:
            results, unmatched = self.context_expander.distribute_images(
                results, retrieved_images
            )
            # Attach any unmatched but high-scoring images to the top result
            if unmatched and results:
                results[0].images.extend(unmatched)

        elapsed = time.perf_counter() - start_time
        total_images = sum(len(r.images) for r in results) if results else 0
        max_conf = results[0].scores.confidence_score if results else 0.0
        logger.info(
            f"Retrieval complete in {elapsed:.3f}s. "
            f"Returned {len(results)} chunks, {total_images} images. "
            f"Max confidence: {max_conf:.2f}"
        )

        return results
