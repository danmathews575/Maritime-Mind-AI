import time
from typing import Any, Dict, List

from app.configs.config import settings
from app.models.schemas import RetrievalResult
from app.retrieval.query_classifier import QueryClassifier
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.reranker import RerankerService
from app.retrieval.scoring import ConfidenceScorer
from app.services.vector_store import VectorStoreService
from app.services.bm25_index import BM25IndexService
from app.services.embedding import TextEmbeddingService
from app.services.clip_embedding import ImageEmbeddingService
from app.retrieval.image_retrieval import ImageRetrievalService
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.controller")

class RetrievalController:
    """
    Top-level orchestrator for the Phase 4 Hybrid Retrieval Engine.
    Coordinates intent classification, hybrid search, reranking, and confidence scoring.
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
        
        # We can disable reranker component in confidence scoring if reranking is disabled
        rerank_weight = 0.3 if settings.RERANKING_ENABLED else 0.0
        bm25_weight = 0.3 if settings.RERANKING_ENABLED else 0.5
        vec_weight = 0.4 if settings.RERANKING_ENABLED else 0.5
        
        self.scorer = ConfidenceScorer(
            bm25_weight=bm25_weight, 
            vector_weight=vec_weight, 
            rerank_weight=rerank_weight
        )

    def retrieve(self, query: str, top_k: int = None, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Executes the full retrieval pipeline:
        1. Classify Intent
        2. Hybrid Search (Vector + BM25 + RRF)
        3. Cross-Encoder Reranking
        4. Confidence Scoring and Threshold Filtering
        """
        top_k = top_k or settings.TOP_K_RESULTS
        start_time = time.perf_counter()

        # 1. Intent Classification
        intent = self.classifier.classify(query)
        logger.info(f"Query: '{query}' classified as Intent: {intent}")

        # Intent specific modifications
        # If DIAGRAM_REQUEST, we might want to pass specific filters in the future.
        # But for Phase 4 we just do text retrieval. Image retrieval is Phase 5.

        # 2. Hybrid Search
        results = self.hybrid_search.search(query, top_k=top_k * 2, filters=filters) # Retrieve more for reranking
        
        # 3. Reranking
        if results:
            results = self.reranker.rerank(query, results, top_n=top_k * 2)

        # 4. Confidence Scoring
        if results:
            results = self.scorer.compute(results)
            # Re-sort by final confidence score
            results.sort(key=lambda r: r.scores.confidence_score, reverse=True)

            # Apply threshold and limit to top_k
            results = self.scorer.apply_threshold(results, threshold=settings.CONFIDENCE_THRESHOLD)
            results = results[:top_k]

        # 5. Multimodal Image Retrieval
        # Retrieve images dynamically linked to the context and directly from semantic search.
        retrieved_images = self.image_retrieval.search(query, text_results=results, top_k=5)
        
        # Embed the images into the top RetrievalResult if results exist, else handle it gracefully
        if results and retrieved_images:
            results[0].images = retrieved_images

        elapsed = time.perf_counter() - start_time
        max_conf = results[0].scores.confidence_score if results else 0.0
        logger.info(f"Retrieval complete in {elapsed:.3f}s. Returned {len(results)} chunks, {len(retrieved_images)} images. Max confidence: {max_conf:.2f}")

        return results
