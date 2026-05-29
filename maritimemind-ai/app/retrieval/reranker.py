from typing import List
import time
from sentence_transformers import CrossEncoder

from app.configs.config import settings
from app.models.schemas import RetrievalResult
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.reranker")

class RerankerService:
    """
    Optional cross-encoder reranking service.
    Scores (query, document) pairs directly to produce highly accurate relevance scores.
    """

    def __init__(self):
        self._model = None
        self._enabled = settings.RERANKING_ENABLED
        self._model_name = settings.RERANKER_MODEL
        self._device = settings.DEVICE

    def _lazy_load(self):
        if self._enabled and self._model is None:
            logger.info(f"Loading cross-encoder model: {self._model_name} on {self._device}")
            self._model = CrossEncoder(self._model_name, device=self._device)

    def rerank(self, query: str, results: List[RetrievalResult], top_n: int = 20) -> List[RetrievalResult]:
        """
        Reranks a list of RetrievalResults using a CrossEncoder.
        Only the top `top_n` items are processed to save computation.
        """
        if not self._enabled:
            logger.debug("Reranking disabled, returning results unchanged.")
            return results
        if not results:
            return results

        self._lazy_load()

        # Limit to top_n to avoid excessive computation
        candidates = results[:top_n]
        remaining = results[top_n:]

        pairs = [[query, res.chunk.content] for res in candidates]
        
        # Predict scores
        start_time = time.time()
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}")
            return results
        elapsed = time.time() - start_time
        logger.info(f"Reranked {len(candidates)} candidates in {elapsed:.3f}s")

        # Update scores in the RetrievalResult objects
        for i, res in enumerate(candidates):
            res.scores.rerank_score = float(scores[i])

        # Re-sort candidates by cross-encoder score descending
        candidates.sort(key=lambda x: x.scores.rerank_score, reverse=True)

        return candidates + remaining
