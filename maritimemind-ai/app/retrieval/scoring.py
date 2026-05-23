from typing import List
from app.models.schemas import RetrievalResult
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.scoring")

class ConfidenceScorer:
    """
    Computes a normalized confidence score [0, 1] for retrieved results based on
    a weighted combination of component scores.
    """

    def __init__(self, bm25_weight: float = 0.3, vector_weight: float = 0.4, rerank_weight: float = 0.3):
        self.w_bm25 = bm25_weight
        self.w_vec = vector_weight
        self.w_rerank = rerank_weight

    def _normalize(self, scores: List[float]) -> List[float]:
        """Min-max normalize a list of scores to [0, 1]."""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s - min_s == 0:
            return [1.0 if s > 0 else 0.0 for s in scores]
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def compute(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Computes and updates the confidence_score field for all results in place.
        """
        if not results:
            return []

        bm25_scores = [r.scores.bm25_score for r in results]
        vector_scores = [-r.scores.vector_score for r in results] # Negative because distance is smaller = better
        rerank_scores = [r.scores.rerank_score for r in results]

        norm_bm25 = self._normalize(bm25_scores)
        norm_vec = self._normalize(vector_scores)
        norm_rerank = self._normalize(rerank_scores)

        for i, res in enumerate(results):
            conf = (
                self.w_bm25 * norm_bm25[i] +
                self.w_vec * norm_vec[i] +
                self.w_rerank * norm_rerank[i]
            )
            res.scores.confidence_score = float(conf)

        return results

    def apply_threshold(self, results: List[RetrievalResult], threshold: float) -> List[RetrievalResult]:
        """Filters results below the specified confidence threshold."""
        filtered = [r for r in results if r.scores.confidence_score >= threshold]
        logger.debug(f"Thresholding removed {len(results) - len(filtered)} low-confidence results.")
        return filtered
