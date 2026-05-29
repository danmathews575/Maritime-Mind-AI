import math
from typing import List, Optional
from app.models.schemas import RetrievalResult, QueryIntent
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.scoring")


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class ConfidenceScorer:
    """
    Computes a normalized confidence score [0, 1] for retrieved results using
    ABSOLUTE scoring signals rather than batch-relative normalization.

    Design rationale:
    - Previous min-max normalization always produced a top score of 1.0 regardless
      of absolute relevance quality, making CONFIDENCE_THRESHOLD meaningless.
    - This version uses raw cosine similarity (1 - distance), sigmoid-transformed
      cross-encoder logits, and normalized BM25 scores to produce meaningful
      absolute confidence values.
    - Intent-aligned metadata boosting rewards chunks whose flags match the
      classified query intent.
    """

    def __init__(
        self,
        bm25_weight: float = 0.25,
        vector_weight: float = 0.35,
        rerank_weight: float = 0.40,
    ):
        self.w_bm25 = bm25_weight
        self.w_vec = vector_weight
        self.w_rerank = rerank_weight

    def _normalize_bm25(self, scores: List[float]) -> List[float]:
        """Min-max normalize BM25 scores (these are inherently relative)."""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s - min_s == 0:
            # All same score — if positive, give moderate confidence; if zero, give 0
            return [0.5 if s > 0 else 0.0 for s in scores]
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _abs_vector_similarity(self, distance: float) -> float:
        """
        Convert Chroma cosine distance to absolute similarity.
        Chroma cosine distance: 0 = identical, 2 = opposite.
        Similarity: 1.0 = identical, 0.0 = orthogonal, negative = opposite.
        We clamp to [0, 1].
        """
        return max(0.0, min(1.0, 1.0 - distance))

    def _abs_rerank_score(self, logit: float) -> float:
        """
        Convert cross-encoder logit to [0, 1] via sigmoid.
        ms-marco cross-encoders output raw logits where:
        - Positive = relevant
        - Negative = irrelevant
        Sigmoid naturally maps this to a probability-like score.
        """
        return _sigmoid(logit)

    def compute(
        self,
        results: List[RetrievalResult],
        intent: Optional[QueryIntent] = None,
    ) -> List[RetrievalResult]:
        """
        Computes and updates the confidence_score field for all results in place.
        Uses absolute scoring signals rather than batch-relative normalization.

        Args:
            results: List of RetrievalResult with raw scores populated.
            intent: Optional query intent for metadata-aligned boosting.
        """
        if not results:
            return []

        # Normalize BM25 scores (relative is acceptable here since BM25 scores
        # are corpus-dependent and have no universal absolute scale)
        bm25_raw = [r.scores.bm25_score for r in results]
        norm_bm25 = self._normalize_bm25(bm25_raw)

        for i, res in enumerate(results):
            # Absolute vector similarity from Chroma cosine distance
            abs_vec = self._abs_vector_similarity(res.scores.vector_score)

            # Absolute rerank score via sigmoid
            abs_rerank = self._abs_rerank_score(res.scores.rerank_score)

            # Weighted combination
            conf = (
                self.w_bm25 * norm_bm25[i]
                + self.w_vec * abs_vec
                + self.w_rerank * abs_rerank
            )

            # Intent-aligned metadata boosting
            if intent is not None:
                conf += self._intent_boost(res, intent)

            # Importance boost
            if res.chunk.importance == "high":
                conf += 0.05
            elif res.chunk.importance == "low":
                conf -= 0.02

            # Clamp to [0, 1]
            res.scores.confidence_score = max(0.0, min(1.0, float(conf)))

        return results

    def _intent_boost(self, result: RetrievalResult, intent: QueryIntent) -> float:
        """
        Boost confidence for chunks whose metadata flags align with the query intent.
        This rewards retrieval of the RIGHT type of content for the question asked.
        """
        chunk = result.chunk
        boost = 0.0

        if intent == QueryIntent.PROCEDURE:
            if chunk.contains_procedure:
                boost += 0.12
        elif intent == QueryIntent.EMERGENCY:
            if chunk.contains_emergency_workflow:
                boost += 0.18
            if chunk.contains_warning:
                boost += 0.05
        elif intent == QueryIntent.TROUBLESHOOTING:
            if QueryIntent.TROUBLESHOOTING in chunk.applicable_intents:
                boost += 0.10
        elif intent == QueryIntent.DIAGRAM_REQUEST:
            if chunk.contains_diagram_reference:
                boost += 0.08
        elif intent == QueryIntent.SOP_LOOKUP:
            if chunk.contains_procedure:
                boost += 0.08

        return boost

    def apply_threshold(
        self, results: List[RetrievalResult], threshold: float
    ) -> List[RetrievalResult]:
        """
        Filters results below the specified confidence threshold.
        Also applies a hard floor based on raw vector similarity to catch
        genuinely irrelevant results that might pass the relative threshold.
        """
        HARD_FLOOR_VECTOR_DISTANCE = 1.5  # cosine distance > 1.5 = nearly orthogonal

        filtered = []
        for r in results:
            # Hard floor: reject if raw vector distance is extremely high
            if r.scores.vector_score > HARD_FLOOR_VECTOR_DISTANCE:
                continue
            # Confidence threshold
            if r.scores.confidence_score >= threshold:
                filtered.append(r)

        removed = len(results) - len(filtered)
        if removed > 0:
            logger.debug(
                f"Thresholding removed {removed} low-confidence results "
                f"(threshold={threshold:.2f})."
            )
        return filtered
