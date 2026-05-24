import math
from typing import List, Tuple
from app.models.schemas import QueryEvalResult

def source_coverage(response_text: str, retrieved_chunks: List[str]) -> float:
    """
    Heuristic for determining what fraction of the response is backed by chunks.
    Since we don't have an LLM grader yet, we'll use a simple n-gram overlap
    or token presence as a placeholder for source coverage.
    """
    if not response_text or not retrieved_chunks:
        return 0.0
        
    response_words = set(response_text.lower().split())
    if not response_words:
        return 0.0
        
    combined_chunks = " ".join(retrieved_chunks).lower()
    chunk_words = set(combined_chunks.split())
    
    # Calculate what percentage of non-stopword-like tokens in response appear in chunks
    # This is a very rough heuristic.
    hits = sum(1 for w in response_words if w in chunk_words and len(w) > 3)
    valid_words = sum(1 for w in response_words if len(w) > 3)
    
    if valid_words == 0:
        return 0.0
        
    return min(1.0, hits / valid_words)

def _spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Spearman rank correlation."""
    n = len(x)
    if n < 2:
        return 0.0
        
    # Helper to get ranks
    def get_ranks(arr):
        sorted_indices = sorted(range(len(arr)), key=lambda k: arr[k])
        ranks = [0.0] * len(arr)
        for rank, index in enumerate(sorted_indices):
            ranks[index] = rank + 1.0
        return ranks
        
    rank_x = get_ranks(x)
    rank_y = get_ranks(y)
    
    # Calculate correlation
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n
    
    numerator = sum((rank_x[i] - mean_x) * (rank_y[i] - mean_y) for i in range(n))
    denom_x = sum((rank_x[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((rank_y[i] - mean_y) ** 2 for i in range(n))
    
    if denom_x == 0 or denom_y == 0:
        return 0.0
        
    return numerator / math.sqrt(denom_x * denom_y)

def confidence_accuracy_correlation(results: List[QueryEvalResult]) -> float:
    """
    Spearman correlation between confidence scores and actual relevance (e.g., NDCG or Precision@1).
    """
    if not results or len(results) < 2:
        return 0.0
        
    confidences = []
    accuracies = []
    
    for res in results:
        # We need the max confidence score from the retrieved chunks for this query
        # Since QueryEvalResult doesn't directly store the raw chunks' confidence score,
        # we assume it's passed or stored in text_metrics temporarily.
        conf = res.text_metrics.get("max_confidence", 0.0)
        acc = res.text_metrics.get("ndcg_at_5", 0.0)
        
        confidences.append(conf)
        accuracies.append(acc)
        
    return _spearman_rank_correlation(confidences, accuracies)

def low_confidence_detection_rate(results: List[QueryEvalResult], threshold: float = 0.6) -> float:
    """
    How often low-confidence correctly predicts irrelevance.
    True Negative Rate: TN / (TN + FP)
    Where:
    Negative prediction = confidence < threshold
    Actual Negative = precision (or hit) == 0
    """
    if not results:
        return 0.0
        
    true_negatives = 0
    false_positives = 0  # low confidence but actually relevant
    
    for res in results:
        conf = res.text_metrics.get("max_confidence", 0.0)
        acc = res.text_metrics.get("precision_at_5", 0.0)
        
        if conf < threshold:
            if acc == 0.0:
                true_negatives += 1
            else:
                false_positives += 1
                
    total_negatives = true_negatives + false_positives
    if total_negatives == 0:
        return 0.0
        
    return true_negatives / total_negatives
