from typing import List
from app.models.schemas import BenchmarkQuery

def image_hit_at_k(retrieved_images: List[str], expected_image: str, k: int) -> bool:
    """Returns True if the expected image is within the top K retrieved images."""
    if not expected_image or not retrieved_images or k <= 0:
        return False
    return expected_image in retrieved_images[:k]

def image_precision_at_k(retrieved_images: List[str], expected_images: List[str], k: int) -> float:
    """Calculates precision @ K for images (if there are multiple expected images)."""
    if not expected_images or not retrieved_images or k <= 0:
        return 0.0
    retrieved_k = retrieved_images[:k]
    expected_set = set(expected_images)
    hits = sum(1 for img in retrieved_k if img in expected_set)
    return hits / len(retrieved_k)

def cross_modal_accuracy(query_hits: List[bool]) -> float:
    """
    Computes overall cross-modal hit rate.
    Pass in a list of booleans representing whether each image-requesting query
    had a successful hit (e.g., hit@K).
    """
    if not query_hits:
        return 0.0
    return sum(1 for hit in query_hits if hit) / len(query_hits)
