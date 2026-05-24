import math
from typing import List

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculates Precision@K."""
    if not retrieved or k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved_k if item in relevant_set)
    return hits / len(retrieved_k)

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculates Recall@K."""
    if not relevant or k <= 0:
        return 0.0
    if not retrieved:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved_k if item in relevant_set)
    return hits / len(relevant_set)

def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Calculates Mean Reciprocal Rank (MRR)."""
    if not retrieved or not relevant:
        return 0.0
    relevant_set = set(relevant)
    for i, item in enumerate(retrieved):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def mean_average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """Calculates Average Precision (AP) for a single query."""
    if not retrieved or not relevant:
        return 0.0
    
    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0.0
    
    for i, item in enumerate(retrieved):
        if item in relevant_set:
            hits += 1
            sum_precisions += hits / (i + 1)
            
    if not relevant_set:
        return 0.0
        
    return sum_precisions / len(relevant_set)

def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain (NDCG) @ K.
    Assumes binary relevance (1 if relevant, 0 otherwise).
    """
    if not retrieved or not relevant or k <= 0:
        return 0.0
        
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    
    dcg = 0.0
    for i, item in enumerate(retrieved_k):
        if item in relevant_set:
            # relevance is 1
            dcg += 1.0 / math.log2(i + 2)  # i is 0-indexed, so rank is i+1, log2(rank + 1) -> log2(i + 2)
            
    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    ideal_hits = min(len(relevant_set), k)
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg
