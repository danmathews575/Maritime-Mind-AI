import pytest
from app.evaluation.retrieval_metrics import precision_at_k, recall_at_k, mrr, mean_average_precision, ndcg_at_k
from app.evaluation.image_retrieval_metrics import image_hit_at_k, image_precision_at_k, cross_modal_accuracy
from app.evaluation.grounding_metrics import source_coverage, confidence_accuracy_correlation, low_confidence_detection_rate
from app.models.schemas import QueryEvalResult, QueryIntent

def test_precision_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = ["B", "D", "F"]
    
    # At 1: ["A"] -> 0 hits
    assert precision_at_k(retrieved, relevant, 1) == 0.0
    # At 2: ["A", "B"] -> 1 hit
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    # At 5: ["A", "B", "C", "D", "E"] -> 2 hits
    assert precision_at_k(retrieved, relevant, 5) == 0.4
    
def test_recall_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = ["B", "D", "F"]
    
    # relevant size = 3
    # At 2: hits = 1 ("B")
    assert recall_at_k(retrieved, relevant, 2) == 1/3
    # At 5: hits = 2 ("B", "D")
    assert recall_at_k(retrieved, relevant, 5) == 2/3
    
def test_mrr():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = ["C", "E"]
    # First hit is "C" at rank 3
    assert mrr(retrieved, relevant) == 1.0 / 3.0
    
def test_ndcg_at_k():
    retrieved = ["A", "B", "C"]
    relevant = ["A", "C"]
    # ranks: A=1, B=2, C=3
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1/1 + 1/2 = 1.5
    # IDCG (best is A, C, B) = 1/log2(2) + 1/log2(3) = 1/1 + 1/1.58 = 1.6309
    
    import math
    expected_dcg = 1.0 + 1.0 / math.log2(4)
    expected_idcg = 1.0 + 1.0 / math.log2(3)
    
    assert abs(ndcg_at_k(retrieved, relevant, 3) - (expected_dcg / expected_idcg)) < 1e-4

def test_image_hit_at_k():
    assert image_hit_at_k(["img1", "img2", "img3"], "img2", 2) == True
    assert image_hit_at_k(["img1", "img2", "img3"], "img3", 2) == False
    
def test_source_coverage():
    chunks = ["The ballast system regulates stability.", "It uses water pumps."]
    response = "The ballast system uses water pumps to regulate stability."
    assert source_coverage(response, chunks) > 0.7
    
    response2 = "I do not know about the engine room."
    assert source_coverage(response2, chunks) < 0.2

def test_confidence_correlation():
    results = [
        QueryEvalResult(query_id="1", query_text="", intent=QueryIntent.EXPLANATION, text_metrics={"max_confidence": 0.9, "ndcg_at_5": 1.0}),
        QueryEvalResult(query_id="2", query_text="", intent=QueryIntent.EXPLANATION, text_metrics={"max_confidence": 0.5, "ndcg_at_5": 0.0}),
        QueryEvalResult(query_id="3", query_text="", intent=QueryIntent.EXPLANATION, text_metrics={"max_confidence": 0.7, "ndcg_at_5": 0.5}),
    ]
    corr = confidence_accuracy_correlation(results)
    assert corr > 0.9 # Should be perfectly correlated in this dummy set

def test_low_confidence_detection_rate():
    results = [
        QueryEvalResult(query_id="1", query_text="", intent=QueryIntent.EXPLANATION, text_metrics={"max_confidence": 0.9, "precision_at_5": 1.0}), # high conf, hit
        QueryEvalResult(query_id="2", query_text="", intent=QueryIntent.EXPLANATION, text_metrics={"max_confidence": 0.4, "precision_at_5": 0.0}), # low conf, miss -> TN
        QueryEvalResult(query_id="3", query_text="", intent=QueryIntent.EXPLANATION, text_metrics={"max_confidence": 0.3, "precision_at_5": 0.5}), # low conf, hit -> FP
    ]
    # TN = 1, FP = 1
    rate = low_confidence_detection_rate(results, threshold=0.6)
    assert rate == 0.5
