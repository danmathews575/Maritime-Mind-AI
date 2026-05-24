import pytest
import os
import json
from unittest.mock import MagicMock, patch

from app.evaluation.evaluation_runner import EvaluationRunner
from app.evaluation.regression_checker import RegressionChecker
from app.models.schemas import EvaluationReport, BenchmarkQuery, QueryIntent, QueryEvalResult

@pytest.fixture
def mock_benchmark_file(tmp_path):
    data = [
        {
            "query_id": "test_01",
            "query_text": "What is the purpose?",
            "intent": "EXPLANATION",
            "expected_manual": "deck",
            "expected_page": 10,
            "expected_chunk_ids": ["chunk_1"],
            "expected_image_id": "img_1"
        }
    ]
    file_path = tmp_path / "test_benchmark.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return str(file_path)

def test_evaluation_runner_run(mock_benchmark_file):
    with patch('app.evaluation.evaluation_runner.RetrievalController') as MockController:
        mock_controller_instance = MockController.return_value
        
        # We don't even need it to return real results for the runner to finish without error,
        # but let's mock empty retrieve
        mock_controller_instance.retrieve.return_value = []
        
        runner = EvaluationRunner()
        report = runner.run(mock_benchmark_file)
        
        assert report is not None
        assert report.total_queries == 1
        assert len(report.per_query_results) == 1
        assert report.per_query_results[0].query_id == "test_01"

def test_regression_checker():
    checker = RegressionChecker(tolerance=0.05)
    
    baseline = EvaluationReport(
        metrics={
            "text_retrieval": {
                "precision_at_5": 0.8,
                "recall_at_5": 0.7
            }
        }
    )
    
    current_ok = EvaluationReport(
        metrics={
            "text_retrieval": {
                "precision_at_5": 0.78, # 2.5% drop (tolerable)
                "recall_at_5": 0.72  # improvement
            }
        }
    )
    
    current_bad = EvaluationReport(
        metrics={
            "text_retrieval": {
                "precision_at_5": 0.6, # 25% drop (bad)
                "recall_at_5": 0.7
            }
        }
    )
    
    regs_ok = checker.check(current_ok, baseline)
    assert len(regs_ok) == 0
    
    regs_bad = checker.check(current_bad, baseline)
    assert len(regs_bad) == 1
    assert regs_bad[0].metric_name == "text_retrieval.precision_at_5"
