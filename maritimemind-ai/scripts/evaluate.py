import os
import json
import argparse
from pydantic import TypeAdapter
from app.configs.config import settings
from app.utils.logger import setup_logger
from app.evaluation.evaluation_runner import EvaluationRunner
from app.evaluation.regression_checker import RegressionChecker
from app.models.schemas import EvaluationReport

logger = setup_logger("maritimemind.evaluate")

def parse_args():
    parser = argparse.ArgumentParser(description="MaritimeMind AI - Retrieval Evaluation System")
    parser.add_argument("--benchmark", type=str, default="app/evaluation/benchmark_queries.json",
                        help="Path to the benchmark queries JSON file")
    parser.add_argument("--output", type=str, default="evaluation_reports/latest_report.json",
                        help="Path to save the evaluation report")
    parser.add_argument("--baseline", type=str, default="app/evaluation/baselines/baseline_v1.json",
                        help="Path to the baseline report for regression checking")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Fractional drop tolerance for regression checking (e.g., 0.05 for 5%)")
    return parser.parse_args()

def run_evaluation(benchmark: str, output_path: str, baseline_path: str, tolerance: float):
    logger.info(f"Starting evaluation using benchmark: {benchmark}")
    
    runner = EvaluationRunner()
    report = runner.run(benchmark)
    
    # Check regressions if baseline exists
    if os.path.exists(baseline_path):
        logger.info(f"Checking for regressions against baseline: {baseline_path}")
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
            baseline_report = TypeAdapter(EvaluationReport).validate_python(baseline_data)
            
        checker = RegressionChecker(tolerance=tolerance)
        checker.check(report, baseline_report)
    else:
        logger.info(f"Baseline not found at {baseline_path}. Skipping regression check.")
        
    runner.save_report(report, output_path)
    logger.info(f"Evaluation pipeline completed. Report saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.benchmark, args.output, args.baseline, args.tolerance)
