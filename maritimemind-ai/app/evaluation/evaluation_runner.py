import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timezone
from pydantic import TypeAdapter

from app.models.schemas import BenchmarkQuery, QueryEvalResult, EvaluationReport
from app.retrieval.controller import RetrievalController
from app.evaluation.retrieval_metrics import precision_at_k, recall_at_k, mrr, mean_average_precision, ndcg_at_k
from app.evaluation.image_retrieval_metrics import image_hit_at_k, image_precision_at_k, cross_modal_accuracy
from app.evaluation.grounding_metrics import confidence_accuracy_correlation, low_confidence_detection_rate
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.eval_runner")

class EvaluationRunner:
    def __init__(self):
        logger.info("Initializing EvaluationRunner (Loading RetrievalController)")
        self.controller = RetrievalController()
        
    def run(self, benchmark_path: str) -> EvaluationReport:
        logger.info(f"Loading benchmark queries from {benchmark_path}")
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        adapter = TypeAdapter(List[BenchmarkQuery])
        queries = adapter.validate_python(data)
        
        logger.info(f"Loaded {len(queries)} benchmark queries. Starting evaluation...")
        
        query_results = []
        for query in queries:
            result = self._evaluate_query(query)
            query_results.append(result)
            
        metrics = self._aggregate_metrics(query_results)
        
        report = EvaluationReport(
            timestamp=datetime.now(timezone.utc),
            benchmark_version="v1.0",
            total_queries=len(queries),
            metrics=metrics,
            per_query_results=query_results,
            failure_analysis=self._generate_failure_analysis(query_results)
        )
        
        logger.info("Evaluation complete.")
        return report

    def _evaluate_query(self, query: BenchmarkQuery) -> QueryEvalResult:
        logger.debug(f"Evaluating query: {query.query_id} - '{query.query_text}'")
        
        # We don't override the classifier, we just let the controller do its thing
        # In a real rigorous test, we might force the intent or check if it matches query.intent
        try:
            results = self.controller.retrieve(query.query_text, top_k=5)
            
            retrieved_chunk_ids = [res.chunk.chunk_id for res in results]
            expected_chunks = query.expected_chunk_ids
            
            max_conf = max([r.scores.confidence_score for r in results]) if results else 0.0
            
            text_metrics = {
                "precision_at_5": precision_at_k(retrieved_chunk_ids, expected_chunks, 5),
                "recall_at_5": recall_at_k(retrieved_chunk_ids, expected_chunks, 5),
                "mrr": mrr(retrieved_chunk_ids, expected_chunks),
                "map": mean_average_precision(retrieved_chunk_ids, expected_chunks),
                "ndcg_at_5": ndcg_at_k(retrieved_chunk_ids, expected_chunks, 5),
                "max_confidence": max_conf
            }
            
            image_metrics = {}
            if query.expected_image_id:
                retrieved_images = []
                if results and results[0].images:
                    retrieved_images = [img.metadata.image_id for img in results[0].images]
                    
                image_metrics = {
                    "hit_at_3": image_hit_at_k(retrieved_images, query.expected_image_id, 3),
                    "precision_at_3": image_precision_at_k(retrieved_images, [query.expected_image_id], 3)
                }
            
            return QueryEvalResult(
                query_id=query.query_id,
                query_text=query.query_text,
                intent=query.intent,
                text_metrics=text_metrics,
                image_metrics=image_metrics,
                grounding_metrics={}, # Will compute globally
                errors=[]
            )
            
        except Exception as e:
            logger.error(f"Error evaluating query {query.query_id}: {e}")
            return QueryEvalResult(
                query_id=query.query_id,
                query_text=query.query_text,
                intent=query.intent,
                text_metrics={},
                image_metrics={},
                grounding_metrics={},
                errors=[str(e)]
            )

    def _aggregate_metrics(self, results: List[QueryEvalResult]) -> Dict:
        if not results:
            return {}
            
        valid_text_results = [r for r in results if r.text_metrics]
        
        avg_precision = sum(r.text_metrics.get("precision_at_5", 0.0) for r in valid_text_results) / len(valid_text_results) if valid_text_results else 0.0
        avg_recall = sum(r.text_metrics.get("recall_at_5", 0.0) for r in valid_text_results) / len(valid_text_results) if valid_text_results else 0.0
        avg_mrr = sum(r.text_metrics.get("mrr", 0.0) for r in valid_text_results) / len(valid_text_results) if valid_text_results else 0.0
        avg_map = sum(r.text_metrics.get("map", 0.0) for r in valid_text_results) / len(valid_text_results) if valid_text_results else 0.0
        avg_ndcg = sum(r.text_metrics.get("ndcg_at_5", 0.0) for r in valid_text_results) / len(valid_text_results) if valid_text_results else 0.0
        
        # Image metrics
        image_queries = [r for r in results if r.image_metrics]
        image_hits = [r.image_metrics.get("hit_at_3", False) for r in image_queries]
        cma = cross_modal_accuracy(image_hits)
        
        avg_image_precision = sum(r.image_metrics.get("precision_at_3", 0.0) for r in image_queries) / len(image_queries) if image_queries else 0.0
        
        # Grounding metrics globally
        conf_corr = confidence_accuracy_correlation(results)
        low_conf_det = low_confidence_detection_rate(results, threshold=0.6)
        
        return {
            "text_retrieval": {
                "precision_at_5": avg_precision,
                "recall_at_5": avg_recall,
                "mrr": avg_mrr,
                "map": avg_map,
                "ndcg_at_5": avg_ndcg
            },
            "image_retrieval": {
                "hit_at_3": sum(1 for h in image_hits if h) / len(image_hits) if image_hits else 0.0,
                "precision_at_3": avg_image_precision,
                "cross_modal_accuracy": cma
            },
            "grounding": {
                "confidence_correlation": conf_corr,
                "low_confidence_detection_rate": low_conf_det
            }
        }

    def _generate_failure_analysis(self, results: List[QueryEvalResult]) -> List[dict]:
        failures = []
        for r in results:
            if r.errors:
                failures.append({"query_id": r.query_id, "reason": "Execution Error", "details": r.errors})
            elif r.text_metrics.get("recall_at_5", 0.0) == 0.0:
                failures.append({"query_id": r.query_id, "reason": "Recall=0", "details": "No relevant chunks retrieved."})
        return failures

    def save_report(self, report: EvaluationReport, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report.model_dump_json(indent=4))
        logger.info(f"Report saved to {path}")
