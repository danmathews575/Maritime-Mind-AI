from typing import List, Dict
from pydantic import BaseModel

from app.models.schemas import EvaluationReport
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.regression_checker")

class Regression(BaseModel):
    metric_name: str
    baseline_value: float
    current_value: float
    drop_percentage: float

class RegressionChecker:
    def __init__(self, tolerance: float = 0.05):
        """
        tolerance: fractional drop to tolerate before flagging (e.g., 0.05 = 5%).
        """
        self.tolerance = tolerance

    def check(self, current: EvaluationReport, baseline: EvaluationReport) -> List[Regression]:
        regressions = []
        
        baseline_flat = self._flatten_metrics(baseline.metrics)
        current_flat = self._flatten_metrics(current.metrics)
        
        for metric, b_val in baseline_flat.items():
            if metric not in current_flat:
                logger.warning(f"Metric {metric} missing in current report.")
                continue
                
            c_val = current_flat[metric]
            
            # If baseline is 0, we can't really drop
            if b_val <= 0.0:
                continue
                
            drop = (b_val - c_val) / b_val
            
            if drop > self.tolerance:
                reg = Regression(
                    metric_name=metric,
                    baseline_value=b_val,
                    current_value=c_val,
                    drop_percentage=drop * 100.0
                )
                regressions.append(reg)
                logger.error(f"REGRESSION DETECTED: {metric} dropped by {drop*100:.1f}% (Baseline: {b_val:.3f}, Current: {c_val:.3f})")
                
        if not regressions:
            logger.info("No regressions detected compared to baseline.")
            
        return regressions

    def _flatten_metrics(self, metrics: Dict, prefix: str = "") -> Dict[str, float]:
        flat = {}
        for k, v in metrics.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_metrics(v, key))
            else:
                flat[key] = v
        return flat
