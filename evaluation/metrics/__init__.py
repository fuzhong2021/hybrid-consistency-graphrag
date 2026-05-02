# evaluation/metrics/__init__.py
"""
Wissenschaftliche Metriken für Konsistenzprüfung.

Module:
- calibration: Expected Calibration Error (ECE), Brier Score
- significance: McNemar's Test, Bootstrap Confidence Intervals
- roc_analysis: ROC-AUC, Precision-Recall Curves

Wissenschaftliche Referenzen:
- Guo et al. (2017): On Calibration of Modern Neural Networks
- Naeini et al. (2015): Obtaining Well Calibrated Probabilities
- Efron & Tibshirani (1993): Bootstrap Confidence Intervals
"""

from evaluation.metrics.calibration import (
    compute_calibration_metrics,
    compute_ece,
    compute_brier_score,
    plot_calibration_curve,
)
from evaluation.metrics.significance import (
    mcnemar_test,
    bootstrap_confidence_interval,
    paired_permutation_test,
)
from evaluation.metrics.roc_analysis import (
    compute_roc_auc,
    compute_pr_auc,
    plot_roc_curve,
    plot_precision_recall_curve,
    compute_optimal_threshold,
)

__all__ = [
    # Calibration
    "compute_calibration_metrics",
    "compute_ece",
    "compute_brier_score",
    "plot_calibration_curve",
    # Significance
    "mcnemar_test",
    "bootstrap_confidence_interval",
    "paired_permutation_test",
    # ROC
    "compute_roc_auc",
    "compute_pr_auc",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "compute_optimal_threshold",
]
