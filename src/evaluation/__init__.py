# src/evaluation/__init__.py
"""
Evaluation-Paket für QA-Benchmark Integration.

Ermöglicht extrinsische Evaluation des Konsistenzmoduls
auf Standard-Benchmarks wie HotpotQA und MuSiQue.
"""

from src.evaluation.benchmark_loader import (
    BenchmarkLoader,
    QAExample,
    BenchmarkType,
)
from src.evaluation.qa_evaluator import (
    QAEvaluator,
    EvaluationResult,
    ExampleResult,
)
from src.evaluation.comparison import (
    ConsistencyAblationStudy,
    AblationResult,
    AblationConfig,
)

__all__ = [
    # Benchmark Loader
    "BenchmarkLoader",
    "QAExample",
    "BenchmarkType",
    # QA Evaluator
    "QAEvaluator",
    "EvaluationResult",
    "ExampleResult",
    # Comparison / Ablation
    "ConsistencyAblationStudy",
    "AblationResult",
    "AblationConfig",
]
