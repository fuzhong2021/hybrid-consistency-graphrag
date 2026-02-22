# src/consistency/embedding/__init__.py
"""
Embedding-basierte Konsistenzprüfung für Knowledge Graphs.

Enthält:
- TransE-basierte Plausibilitätsbewertung
- Graph-Anomalie-Erkennung
"""

from src.consistency.embedding.transe_scorer import TransEScorer, TransEConfig
from src.consistency.embedding.graph_anomaly import GraphAnomalyDetector, GraphAnomalyConfig

__all__ = [
    "TransEScorer",
    "TransEConfig",
    "GraphAnomalyDetector",
    "GraphAnomalyConfig",
]
