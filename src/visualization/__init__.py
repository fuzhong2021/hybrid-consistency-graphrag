# src/visualization/__init__.py
"""
Visualisierungsmodul für Knowledge Graph und Evaluation.

Stellt Funktionen bereit für:
- Knowledge Graph Visualisierung (NetworkX + Matplotlib)
- Konsistenzmodul-Metriken (Balkendiagramme)
- Konfidenz-Verteilungen (Histogramme)
- Verarbeitungszeiten (Boxplots)
"""

from src.visualization.plots import (
    plot_knowledge_graph,
    plot_validation_metrics,
    plot_confidence_distribution,
    plot_processing_times,
    plot_entity_type_distribution,
    generate_all_visualizations
)

__all__ = [
    "plot_knowledge_graph",
    "plot_validation_metrics",
    "plot_confidence_distribution",
    "plot_processing_times",
    "plot_entity_type_distribution",
    "generate_all_visualizations"
]
