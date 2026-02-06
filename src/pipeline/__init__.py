# src/pipeline/__init__.py
"""
GraphRAG Pipeline Modul.

Integriert alle Komponenten:
- Triple-Extraktion
- Konsistenzprüfung
- Graph-Speicherung
- Antwortgenerierung
"""

from src.pipeline.graphrag_pipeline import (
    GraphRAGPipeline,
    PipelineConfig,
    PipelineResult,
    InMemoryGraphStore,
)

__all__ = [
    "GraphRAGPipeline",
    "PipelineConfig",
    "PipelineResult",
    "InMemoryGraphStore",
]
