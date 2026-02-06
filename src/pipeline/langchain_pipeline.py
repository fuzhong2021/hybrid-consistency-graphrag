# src/pipeline/langchain_pipeline.py
"""
Deprecated: Verwende src.pipeline.graphrag_pipeline.

Dieses Modul existiert nur noch für Rückwärtskompatibilität.
Alle Funktionalität wurde in graphrag_pipeline.py konsolidiert.
"""

from src.pipeline.graphrag_pipeline import (
    GraphRAGPipeline as LangChainGraphRAGPipeline,
    PipelineConfig as LangChainPipelineConfig,
    PipelineResult as IngestionResult,
    PipelineResult as QueryResult,
    PipelineResult,
    PipelineConfig,
    GraphRAGPipeline,
    InMemoryGraphStore,
    create_pipeline,
    main,
)

__all__ = [
    "LangChainGraphRAGPipeline",
    "LangChainPipelineConfig",
    "IngestionResult",
    "QueryResult",
    "PipelineResult",
    "PipelineConfig",
    "GraphRAGPipeline",
    "InMemoryGraphStore",
    "create_pipeline",
    "main",
]

if __name__ == "__main__":
    main()
