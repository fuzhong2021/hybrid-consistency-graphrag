# src/extraction/__init__.py
"""
Triple-Extraktion Modul.

Implementiert LLM-basierte Wissensextraktion basierend auf:
- Microsoft GraphRAG (2024): Hierarchische Extraktion
- iText2KG (Lairgi et al., 2024): Iterative Verfeinerung
- Graphiti (Rasmussen et al., 2025): Bi-temporale Awareness
"""

from src.extraction.triple_extractor import (
    TripleExtractor,
    ExtractionConfig,
    ExtractionResult,
)
from src.extraction.chunking import (
    DocumentChunker,
    ChunkingStrategy,
    TextChunk,
)

__all__ = [
    "TripleExtractor",
    "ExtractionConfig",
    "ExtractionResult",
    "DocumentChunker",
    "ChunkingStrategy",
    "TextChunk",
]
