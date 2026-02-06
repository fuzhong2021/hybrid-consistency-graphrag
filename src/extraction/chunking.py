# src/extraction/chunking.py
"""
Document Chunking für Knowledge Extraction.

Implementiert verschiedene Chunking-Strategien:
- Sentence-based (für präzise Extraktion)
- Semantic (für Kontext-Erhaltung)
- Sliding Window (für Überlappung)

Basiert auf Best Practices aus:
- LangChain Text Splitters
- Microsoft GraphRAG Chunking
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Verfügbare Chunking-Strategien."""
    FIXED_SIZE = "fixed_size"           # Feste Zeichenanzahl
    SENTENCE = "sentence"               # Satzbasiert
    PARAGRAPH = "paragraph"             # Absatzbasiert
    SEMANTIC = "semantic"               # Semantisch (mit Embeddings)
    SLIDING_WINDOW = "sliding_window"   # Überlappende Fenster


@dataclass
class TextChunk:
    """Ein Text-Chunk mit Metadaten."""
    text: str
    chunk_id: str
    document_id: Optional[str] = None

    # Position im Originaldokument
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0

    # Für Overlap-Tracking
    overlap_with_previous: int = 0
    overlap_with_next: int = 0

    # Metadaten
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"TextChunk({self.chunk_id}: '{preview}')"


@dataclass
class ChunkingConfig:
    """Konfiguration für Chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE
    chunk_size: int = 1000          # Max. Zeichen pro Chunk
    chunk_overlap: int = 100        # Überlappung zwischen Chunks
    min_chunk_size: int = 100       # Min. Zeichen pro Chunk

    # Sentence-spezifisch
    sentences_per_chunk: int = 5    # Sätze pro Chunk (wenn strategy=SENTENCE)

    # Separator-Hierarchie (für rekursives Splitting)
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",   # Absätze
        "\n",     # Zeilenumbrüche
        ". ",     # Sätze
        "! ",
        "? ",
        "; ",
        ", ",
        " ",      # Wörter
    ])


class DocumentChunker:
    """
    Zerlegt Dokumente in Chunks für die Wissensextraktion.

    Unterstützt verschiedene Strategien je nach Anwendungsfall.
    """

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def chunk(
        self,
        text: str,
        document_id: str = None
    ) -> List[TextChunk]:
        """
        Zerlegt Text in Chunks.

        Args:
            text: Zu zerlegender Text
            document_id: Optionale Dokument-ID

        Returns:
            Liste von TextChunk-Objekten
        """
        if not text or not text.strip():
            return []

        strategy = self.config.strategy

        if strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text, document_id)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text, document_id)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text, document_id)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            return self._chunk_sliding_window(text, document_id)
        elif strategy == ChunkingStrategy.SEMANTIC:
            # Semantic Chunking benötigt Embeddings - Fallback zu Sentence
            logger.warning("Semantic Chunking benötigt Embeddings - verwende Sentence-basiert")
            return self._chunk_by_sentence(text, document_id)
        else:
            raise ValueError(f"Unbekannte Strategie: {strategy}")

    def chunk_stream(
        self,
        text: str,
        document_id: str = None
    ) -> Iterator[TextChunk]:
        """
        Streaming-Version des Chunkings.

        Nützlich für sehr große Dokumente.
        """
        for chunk in self.chunk(text, document_id):
            yield chunk

    def _chunk_fixed_size(
        self,
        text: str,
        document_id: str
    ) -> List[TextChunk]:
        """Chunking mit fester Größe und Überlappung."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Versuche an Wortgrenze zu enden
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start + self.config.min_chunk_size:
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_id=f"{document_id or 'doc'}_{chunk_idx}",
                    document_id=document_id,
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_idx,
                    overlap_with_previous=min(overlap, start) if chunk_idx > 0 else 0
                ))
                chunk_idx += 1

            # Nächster Start mit Überlappung
            start = end - overlap if end < len(text) else len(text)

        return chunks

    def _chunk_by_sentence(
        self,
        text: str,
        document_id: str
    ) -> List[TextChunk]:
        """Chunking basierend auf Sätzen."""
        # Sätze extrahieren
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_start = 0
        chunk_idx = 0
        char_pos = 0

        for sentence in sentences:
            sentence_start = text.find(sentence, char_pos)
            if sentence_start == -1:
                sentence_start = char_pos
            char_pos = sentence_start + len(sentence)

            current_sentences.append(sentence)
            current_text = ' '.join(current_sentences)

            # Chunk erstellen wenn:
            # 1. Genug Sätze ODER
            # 2. Genug Zeichen
            should_create = (
                len(current_sentences) >= self.config.sentences_per_chunk or
                len(current_text) >= self.config.chunk_size
            )

            if should_create:
                chunks.append(TextChunk(
                    text=current_text.strip(),
                    chunk_id=f"{document_id or 'doc'}_{chunk_idx}",
                    document_id=document_id,
                    start_char=current_start,
                    end_char=char_pos,
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1

                # Überlappung: Letzte Sätze behalten
                overlap_sentences = max(1, len(current_sentences) // 3)
                current_sentences = current_sentences[-overlap_sentences:]
                current_start = text.find(current_sentences[0], current_start)

        # Letzten Chunk hinzufügen
        if current_sentences:
            current_text = ' '.join(current_sentences)
            if len(current_text) >= self.config.min_chunk_size:
                chunks.append(TextChunk(
                    text=current_text.strip(),
                    chunk_id=f"{document_id or 'doc'}_{chunk_idx}",
                    document_id=document_id,
                    start_char=current_start,
                    end_char=len(text),
                    chunk_index=chunk_idx
                ))

        return chunks

    def _chunk_by_paragraph(
        self,
        text: str,
        document_id: str
    ) -> List[TextChunk]:
        """Chunking basierend auf Absätzen."""
        # Absätze durch doppelte Zeilenumbrüche trennen
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_paragraphs = []
        current_start = 0
        chunk_idx = 0
        char_pos = 0

        for para in paragraphs:
            para_start = text.find(para, char_pos)
            if para_start == -1:
                para_start = char_pos
            char_pos = para_start + len(para)

            # Prüfen ob Absatz alleine zu groß ist
            if len(para) > self.config.chunk_size:
                # Großen Absatz separat chunken
                if current_paragraphs:
                    chunks.append(TextChunk(
                        text='\n\n'.join(current_paragraphs),
                        chunk_id=f"{document_id or 'doc'}_{chunk_idx}",
                        document_id=document_id,
                        start_char=current_start,
                        end_char=para_start,
                        chunk_index=chunk_idx
                    ))
                    chunk_idx += 1
                    current_paragraphs = []

                # Großen Absatz mit Fixed-Size chunken
                sub_chunks = self._chunk_fixed_size(para, f"{document_id}_{chunk_idx}")
                for sub in sub_chunks:
                    sub.chunk_index = chunk_idx
                    sub.start_char += para_start
                    sub.end_char += para_start
                    chunks.append(sub)
                    chunk_idx += 1

                current_start = char_pos
                continue

            current_paragraphs.append(para)
            current_text = '\n\n'.join(current_paragraphs)

            if len(current_text) >= self.config.chunk_size:
                chunks.append(TextChunk(
                    text=current_text,
                    chunk_id=f"{document_id or 'doc'}_{chunk_idx}",
                    document_id=document_id,
                    start_char=current_start,
                    end_char=char_pos,
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1

                # Überlappung: Letzten Absatz behalten
                current_paragraphs = [current_paragraphs[-1]] if current_paragraphs else []
                current_start = text.find(current_paragraphs[0], current_start) if current_paragraphs else char_pos

        # Letzten Chunk
        if current_paragraphs:
            current_text = '\n\n'.join(current_paragraphs)
            if len(current_text) >= self.config.min_chunk_size:
                chunks.append(TextChunk(
                    text=current_text,
                    chunk_id=f"{document_id or 'doc'}_{chunk_idx}",
                    document_id=document_id,
                    start_char=current_start,
                    end_char=len(text),
                    chunk_index=chunk_idx
                ))

        return chunks

    def _chunk_sliding_window(
        self,
        text: str,
        document_id: str
    ) -> List[TextChunk]:
        """Chunking mit überlappenden Fenstern."""
        chunks = []
        window_size = self.config.chunk_size
        step_size = window_size - self.config.chunk_overlap

        chunk_idx = 0
        for start in range(0, len(text), step_size):
            end = min(start + window_size, len(text))

            # An Satzgrenze anpassen wenn möglich
            if end < len(text):
                # Suche Satzende im letzten Drittel
                search_start = start + (2 * window_size // 3)
                for sep in ['. ', '! ', '? ', '\n']:
                    sep_pos = text.find(sep, search_start, end)
                    if sep_pos != -1:
                        end = sep_pos + len(sep)
                        break

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_id=f"{document_id or 'doc'}_{chunk_idx}",
                    document_id=document_id,
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_idx,
                    overlap_with_previous=self.config.chunk_overlap if chunk_idx > 0 else 0
                ))
                chunk_idx += 1

            if end >= len(text):
                break

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Teilt Text in Sätze."""
        # Einfache Satz-Trennung
        sentences = self._sentence_pattern.split(text)

        # Nachbearbeitung: Leere entfernen, trimmen
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def get_statistics(self, chunks: List[TextChunk]) -> dict:
        """Berechnet Statistiken über die Chunks."""
        if not chunks:
            return {"count": 0}

        lengths = [c.length for c in chunks]

        return {
            "count": len(chunks),
            "total_chars": sum(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "strategy": self.config.strategy.value,
        }


def chunk_document(
    text: str,
    strategy: str = "sentence",
    chunk_size: int = 1000,
    overlap: int = 100
) -> List[TextChunk]:
    """
    Convenience-Funktion für Dokument-Chunking.

    Args:
        text: Zu zerlegender Text
        strategy: "fixed_size", "sentence", "paragraph", "sliding_window"
        chunk_size: Max. Zeichen pro Chunk
        overlap: Überlappung

    Returns:
        Liste von TextChunk-Objekten
    """
    strategy_enum = ChunkingStrategy(strategy)

    config = ChunkingConfig(
        strategy=strategy_enum,
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunker = DocumentChunker(config)
    return chunker.chunk(text)


if __name__ == "__main__":
    # Test
    test_text = """
    Albert Einstein wurde am 14. März 1879 in Ulm geboren. Er war ein theoretischer Physiker.
    Seine Arbeiten zur Relativitätstheorie machten ihn weltberühmt.

    Einstein arbeitete zunächst am Patentamt in Bern. Dort entwickelte er 1905 die spezielle
    Relativitätstheorie. Diese Arbeit revolutionierte unser Verständnis von Raum und Zeit.

    Später wurde er Professor in Berlin. 1921 erhielt er den Nobelpreis für Physik.
    Er emigrierte 1933 in die USA und arbeitete bis zu seinem Tod 1955 in Princeton.
    """

    print("=== Document Chunking Test ===\n")

    for strategy in [ChunkingStrategy.SENTENCE, ChunkingStrategy.PARAGRAPH]:
        config = ChunkingConfig(strategy=strategy, chunk_size=300)
        chunker = DocumentChunker(config)

        chunks = chunker.chunk(test_text, "test_doc")

        print(f"\n--- Strategie: {strategy.value} ---")
        print(f"Anzahl Chunks: {len(chunks)}")

        for chunk in chunks:
            print(f"\n{chunk.chunk_id}:")
            print(f"  Länge: {chunk.length} Zeichen")
            print(f"  Text: {chunk.text[:80]}...")

        stats = chunker.get_statistics(chunks)
        print(f"\nStatistiken: {stats}")
