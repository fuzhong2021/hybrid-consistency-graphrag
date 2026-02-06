# src/evaluation/benchmark_loader.py
"""
Benchmark-Loader für QA-Datasets.

Unterstützt:
- HotpotQA: Multi-Hop Reasoning Benchmark
- MuSiQue: Multi-Step Question Answering

Verwendet HuggingFace datasets für einfachen Zugriff.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator
from enum import Enum

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Unterstützte Benchmark-Typen."""
    HOTPOTQA = "hotpotqa"
    MUSIQUE = "musique"


class QuestionType(Enum):
    """Fragetypen in Multi-Hop QA."""
    BRIDGE = "bridge"           # Brückenfragen (A->B->C)
    COMPARISON = "comparison"   # Vergleichsfragen
    INFERENCE = "inference"     # Inferenzfragen
    COMPOSITIONAL = "compositional"  # Zusammengesetzte Fragen
    UNKNOWN = "unknown"


@dataclass
class SupportingFact:
    """Ein unterstützender Fakt für die Antwort."""
    title: str          # Dokumenttitel
    sentence_idx: int   # Index des Satzes im Dokument
    text: str = ""      # Der tatsächliche Text


@dataclass
class QAExample:
    """
    Ein QA-Beispiel aus einem Benchmark.

    Enthält alle Informationen für Multi-Hop Reasoning Evaluation.
    """
    id: str
    question: str
    answer: str
    question_type: QuestionType = QuestionType.UNKNOWN
    level: str = ""  # "easy", "medium", "hard"

    # Supporting Facts für Multi-Hop Reasoning
    supporting_facts: List[SupportingFact] = field(default_factory=list)

    # Kontext-Paragraphen (für Graph-Aufbau)
    context_paragraphs: List[Dict[str, Any]] = field(default_factory=list)

    # Metadaten
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_hops(self) -> int:
        """Geschätzte Anzahl der Reasoning-Hops."""
        return len(self.supporting_facts)

    @property
    def context_text(self) -> str:
        """Kombinierter Kontext-Text."""
        texts = []
        for para in self.context_paragraphs:
            title = para.get("title", "")
            sentences = para.get("sentences", [])
            if sentences:
                texts.append(f"[{title}] {' '.join(sentences)}")
        return "\n\n".join(texts)

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type.value,
            "level": self.level,
            "num_hops": self.num_hops,
            "supporting_facts": [
                {"title": sf.title, "sentence_idx": sf.sentence_idx, "text": sf.text}
                for sf in self.supporting_facts
            ],
            "metadata": self.metadata,
        }


class BenchmarkLoader:
    """
    Lädt QA-Benchmarks für extrinsische Evaluation.

    Unterstützt HotpotQA und MuSiQue mit einheitlichem Interface.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Verzeichnis für Dataset-Cache
        """
        self.cache_dir = cache_dir
        self._datasets_available = self._check_datasets_library()

    def _check_datasets_library(self) -> bool:
        """Prüft ob HuggingFace datasets verfügbar ist."""
        try:
            import datasets
            return True
        except ImportError:
            logger.warning(
                "HuggingFace 'datasets' nicht installiert. "
                "Installieren mit: pip install datasets"
            )
            return False

    def load_hotpotqa(
        self,
        split: str = "validation",
        sample_size: Optional[int] = None,
        difficulty: Optional[str] = None
    ) -> List[QAExample]:
        """
        Lädt HotpotQA Dataset.

        Args:
            split: "train", "validation", oder "test"
            sample_size: Optionale Begrenzung der Beispiele
            difficulty: "easy", "medium", "hard" oder None für alle

        Returns:
            Liste von QAExample-Objekten
        """
        if not self._datasets_available:
            logger.error("datasets-Bibliothek nicht verfügbar")
            return []

        try:
            from datasets import load_dataset

            logger.info(f"Lade HotpotQA ({split})...")

            # HotpotQA laden (distractor setting)
            dataset = load_dataset(
                "hotpot_qa",
                "distractor",
                split=split,
                cache_dir=self.cache_dir
            )

            examples = []
            for item in dataset:
                # Difficulty filter
                item_level = item.get("level", "")
                if difficulty and item_level != difficulty:
                    continue

                # Question Type bestimmen
                q_type = self._classify_question_type(
                    item.get("question", ""),
                    item.get("type", "")
                )

                # Supporting Facts parsen
                sf_titles = item.get("supporting_facts", {}).get("title", [])
                sf_indices = item.get("supporting_facts", {}).get("sent_id", [])

                supporting_facts = []
                for title, sent_idx in zip(sf_titles, sf_indices):
                    # Text aus Kontext extrahieren
                    text = self._extract_supporting_text(
                        item.get("context", {}),
                        title,
                        sent_idx
                    )
                    supporting_facts.append(SupportingFact(
                        title=title,
                        sentence_idx=sent_idx,
                        text=text
                    ))

                # Kontext-Paragraphen aufbereiten
                context_titles = item.get("context", {}).get("title", [])
                context_sentences = item.get("context", {}).get("sentences", [])

                context_paragraphs = []
                for title, sentences in zip(context_titles, context_sentences):
                    context_paragraphs.append({
                        "title": title,
                        "sentences": sentences
                    })

                example = QAExample(
                    id=item.get("id", ""),
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    question_type=q_type,
                    level=item_level,
                    supporting_facts=supporting_facts,
                    context_paragraphs=context_paragraphs,
                    metadata={
                        "type": item.get("type", ""),
                        "dataset": "hotpotqa"
                    }
                )
                examples.append(example)

                if sample_size and len(examples) >= sample_size:
                    break

            logger.info(f"HotpotQA geladen: {len(examples)} Beispiele")
            return examples

        except Exception as e:
            logger.error(f"Fehler beim Laden von HotpotQA: {e}")
            return []

    def load_musique(
        self,
        split: str = "validation",
        sample_size: Optional[int] = None
    ) -> List[QAExample]:
        """
        Lädt MuSiQue Dataset.

        Args:
            split: "train", "validation", oder "test"
            sample_size: Optionale Begrenzung der Beispiele

        Returns:
            Liste von QAExample-Objekten
        """
        if not self._datasets_available:
            logger.error("datasets-Bibliothek nicht verfügbar")
            return []

        try:
            from datasets import load_dataset

            logger.info(f"Lade MuSiQue ({split})...")

            # MuSiQue laden
            dataset = load_dataset(
                "bdsaglam/musique",
                split=split,
                cache_dir=self.cache_dir
            )

            examples = []
            for item in dataset:
                # Question Type aus Decomposition ableiten
                decomposition = item.get("question_decomposition", [])
                num_hops = len(decomposition) if decomposition else 2

                q_type = QuestionType.COMPOSITIONAL if num_hops > 2 else QuestionType.BRIDGE

                # Supporting Facts aus Paragraphen extrahieren
                paragraphs = item.get("paragraphs", [])
                supporting_facts = []
                context_paragraphs = []

                for para in paragraphs:
                    is_supporting = para.get("is_supporting", False)
                    title = para.get("title", "")
                    text = para.get("paragraph_text", "")

                    context_paragraphs.append({
                        "title": title,
                        "sentences": [text],
                        "is_supporting": is_supporting
                    })

                    if is_supporting:
                        supporting_facts.append(SupportingFact(
                            title=title,
                            sentence_idx=0,
                            text=text
                        ))

                example = QAExample(
                    id=item.get("id", ""),
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    question_type=q_type,
                    level=f"{num_hops}-hop",
                    supporting_facts=supporting_facts,
                    context_paragraphs=context_paragraphs,
                    metadata={
                        "answerable": item.get("answerable", True),
                        "question_decomposition": decomposition,
                        "dataset": "musique"
                    }
                )
                examples.append(example)

                if sample_size and len(examples) >= sample_size:
                    break

            logger.info(f"MuSiQue geladen: {len(examples)} Beispiele")
            return examples

        except Exception as e:
            logger.error(f"Fehler beim Laden von MuSiQue: {e}")
            return []

    def load(
        self,
        benchmark: BenchmarkType,
        split: str = "validation",
        sample_size: Optional[int] = None,
        **kwargs
    ) -> List[QAExample]:
        """
        Universelle Lade-Methode.

        Args:
            benchmark: Benchmark-Typ
            split: Daten-Split
            sample_size: Optionale Sample-Größe
            **kwargs: Zusätzliche Parameter für spezifischen Loader

        Returns:
            Liste von QAExample-Objekten
        """
        if benchmark == BenchmarkType.HOTPOTQA:
            return self.load_hotpotqa(split, sample_size, **kwargs)
        elif benchmark == BenchmarkType.MUSIQUE:
            return self.load_musique(split, sample_size, **kwargs)
        else:
            raise ValueError(f"Unbekannter Benchmark: {benchmark}")

    def _classify_question_type(self, question: str, type_hint: str = "") -> QuestionType:
        """Klassifiziert den Fragetyp."""
        question_lower = question.lower()

        # Aus HotpotQA type-Feld
        if type_hint == "comparison":
            return QuestionType.COMPARISON
        if type_hint == "bridge":
            return QuestionType.BRIDGE

        # Heuristiken
        if any(word in question_lower for word in ["compare", "difference", "both", "either"]):
            return QuestionType.COMPARISON

        if any(word in question_lower for word in ["when", "where", "who", "what"]):
            return QuestionType.BRIDGE

        return QuestionType.UNKNOWN

    def _extract_supporting_text(
        self,
        context: Dict[str, Any],
        title: str,
        sent_idx: int
    ) -> str:
        """Extrahiert den Text eines Supporting Facts aus dem Kontext."""
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])

        for i, t in enumerate(titles):
            if t == title and i < len(sentences_list):
                sentences = sentences_list[i]
                if sent_idx < len(sentences):
                    return sentences[sent_idx]

        return ""

    def stream(
        self,
        benchmark: BenchmarkType,
        split: str = "validation",
        **kwargs
    ) -> Iterator[QAExample]:
        """
        Streamt Beispiele für speichereffiziente Verarbeitung.

        Args:
            benchmark: Benchmark-Typ
            split: Daten-Split
            **kwargs: Zusätzliche Parameter

        Yields:
            QAExample-Objekte
        """
        examples = self.load(benchmark, split, **kwargs)
        for example in examples:
            yield example

    def get_statistics(self, examples: List[QAExample]) -> Dict[str, Any]:
        """
        Berechnet Statistiken über geladene Beispiele.

        Args:
            examples: Liste von QAExample-Objekten

        Returns:
            Dictionary mit Statistiken
        """
        if not examples:
            return {"count": 0}

        question_types = {}
        levels = {}
        avg_hops = 0
        avg_context_len = 0

        for ex in examples:
            # Question Types zählen
            q_type = ex.question_type.value
            question_types[q_type] = question_types.get(q_type, 0) + 1

            # Levels zählen
            level = ex.level or "unknown"
            levels[level] = levels.get(level, 0) + 1

            avg_hops += ex.num_hops
            avg_context_len += len(ex.context_text)

        return {
            "count": len(examples),
            "question_types": question_types,
            "levels": levels,
            "avg_hops": avg_hops / len(examples),
            "avg_context_length": avg_context_len / len(examples),
        }


# Convenience-Funktion für direkten Aufruf
def load_benchmark(
    name: str,
    split: str = "validation",
    sample_size: Optional[int] = None
) -> List[QAExample]:
    """
    Convenience-Funktion zum Laden eines Benchmarks.

    Args:
        name: "hotpotqa" oder "musique"
        split: Daten-Split
        sample_size: Optionale Sample-Größe

    Returns:
        Liste von QAExample-Objekten
    """
    loader = BenchmarkLoader()

    if name.lower() == "hotpotqa":
        return loader.load_hotpotqa(split, sample_size)
    elif name.lower() == "musique":
        return loader.load_musique(split, sample_size)
    else:
        raise ValueError(f"Unbekannter Benchmark: {name}")


if __name__ == "__main__":
    # Test-Ausführung
    import sys

    logging.basicConfig(level=logging.INFO)

    loader = BenchmarkLoader()

    print("\n=== Teste BenchmarkLoader ===\n")

    # HotpotQA testen
    print("Lade HotpotQA (5 Beispiele)...")
    examples = loader.load_hotpotqa(split="validation", sample_size=5)

    if examples:
        print(f"Geladen: {len(examples)} Beispiele")
        print("\nBeispiel 1:")
        ex = examples[0]
        print(f"  Frage: {ex.question}")
        print(f"  Antwort: {ex.answer}")
        print(f"  Typ: {ex.question_type.value}")
        print(f"  Hops: {ex.num_hops}")

        stats = loader.get_statistics(examples)
        print(f"\nStatistiken: {stats}")
    else:
        print("Keine Beispiele geladen (datasets evtl. nicht installiert)")

    print("\n=== Test abgeschlossen ===")
