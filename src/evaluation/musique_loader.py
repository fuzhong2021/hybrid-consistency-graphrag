#!/usr/bin/env python3
# src/evaluation/musique_loader.py
"""
MuSiQue Dataset Loader für Multi-Step Reasoning.

MuSiQue (Multi-hop Questions via Single-hop Question Composition) ist
ein anspruchsvolles Dataset für Multi-hop Reasoning:
- 2-4 hop Fragen
- Dekompositionen verfügbar
- Striktere Evaluation als HotpotQA

Wissenschaftliche Referenz:
- Trivedi et al. (2022): MuSiQue: Multihop Questions via Single-hop Question Composition
- https://github.com/StonyBrookNLP/musique

Dataset-Struktur:
- question: Die Multi-hop Frage
- answer: Die korrekte Antwort
- paragraphs: Kontext-Paragraphen
- question_decomposition: Zerlegung in Single-hop Fragen
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MuSiQueDecomposition:
    """Eine Single-hop Frage aus der Dekompositon."""
    question: str
    answer: str
    paragraph_support_idx: int


@dataclass
class MuSiQueExample:
    """Ein Beispiel aus dem MuSiQue Dataset."""
    id: str
    question: str
    answer: str
    answer_aliases: List[str] = field(default_factory=list)
    paragraphs: List[Dict[str, Any]] = field(default_factory=list)
    decomposition: List[MuSiQueDecomposition] = field(default_factory=list)
    answerable: bool = True

    @property
    def n_hops(self) -> int:
        """Anzahl der Hops (Reasoning-Schritte)."""
        return len(self.decomposition)

    @property
    def supporting_paragraphs(self) -> List[Dict]:
        """Gibt nur die unterstützenden Paragraphen zurück."""
        return [p for p in self.paragraphs if p.get("is_supporting", False)]


class MuSiQueLoader:
    """
    Lädt das MuSiQue Dataset für Multi-hop Reasoning.

    MuSiQue ist anspruchsvoller als HotpotQA:
    - Striktere Konstruktion (kein Shortcut-Reasoning möglich)
    - Explizite Dekompositionen verfügbar
    - 2-4 hop Fragen

    Verwendung:
        loader = MuSiQueLoader()
        examples = loader.load(split="dev", sample_size=100)
        for ex in examples:
            print(f"{ex.question} ({ex.n_hops} hops)")
    """

    def __init__(self, cache_dir: str = "data/musique"):
        self.cache_dir = cache_dir
        self._dataset = None

    def load(
        self,
        split: str = "validation",
        sample_size: Optional[int] = None,
        min_hops: int = 2,
        max_hops: int = 4,
        only_answerable: bool = True
    ) -> List[MuSiQueExample]:
        """
        Lädt MuSiQue Beispiele.

        Args:
            split: "train", "validation", oder "test"
            sample_size: Anzahl der Beispiele (None = alle)
            min_hops: Minimum Anzahl Hops
            max_hops: Maximum Anzahl Hops
            only_answerable: Nur beantwortbare Fragen

        Returns:
            Liste von MuSiQueExample Objekten
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets nicht installiert: pip install datasets")
            return []

        logger.info(f"Lade MuSiQue ({split})...")

        try:
            # MuSiQue von Hugging Face (bdsaglam/musique ist der korrekte ID)
            dataset = load_dataset("bdsaglam/musique", split=split, cache_dir=self.cache_dir)
        except Exception as e:
            logger.warning(f"Fehler beim Laden von MuSiQue via HuggingFace: {e}")
            # Fallback: JSONL-Dateien wenn vorhanden
            return self._load_from_jsonl(split, sample_size, min_hops, max_hops, only_answerable)

        examples = []

        for i, item in enumerate(dataset):
            if sample_size and len(examples) >= sample_size:
                break

            # Dekompositionen parsen
            decomposition = []
            for dec in item.get("question_decomposition", []):
                decomposition.append(MuSiQueDecomposition(
                    question=dec.get("question", ""),
                    answer=dec.get("answer", ""),
                    paragraph_support_idx=dec.get("paragraph_support_idx", -1)
                ))

            # Filter nach Hops
            n_hops = len(decomposition)
            if n_hops < min_hops or n_hops > max_hops:
                continue

            answerable = item.get("answerable", True)
            if only_answerable and not answerable:
                continue

            example = MuSiQueExample(
                id=str(item.get("id", i)),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                answer_aliases=item.get("answer_aliases", []),
                paragraphs=item.get("paragraphs", []),
                decomposition=decomposition,
                answerable=answerable,
            )
            examples.append(example)

        logger.info(f"  ✓ {len(examples)} Beispiele geladen")

        # Hop-Verteilung
        hop_counts = {}
        for ex in examples:
            hop_counts[ex.n_hops] = hop_counts.get(ex.n_hops, 0) + 1
        logger.info(f"  → Hops: {hop_counts}")

        return examples

    def _load_from_jsonl(
        self,
        split: str,
        sample_size: Optional[int],
        min_hops: int,
        max_hops: int,
        only_answerable: bool
    ) -> List[MuSiQueExample]:
        """Fallback: Lade aus lokalen JSONL-Dateien."""
        import json
        import os

        jsonl_path = os.path.join(self.cache_dir, f"musique_{split}.jsonl")

        if not os.path.exists(jsonl_path):
            logger.warning(f"MuSiQue-Datei nicht gefunden: {jsonl_path}")
            logger.info("Download von: https://github.com/StonyBrookNLP/musique")
            return []

        examples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and len(examples) >= sample_size:
                    break

                item = json.loads(line)

                decomposition = []
                for dec in item.get("question_decomposition", []):
                    decomposition.append(MuSiQueDecomposition(
                        question=dec.get("question", ""),
                        answer=dec.get("answer", ""),
                        paragraph_support_idx=dec.get("paragraph_support_idx", -1)
                    ))

                n_hops = len(decomposition)
                if n_hops < min_hops or n_hops > max_hops:
                    continue

                answerable = item.get("answerable", True)
                if only_answerable and not answerable:
                    continue

                example = MuSiQueExample(
                    id=str(item.get("id", i)),
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    answer_aliases=item.get("answer_aliases", []),
                    paragraphs=item.get("paragraphs", []),
                    decomposition=decomposition,
                    answerable=answerable,
                )
                examples.append(example)

        logger.info(f"  ✓ {len(examples)} Beispiele aus JSONL geladen")
        return examples

    def convert_to_qa_examples(
        self,
        examples: List[MuSiQueExample]
    ) -> List[Dict[str, Any]]:
        """
        Konvertiert MuSiQue zu QAExample-Format für einheitliche Evaluation.

        Args:
            examples: Liste von MuSiQueExample Objekten

        Returns:
            Liste von QAExample-kompatiblen Dictionaries
        """
        from src.evaluation.benchmark_loader import QAExample, SupportingFact

        qa_examples = []

        for ex in examples:
            # Supporting Facts aus Dekompositionen
            supporting_facts = []
            for dec in ex.decomposition:
                if dec.paragraph_support_idx >= 0 and dec.paragraph_support_idx < len(ex.paragraphs):
                    para = ex.paragraphs[dec.paragraph_support_idx]
                    supporting_facts.append(SupportingFact(
                        title=para.get("title", ""),
                        sentence_idx=0
                    ))

            # Context Paragraphs
            context_paragraphs = [
                {
                    "title": p.get("title", ""),
                    "sentences": p.get("paragraph_text", "").split(". "),
                    "is_supporting": p.get("is_supporting", False)
                }
                for p in ex.paragraphs
            ]

            qa_example = QAExample(
                id=ex.id,
                question=ex.question,
                answer=ex.answer,
                question_type="multi_hop",
                supporting_facts=supporting_facts,
                context_paragraphs=context_paragraphs
            )
            qa_examples.append(qa_example)

        return qa_examples

    def get_statistics(self, examples: List[MuSiQueExample]) -> Dict[str, Any]:
        """Gibt Statistiken über die geladenen Beispiele zurück."""
        if not examples:
            return {"total": 0}

        hop_counts = {}
        for ex in examples:
            hop_counts[ex.n_hops] = hop_counts.get(ex.n_hops, 0) + 1

        return {
            "total": len(examples),
            "answerable": sum(1 for ex in examples if ex.answerable),
            "unanswerable": sum(1 for ex in examples if not ex.answerable),
            "hop_distribution": hop_counts,
            "avg_paragraphs": sum(len(ex.paragraphs) for ex in examples) / len(examples),
            "avg_supporting": sum(len(ex.supporting_paragraphs) for ex in examples) / len(examples),
            "avg_question_length": sum(len(ex.question.split()) for ex in examples) / len(examples),
        }


# ===========================================================================
# CLI Interface
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuSiQue Dataset Loader")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=4)

    args = parser.parse_args()

    loader = MuSiQueLoader()
    examples = loader.load(
        split=args.split,
        sample_size=args.sample_size,
        min_hops=args.min_hops,
        max_hops=args.max_hops
    )

    stats = loader.get_statistics(examples)
    print("\nStatistiken:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nBeispiele:")
    for ex in examples[:5]:
        print(f"  [{ex.n_hops} hops] {ex.question[:80]}...")
        print(f"    Antwort: {ex.answer}")
