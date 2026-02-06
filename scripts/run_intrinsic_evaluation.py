#!/usr/bin/env python3
"""
Intrinsische Evaluation des Konsistenzmoduls.

Testet DIREKT ob das Konsistenzmodul Fehler erkennt durch:
1. Synthetische Fehler-Injektion in saubere Daten
2. Messung von Precision/Recall der Fehlererkennung

Fehlertypen die getestet werden:
- DUPLIKATE: Gleiche Entität mit verschiedenen Namen
- WIDERSPRÜCHE: Konfligierende Fakten
- SCHEMA-VERLETZUNGEN: Ungültige Typen/Relationen
- TEMPORALE FEHLER: Unmögliche Zeitangaben

Verwendung:
    # Schneller Test (ohne Embedding-Modell)
    python scripts/run_intrinsic_evaluation.py --mode quick

    # Vollständige Evaluation (mit Embedding-Modell für Duplikaterkennung)
    python scripts/run_intrinsic_evaluation.py --mode full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta
from enum import Enum

from src.models.entities import Entity, EntityType, Triple, ValidationStatus
from src.consistency.base import ConsistencyConfig
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.embedding_validator import EmbeddingValidator, LocalEmbeddingModel
from src.graph.memory_repository import InMemoryGraphRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Fehlertypen für Evaluation
# =============================================================================

class ErrorType(Enum):
    """Typen von Fehlern die wir injizieren und testen."""
    DUPLICATE_ENTITY = "duplicate_entity"
    CONTRADICTORY_FACT = "contradictory_fact"
    SCHEMA_VIOLATION = "schema_violation"
    TEMPORAL_ERROR = "temporal_error"
    INVALID_RELATION = "invalid_relation"


@dataclass
class TestCase:
    """Ein Testfall mit bekanntem Fehler."""
    id: str
    triple: Triple
    error_type: ErrorType
    has_error: bool  # True = sollte REJECTED werden
    description: str
    expected_stage: str = "stage1"  # Welche Stufe sollte den Fehler finden


@dataclass
class EvaluationMetrics:
    """Metriken für die intrinsische Evaluation."""
    total_cases: int = 0

    # Confusion Matrix
    true_positives: int = 0   # Fehler erkannt, war Fehler
    false_positives: int = 0  # Fehler erkannt, war KEIN Fehler
    true_negatives: int = 0   # Kein Fehler erkannt, war kein Fehler
    false_negatives: int = 0  # Kein Fehler erkannt, WAR aber Fehler

    # Pro Fehlertyp
    per_type_results: Dict[str, Dict] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 = 2 * P * R / (P + R)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / Total"""
        if self.total_cases == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total_cases


# =============================================================================
# Test-Daten Generator
# =============================================================================

class TestDataGenerator:
    """Generiert Testfälle mit kontrollierten Fehlern."""

    def __init__(self, config: ConsistencyConfig):
        self.config = config
        self.test_cases: List[TestCase] = []

    def generate_all_test_cases(self) -> List[TestCase]:
        """Generiert alle Testfälle."""
        self.test_cases = []

        # 1. Korrekte Tripel (sollten ACCEPTED werden)
        self._generate_valid_triples()

        # 2. Duplikat-Fehler
        self._generate_duplicate_errors()

        # 3. Schema-Verletzungen
        self._generate_schema_errors()

        # 4. Temporale Fehler
        self._generate_temporal_errors()

        # 5. Widersprüchliche Fakten
        self._generate_contradictory_facts()

        logger.info(f"Generiert: {len(self.test_cases)} Testfälle")
        return self.test_cases

    def _generate_valid_triples(self):
        """Generiert korrekte Tripel die akzeptiert werden sollten."""
        valid_cases = [
            ("Albert Einstein", "Person", "GEBOREN_IN", "Ulm", "Ort",
             "Einstein wurde in Ulm geboren"),
            ("Marie Curie", "Person", "ARBEITET_BEI", "Universität Paris", "Organisation",
             "Curie arbeitete an der Universität Paris"),
            ("Berlin", "Ort", "BEFINDET_SICH_IN", "Deutschland", "Ort",
             "Berlin liegt in Deutschland"),
            ("Max Planck", "Person", "ERHIELT", "Nobelpreis für Physik", "Ereignis",
             "Planck erhielt den Nobelpreis"),
        ]

        for i, (subj, subj_type, pred, obj, obj_type, desc) in enumerate(valid_cases):
            triple = Triple(
                subject=Entity(name=subj, entity_type=EntityType.from_string(subj_type)),
                predicate=pred,
                object=Entity(name=obj, entity_type=EntityType.from_string(obj_type)),
                source_text=desc
            )
            self.test_cases.append(TestCase(
                id=f"valid_{i}",
                triple=triple,
                error_type=ErrorType.SCHEMA_VIOLATION,  # Kein echter Fehler
                has_error=False,
                description=f"Valides Triple: {desc}",
                expected_stage="none"
            ))

    def _generate_duplicate_errors(self):
        """Generiert Duplikat-Szenarien."""
        # Szenario: Gleiche Person mit verschiedenen Namen
        duplicates = [
            ("Albert Einstein", "A. Einstein", "Namensvariation"),
            ("Albert Einstein", "Einstein", "Kurzform"),
            ("Marie Curie", "Maria Sklodowska-Curie", "Geburtsname"),
            ("ETH Zürich", "Eidgenössische Technische Hochschule", "Vollständiger Name"),
        ]

        for i, (name1, name2, desc) in enumerate(duplicates):
            # Erstes Triple (Original)
            triple1 = Triple(
                subject=Entity(name=name1, entity_type=EntityType.PERSON),
                predicate="GEBOREN_IN",
                object=Entity(name="Teststadt", entity_type=EntityType.LOCATION),
                source_text=f"{name1} wurde in Teststadt geboren"
            )

            # Zweites Triple (Duplikat mit anderem Namen)
            triple2 = Triple(
                subject=Entity(name=name2, entity_type=EntityType.PERSON),
                predicate="GEBOREN_IN",
                object=Entity(name="Teststadt", entity_type=EntityType.LOCATION),
                source_text=f"{name2} wurde in Teststadt geboren"
            )

            self.test_cases.append(TestCase(
                id=f"dup_{i}_original",
                triple=triple1,
                error_type=ErrorType.DUPLICATE_ENTITY,
                has_error=False,  # Original ist kein Fehler
                description=f"Original: {name1}",
                expected_stage="none"
            ))

            self.test_cases.append(TestCase(
                id=f"dup_{i}_duplicate",
                triple=triple2,
                error_type=ErrorType.DUPLICATE_ENTITY,
                has_error=True,  # SOLLTE als Duplikat erkannt werden
                description=f"Duplikat: {name2} = {name1} ({desc})",
                expected_stage="stage2"  # Embedding-Validator
            ))

    def _generate_schema_errors(self):
        """Generiert Schema-Verletzungen."""
        schema_errors = [
            # Ungültiger Entity-Typ
            (Entity(name="Test", entity_type=EntityType.UNKNOWN),
             "WOHNT_IN",
             Entity(name="Berlin", entity_type=EntityType.LOCATION),
             "Ungültiger Subject-Typ (UNKNOWN)"),

            # Ungültiger Relationstyp
            (Entity(name="Max", entity_type=EntityType.PERSON),
             "FLIEGT_NACH",  # Nicht im Schema
             Entity(name="Mars", entity_type=EntityType.LOCATION),
             "Ungültiger Relationstyp (FLIEGT_NACH)"),

            # Ungültiger Object-Typ
            (Entity(name="Anna", entity_type=EntityType.PERSON),
             "ARBEITET_BEI",
             Entity(name="XYZ", entity_type=EntityType.UNKNOWN),
             "Ungültiger Object-Typ (UNKNOWN)"),
        ]

        for i, (subject, predicate, obj, desc) in enumerate(schema_errors):
            triple = Triple(
                subject=subject,
                predicate=predicate,
                object=obj,
                source_text=f"Test: {desc}"
            )
            self.test_cases.append(TestCase(
                id=f"schema_{i}",
                triple=triple,
                error_type=ErrorType.SCHEMA_VIOLATION,
                has_error=True,
                description=desc,
                expected_stage="stage1"  # Rule-Validator
            ))

    def _generate_temporal_errors(self):
        """Generiert temporale Inkonsistenzen."""
        # Entität mit valid_from > valid_until
        subject = Entity(
            name="Temporaler Fehler Person",
            entity_type=EntityType.PERSON,
            valid_from=datetime(2020, 1, 1),
            valid_until=datetime(2019, 1, 1)  # VOR valid_from!
        )

        triple = Triple(
            subject=subject,
            predicate="WOHNT_IN",
            object=Entity(name="Berlin", entity_type=EntityType.LOCATION),
            source_text="Person mit temporalem Fehler"
        )

        self.test_cases.append(TestCase(
            id="temporal_0",
            triple=triple,
            error_type=ErrorType.TEMPORAL_ERROR,
            has_error=True,
            description="valid_from (2020) > valid_until (2019)",
            expected_stage="stage1"
        ))

    def _generate_contradictory_facts(self):
        """Generiert widersprüchliche Fakten."""
        # Person ist an zwei Orten gleichzeitig geboren
        contradictions = [
            ("Einstein", "Ulm", "München", "Zwei verschiedene Geburtsorte"),
        ]

        for i, (person, place1, place2, desc) in enumerate(contradictions):
            # Erstes Triple
            triple1 = Triple(
                subject=Entity(name=person, entity_type=EntityType.PERSON),
                predicate="GEBOREN_IN",
                object=Entity(name=place1, entity_type=EntityType.LOCATION),
                source_text=f"{person} wurde in {place1} geboren"
            )

            # Widersprüchliches Triple
            triple2 = Triple(
                subject=Entity(name=person, entity_type=EntityType.PERSON),
                predicate="GEBOREN_IN",
                object=Entity(name=place2, entity_type=EntityType.LOCATION),
                source_text=f"{person} wurde in {place2} geboren"
            )

            self.test_cases.append(TestCase(
                id=f"contra_{i}_fact1",
                triple=triple1,
                error_type=ErrorType.CONTRADICTORY_FACT,
                has_error=False,
                description=f"Erster Fakt: {person} geboren in {place1}",
                expected_stage="none"
            ))

            self.test_cases.append(TestCase(
                id=f"contra_{i}_fact2",
                triple=triple2,
                error_type=ErrorType.CONTRADICTORY_FACT,
                has_error=True,  # SOLLTE als Widerspruch erkannt werden
                description=f"Widerspruch: {person} auch geboren in {place2}",
                expected_stage="stage2"  # oder stage3
            ))


# =============================================================================
# Intrinsische Evaluation
# =============================================================================

class IntrinsicEvaluator:
    """
    Führt die intrinsische Evaluation durch.

    Wichtig: Testfälle werden in der richtigen Reihenfolge verarbeitet,
    sodass Duplikate und Widersprüche erkannt werden können (Originals vor Duplikaten).
    """

    def __init__(
        self,
        orchestrator: ConsistencyOrchestrator,
        graph_repo: Optional[InMemoryGraphRepository] = None
    ):
        self.orchestrator = orchestrator
        self.graph_repo = graph_repo
        self.metrics = EvaluationMetrics()
        self.detailed_results: List[Dict] = []

    def evaluate(self, test_cases: List[TestCase]) -> EvaluationMetrics:
        """
        Evaluiert alle Testfälle.

        Die Testfälle werden in der Reihenfolge verarbeitet, in der sie generiert wurden.
        Das bedeutet:
        1. Originals werden zuerst verarbeitet und in den Graph gespeichert
        2. Duplikate werden danach geprüft (und sollten erkannt werden)
        3. Widersprüchliche Fakten werden nach dem Original geprüft
        """
        logger.info(f"Starte Evaluation von {len(test_cases)} Testfällen...")

        self.metrics = EvaluationMetrics()
        self.metrics.total_cases = len(test_cases)

        # Pro Fehlertyp initialisieren
        for error_type in ErrorType:
            self.metrics.per_type_results[error_type.value] = {
                "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0
            }

        for test_case in test_cases:
            result = self._evaluate_single(test_case)
            self.detailed_results.append(result)

        return self.metrics

    def _evaluate_single(self, test_case: TestCase) -> Dict:
        """
        Evaluiert einen einzelnen Testfall.

        Bei Testfällen ohne Fehler (has_error=False) wird das Triple
        nach der Validierung in den Graph gespeichert, damit nachfolgende
        Testfälle (Duplikate, Widersprüche) erkannt werden können.
        """
        # Triple durch Konsistenzmodul schicken
        validated = self.orchestrator.process(test_case.triple)

        # War es ein Fehler?
        detected_error = validated.validation_status in [
            ValidationStatus.REJECTED,
            ValidationStatus.NEEDS_REVIEW,
            ValidationStatus.CONFLICTING
        ]

        # Ground Truth
        actual_error = test_case.has_error

        # Wenn das Triple akzeptiert wurde UND es kein echter Fehler ist,
        # speichere es im Graph für nachfolgende Prüfungen
        # WICHTIG: Verwende 'is not None' statt nur 'self.graph_repo', da
        # InMemoryGraphRepository.__len__ 0 zurückgibt wenn leer, was bool() zu False macht
        if not detected_error and not actual_error and self.graph_repo is not None:
            try:
                self.graph_repo.save_triple(test_case.triple)
            except Exception as e:
                logger.debug(f"Fehler beim Speichern im Graph: {e}")

        # Confusion Matrix aktualisieren
        if detected_error and actual_error:
            self.metrics.true_positives += 1
            outcome = "TP"
        elif detected_error and not actual_error:
            self.metrics.false_positives += 1
            outcome = "FP"
        elif not detected_error and not actual_error:
            self.metrics.true_negatives += 1
            outcome = "TN"
        else:  # not detected_error and actual_error
            self.metrics.false_negatives += 1
            outcome = "FN"

        # Pro Fehlertyp
        type_key = test_case.error_type.value
        self.metrics.per_type_results[type_key]["total"] += 1
        self.metrics.per_type_results[type_key][outcome.lower()] += 1

        # Konflikt-Details sammeln
        conflict_details = []
        if validated.conflicts:
            for conflict in validated.conflicts:
                conflict_details.append({
                    "type": conflict.conflict_type.value if hasattr(conflict, 'conflict_type') else str(type(conflict)),
                    "description": conflict.description if hasattr(conflict, 'description') else str(conflict),
                })

        return {
            "id": test_case.id,
            "description": test_case.description,
            "error_type": test_case.error_type.value,
            "has_error": actual_error,
            "detected_error": detected_error,
            "outcome": outcome,
            "validation_status": validated.validation_status.value,
            "expected_stage": test_case.expected_stage,
            "actual_stages": [h["stage"] for h in validated.validation_history],
            "conflicts_found": conflict_details,
        }

    def print_report(self):
        """Gibt einen detaillierten Bericht aus."""
        print("\n" + "="*70)
        print("INTRINSISCHE EVALUATION - ERGEBNISSE")
        print("="*70)

        print(f"\nGESAMTMETRIKEN:")
        print(f"  Total Testfälle: {self.metrics.total_cases}")
        print(f"  True Positives:  {self.metrics.true_positives}")
        print(f"  False Positives: {self.metrics.false_positives}")
        print(f"  True Negatives:  {self.metrics.true_negatives}")
        print(f"  False Negatives: {self.metrics.false_negatives}")
        print(f"\n  Precision: {self.metrics.precision:.2%}")
        print(f"  Recall:    {self.metrics.recall:.2%}")
        print(f"  F1-Score:  {self.metrics.f1_score:.2%}")
        print(f"  Accuracy:  {self.metrics.accuracy:.2%}")

        print(f"\nPRO FEHLERTYP:")
        for error_type, data in self.metrics.per_type_results.items():
            if data["total"] > 0:
                tp = data.get("tp", 0)
                fn = data.get("fn", 0)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                print(f"  {error_type}:")
                print(f"    Total: {data['total']}, TP: {tp}, FN: {fn}, Recall: {recall:.0%}")

        print(f"\nDETAILLIERTE ERGEBNISSE:")
        for result in self.detailed_results:
            status = "✓" if result["outcome"] in ["TP", "TN"] else "✗"
            print(f"  [{status}] {result['id']}: {result['outcome']} - {result['description'][:50]}")

        print("="*70)


# =============================================================================
# Main
# =============================================================================

def create_orchestrator_with_embeddings(
    config: ConsistencyConfig,
    use_embeddings: bool = True
) -> Tuple[ConsistencyOrchestrator, Optional[InMemoryGraphRepository], Optional[LocalEmbeddingModel]]:
    """
    Erstellt Orchestrator mit optionalem Embedding-Modell und Graph-Repository.

    Args:
        config: Konsistenz-Konfiguration
        use_embeddings: Wenn True, wird SentenceTransformers geladen

    Returns:
        Tuple von (orchestrator, graph_repo, embedding_model)
    """
    graph_repo = InMemoryGraphRepository()
    embedding_model = None

    if use_embeddings:
        try:
            logger.info("Lade Embedding-Modell für Duplikaterkennung...")
            embedding_model = LocalEmbeddingModel(model_name="all-MiniLM-L6-v2")
            # Teste ob das Modell funktioniert
            _ = embedding_model.embed_query("Test")
            logger.info("Embedding-Modell erfolgreich geladen!")
        except ImportError as e:
            logger.warning(f"Embedding-Modell konnte nicht geladen werden: {e}")
            logger.warning("Installiere mit: pip install sentence-transformers")
            embedding_model = None
        except Exception as e:
            logger.warning(f"Fehler beim Laden des Embedding-Modells: {e}")
            embedding_model = None

    # Orchestrator erstellen
    orchestrator = ConsistencyOrchestrator(
        config=config,
        embedding_model=embedding_model,
        graph_repo=graph_repo,
        enable_metrics=True,
        always_check_duplicates=True  # Immer Stufe 2 ausführen wenn möglich
    )

    return orchestrator, graph_repo, embedding_model


def main():
    parser = argparse.ArgumentParser(
        description="Intrinsische Evaluation des Konsistenzmoduls"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="quick=ohne Embedding-Modell (schnell), full=mit Embedding-Modell (genauer)"
    )
    args = parser.parse_args()

    use_embeddings = args.mode == "full"

    print("\n" + "="*70)
    print("INTRINSISCHE EVALUATION DES KONSISTENZMODULS")
    print("="*70)
    print(f"\nModus: {'VOLLSTÄNDIG (mit Embedding-Modell)' if use_embeddings else 'SCHNELL (ohne Embedding-Modell)'}")
    print("\nDiese Evaluation testet DIREKT ob das Konsistenzmodul")
    print("verschiedene Fehlertypen erkennt:\n")
    print("  - DUPLIKATE: Gleiche Entität mit verschiedenen Namen")
    print("  - SCHEMA-FEHLER: Ungültige Typen/Relationen")
    print("  - TEMPORALE FEHLER: valid_from > valid_until")
    print("  - WIDERSPRÜCHE: Konfligierende Fakten")

    if not use_embeddings:
        print("\n⚠️  HINWEIS: Im Quick-Modus werden Duplikate und Widersprüche")
        print("   nicht erkannt, da kein Embedding-Modell geladen wird.")
        print("   Für vollständige Evaluation: python scripts/run_intrinsic_evaluation.py --mode full")

    # Konfiguration
    config = ConsistencyConfig(
        valid_entity_types=["Person", "Organisation", "Ort", "Ereignis", "Konzept"],
        valid_relation_types=[
            "GEBOREN_IN", "WOHNT_IN", "ARBEITET_BEI", "STUDIERT_AN",
            "BEFINDET_SICH_IN", "ERHIELT", "KENNT"
        ],
        cardinality_rules={
            "GEBOREN_IN": {"max": 1},  # Person kann nur an einem Ort geboren sein
        },
        high_confidence_threshold=0.9,
        medium_confidence_threshold=0.7,
        similarity_threshold=0.75  # Etwas niedriger für bessere Duplikaterkennung
    )

    # Orchestrator mit Embeddings erstellen
    orchestrator, graph_repo, embedding_model = create_orchestrator_with_embeddings(
        config, use_embeddings=use_embeddings
    )

    # Testdaten generieren
    generator = TestDataGenerator(config)
    test_cases = generator.generate_all_test_cases()

    # Evaluation durchführen
    evaluator = IntrinsicEvaluator(orchestrator, graph_repo)
    metrics = evaluator.evaluate(test_cases)

    # Bericht ausgeben
    evaluator.print_report()

    # Konsistenzmodul-Metriken
    print("\nKONSISTENZMODUL-STATISTIKEN:")
    stats = orchestrator.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Graph-Statistiken
    if graph_repo:
        print("\nGRAPH-STATISTIKEN:")
        graph_stats = graph_repo.get_stats()
        for k, v in graph_stats.items():
            print(f"  {k}: {v}")

    # Ergebnisse speichern
    output = {
        "mode": args.mode,
        "embeddings_used": embedding_model is not None,
        "metrics": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "accuracy": metrics.accuracy,
            "confusion_matrix": {
                "tp": metrics.true_positives,
                "fp": metrics.false_positives,
                "tn": metrics.true_negatives,
                "fn": metrics.false_negatives,
            }
        },
        "per_type": metrics.per_type_results,
        "detailed_results": evaluator.detailed_results,
    }

    with open("results/intrinsic_evaluation.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nErgebnisse gespeichert: results/intrinsic_evaluation.json")

    return metrics


if __name__ == "__main__":
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    main()
