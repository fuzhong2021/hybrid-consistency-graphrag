#!/usr/bin/env python3
# scripts/run_extrinsic_evaluation.py
"""
Extrinsische Evaluation mit Multi-Hop QA Benchmarks.

Dieses Skript:
1. Lädt HotpotQA oder MuSiQue Daten
2. Extrahiert Triples aus den supporting_facts
3. Baut einen Knowledge Graph auf
4. Validiert mit dem dreistufigen Konsistenzmodul
5. Exportiert nach Neo4j für visuelle Inspektion
6. Generiert Visualisierungen
7. Speichert alle Ergebnisse

Verwendung:
    python scripts/run_extrinsic_evaluation.py --benchmark hotpotqa --sample-size 500
    python scripts/run_extrinsic_evaluation.py --benchmark musique --sample-size 100
    python scripts/run_extrinsic_evaluation.py --visualize-only
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Projektroot zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.models.entities import Entity, EntityType, Triple, ValidationStatus
from src.graph.memory_repository import InMemoryGraphRepository
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.evaluation.benchmark_loader import BenchmarkLoader, QAExample

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtrinsicEvaluator:
    """
    Extrinsische Evaluation mit Multi-Hop QA Benchmarks.

    Evaluiert das Konsistenzmodul anhand realer QA-Daten
    und misst die Qualität des aufgebauten Knowledge Graphs.
    """

    def __init__(
        self,
        output_dir: str = "results/extrinsic",
        use_embeddings: bool = True,
        use_llm: bool = False,  # Standardmäßig ohne LLM für schnellere Tests
        skip_neo4j: bool = False
    ):
        """
        Initialisiert den Evaluator.

        Args:
            output_dir: Ausgabeverzeichnis für Ergebnisse
            use_embeddings: Embedding-basierte Validierung nutzen
            use_llm: LLM-Arbitration nutzen (kostet Geld!)
            skip_neo4j: Neo4j Export überspringen
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_embeddings = use_embeddings
        self.use_llm = use_llm
        self.skip_neo4j = skip_neo4j

        # Graph Repository
        self.graph_repo = InMemoryGraphRepository()

        # Konsistenzmodul
        self.orchestrator = self._setup_orchestrator()

        # Statistiken
        self.stats = {
            "examples_processed": 0,
            "triples_extracted": 0,
            "triples_accepted": 0,
            "triples_rejected": 0,
            "triples_review": 0,
            "entities_created": 0,
            "relations_created": 0,
            "processing_time_ms": 0.0
        }

        # Ergebnisse
        self.results = []

        logger.info(f"ExtrinsicEvaluator initialisiert")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Embeddings: {self.use_embeddings}")
        logger.info(f"  LLM: {self.use_llm}")
        logger.info(f"  Neo4j: {'disabled' if self.skip_neo4j else 'enabled'}")

    def _setup_orchestrator(self) -> ConsistencyOrchestrator:
        """Erstellt und konfiguriert den ConsistencyOrchestrator."""
        # Erweiterte Relationstypen für Multi-Hop QA
        extended_relation_types = [
            # Standard-Typen
            "WOHNT_IN", "ARBEITET_BEI", "KENNT", "BETEILIGT_AN",
            "BEFINDET_SICH_IN", "HAT_BEZIEHUNG_ZU", "TEIL_VON",
            # Zusätzliche Typen für QA-Benchmarks
            "GEBOREN_IN", "GESTORBEN_IN", "GRUENDETE", "MITGLIED_VON",
            "REGIE_BEI", "VERHEIRATET_MIT", "VERWANDT_MIT", "SPIELT_FUER",
            "ASSOZIIERT_MIT", "VERBUNDEN_MIT", "TEILNAHME_AN", "RELATED_TO"
        ]

        config = ConsistencyConfig(
            high_confidence_threshold=0.9,
            medium_confidence_threshold=0.7,
            similarity_threshold=0.85,  # Für Duplikaterkennung
            valid_relation_types=extended_relation_types
        )

        # Embedding-Modell laden wenn aktiviert
        embedding_model = None
        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding-Modell geladen: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Embedding-Modell konnte nicht geladen werden: {e}")

        # LLM-Client wenn aktiviert
        llm_client = None
        if self.use_llm:
            try:
                from openai import OpenAI
                llm_client = OpenAI()
                logger.info("OpenAI Client initialisiert")
            except Exception as e:
                logger.warning(f"OpenAI Client konnte nicht initialisiert werden: {e}")

        return ConsistencyOrchestrator(
            config=config,
            graph_repo=self.graph_repo,
            embedding_model=embedding_model,
            llm_client=llm_client,
            enable_metrics=True,
            always_check_duplicates=True
        )

    def extract_triples_from_example(self, example: QAExample) -> List[Triple]:
        """
        Extrahiert Triples aus einem QA-Beispiel.

        Nutzt die supporting_facts und context_paragraphs um
        Entitäten und Relationen zu extrahieren.

        Args:
            example: Ein QA-Beispiel aus dem Benchmark

        Returns:
            Liste von extrahierten Triples
        """
        triples = []

        # Strategie: Extrahiere Entitäten aus Titeln und verknüpfe sie
        entity_cache = {}  # title -> Entity

        # 1. Erstelle Entitäten aus Dokumenttiteln
        for para in example.context_paragraphs:
            title = para.get("title", "")
            if not title or title in entity_cache:
                continue

            # Heuristik für Entity-Typ basierend auf Namen
            entity_type = self._infer_entity_type(title)

            entity = Entity(
                name=title,
                entity_type=entity_type,
                source_document=example.id
            )
            entity_cache[title] = entity

        # 2. Erstelle Relationen zwischen supporting_facts
        sf_titles = [sf.title for sf in example.supporting_facts]

        # Verknüpfe aufeinanderfolgende supporting_facts (Bridge-Struktur)
        for i in range(len(sf_titles) - 1):
            title1 = sf_titles[i]
            title2 = sf_titles[i + 1]

            if title1 not in entity_cache or title2 not in entity_cache:
                continue

            # Inferiere Relationstyp aus dem Kontext
            relation_type = self._infer_relation_type(
                entity_cache[title1],
                entity_cache[title2],
                example
            )

            triple = Triple(
                subject=entity_cache[title1],
                predicate=relation_type,
                object=entity_cache[title2],
                source_text=example.question,
                source_document_id=example.id,
                extraction_confidence=0.8  # Heuristisch extrahiert
            )
            triples.append(triple)

        # 3. Verknüpfe alle supporting_facts mit der Antwort (falls Entität)
        answer = example.answer
        if answer and len(answer) > 2:
            answer_type = self._infer_entity_type(answer)
            answer_entity = Entity(
                name=answer,
                entity_type=answer_type,
                source_document=example.id
            )

            for sf in example.supporting_facts:
                if sf.title in entity_cache:
                    triple = Triple(
                        subject=entity_cache[sf.title],
                        predicate="RELATED_TO",
                        object=answer_entity,
                        source_text=sf.text,
                        source_document_id=example.id,
                        extraction_confidence=0.7
                    )
                    triples.append(triple)

        return triples

    def _infer_entity_type(self, name: str) -> EntityType:
        """Inferiert den Entitätstyp aus dem Namen."""
        name_lower = name.lower()

        # Orte
        location_keywords = ["city", "country", "river", "mountain", "lake",
                           "island", "state", "county", "village", "town"]
        if any(kw in name_lower for kw in location_keywords):
            return EntityType.LOCATION

        # Organisationen
        org_keywords = ["university", "company", "corporation", "inc", "ltd",
                       "organization", "institute", "foundation", "college",
                       "school", "hospital", "museum", "library", "bank"]
        if any(kw in name_lower for kw in org_keywords):
            return EntityType.ORGANIZATION

        # Ereignisse
        event_keywords = ["war", "battle", "election", "festival", "championship",
                         "world cup", "olympics", "conference", "summit"]
        if any(kw in name_lower for kw in event_keywords):
            return EntityType.EVENT

        # Standard: Person (häufigster Typ in QA-Daten)
        return EntityType.PERSON

    def _infer_relation_type(
        self,
        subject: Entity,
        obj: Entity,
        example: QAExample
    ) -> str:
        """Inferiert den Relationstyp basierend auf Kontext."""
        question_lower = example.question.lower()

        # Frage-basierte Heuristiken
        if "born" in question_lower or "birthplace" in question_lower:
            return "GEBOREN_IN"
        if "died" in question_lower or "death" in question_lower:
            return "GESTORBEN_IN"
        if "work" in question_lower or "employ" in question_lower:
            return "ARBEITET_BEI"
        if "direct" in question_lower or "film" in question_lower or "movie" in question_lower:
            return "REGIE_BEI"
        if "found" in question_lower or "establish" in question_lower:
            return "GRUENDETE"
        if "locat" in question_lower or "where" in question_lower:
            return "BEFINDET_SICH_IN"
        if "member" in question_lower or "belong" in question_lower:
            return "MITGLIED_VON"
        if "play" in question_lower or "sport" in question_lower:
            return "SPIELT_FUER"
        if "marr" in question_lower or "spouse" in question_lower:
            return "VERHEIRATET_MIT"
        if "child" in question_lower or "parent" in question_lower or "son" in question_lower or "daughter" in question_lower:
            return "VERWANDT_MIT"

        # Typ-basierte Heuristiken
        if subject.entity_type == EntityType.PERSON:
            if obj.entity_type == EntityType.ORGANIZATION:
                return "ASSOZIIERT_MIT"
            if obj.entity_type == EntityType.LOCATION:
                return "VERBUNDEN_MIT"
            if obj.entity_type == EntityType.EVENT:
                return "TEILNAHME_AN"

        return "RELATED_TO"

    def process_examples(
        self,
        examples: List[QAExample],
        progress_callback: callable = None
    ) -> None:
        """
        Verarbeitet eine Liste von QA-Beispielen.

        Args:
            examples: Liste von QA-Beispielen
            progress_callback: Optional callback für Fortschrittsanzeige
        """
        start_time = datetime.now()
        total = len(examples)

        logger.info(f"Verarbeite {total} Beispiele...")

        for i, example in enumerate(examples):
            # Fortschritt loggen
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"Fortschritt: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

            if progress_callback:
                progress_callback(i + 1, total)

            try:
                # Triples extrahieren
                triples = self.extract_triples_from_example(example)
                self.stats["triples_extracted"] += len(triples)

                example_result = {
                    "example_id": example.id,
                    "question": example.question,
                    "answer": example.answer,
                    "num_hops": example.num_hops,
                    "triples_extracted": len(triples),
                    "triples_accepted": 0,
                    "triples_rejected": 0,
                    "validation_details": []
                }

                # Validiere jedes Triple
                for triple in triples:
                    validated_triple = self.orchestrator.process(triple)

                    # Statistik aktualisieren
                    if validated_triple.validation_status == ValidationStatus.ACCEPTED:
                        self.stats["triples_accepted"] += 1
                        example_result["triples_accepted"] += 1

                        # Speichere im Graph
                        try:
                            self.graph_repo.save_triple(validated_triple)
                        except Exception as e:
                            logger.debug(f"Triple konnte nicht gespeichert werden: {e}")

                    elif validated_triple.validation_status == ValidationStatus.REJECTED:
                        self.stats["triples_rejected"] += 1
                        example_result["triples_rejected"] += 1
                    else:
                        self.stats["triples_review"] += 1

                    # Details speichern
                    example_result["validation_details"].append({
                        "subject": validated_triple.subject.name,
                        "predicate": validated_triple.predicate,
                        "object": validated_triple.object.name,
                        "status": validated_triple.validation_status.value,
                        "conflicts": len(validated_triple.conflicts)
                    })

                self.results.append(example_result)
                self.stats["examples_processed"] += 1

            except Exception as e:
                logger.error(f"Fehler bei Beispiel {example.id}: {e}")
                continue

        # Statistiken finalisieren
        duration = (datetime.now() - start_time).total_seconds() * 1000
        self.stats["processing_time_ms"] = duration

        graph_stats = self.graph_repo.get_stats()
        self.stats["entities_created"] = graph_stats.get("valid_entities", 0)
        self.stats["relations_created"] = graph_stats.get("valid_relations", 0)

        logger.info(f"Verarbeitung abgeschlossen in {duration/1000:.2f}s")

    def export_to_neo4j(self) -> Optional[Dict[str, Any]]:
        """Exportiert den Graph nach Neo4j."""
        if self.skip_neo4j:
            logger.info("Neo4j Export übersprungen (--skip-neo4j)")
            return None

        try:
            from src.graph.neo4j_exporter import create_exporter_from_config

            exporter = create_exporter_from_config()
            if exporter is None:
                logger.warning("Neo4j nicht erreichbar - Export übersprungen")
                return None

            logger.info("Exportiere Graph nach Neo4j...")
            export_stats = exporter.export_from_memory(
                self.graph_repo,
                clear_first=True
            )

            # Verifiziere
            verification = exporter.verify_export(self.graph_repo)

            return {
                "export_stats": export_stats,
                "verification": verification
            }

        except Exception as e:
            logger.error(f"Neo4j Export fehlgeschlagen: {e}")
            return None

    def generate_visualizations(self) -> Dict[str, str]:
        """Generiert alle Visualisierungen."""
        try:
            from src.visualization import generate_all_visualizations

            # Hole Metriken vom Orchestrator
            metrics = self.orchestrator.get_evaluation_report()

            files = generate_all_visualizations(
                self.graph_repo,
                metrics,
                output_dir=str(self.output_dir)
            )

            return files

        except Exception as e:
            logger.error(f"Visualisierungen konnten nicht erstellt werden: {e}")
            return {}

    def save_results(self) -> Dict[str, str]:
        """Speichert alle Ergebnisse."""
        saved_files = {}

        # 1. Evaluation Results als JSON
        results_path = self.output_dir / "evaluation_results.json"
        full_results = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "orchestrator_metrics": self.orchestrator.get_evaluation_report(),
            "graph_stats": self.graph_repo.get_stats(),
            "detailed_results": self.results[:100]  # Erste 100 für Analyse
        }

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        saved_files["evaluation_results"] = str(results_path)

        # 2. Zusammenfassung als Text
        summary_path = self.output_dir / "evaluation_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_summary())
        saved_files["summary"] = str(summary_path)

        # 3. LaTeX-Tabellen
        latex_path = self.output_dir / "latex_tables.tex"
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(self.orchestrator.get_latex_table())
        saved_files["latex_tables"] = str(latex_path)

        # 4. Metriken exportieren
        metrics_path = self.output_dir / "metrics_export.json"
        self.orchestrator.export_metrics(str(metrics_path), format="json")
        saved_files["metrics_export"] = str(metrics_path)

        logger.info(f"Ergebnisse gespeichert: {len(saved_files)} Dateien")
        return saved_files

    def _generate_summary(self) -> str:
        """Generiert eine menschenlesbare Zusammenfassung."""
        stats = self.stats
        orch_stats = self.orchestrator.get_statistics()

        summary = f"""
================================================================================
EXTRINSISCHE EVALUATION - ZUSAMMENFASSUNG
================================================================================
Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VERARBEITUNG
------------
Beispiele verarbeitet: {stats['examples_processed']}
Triples extrahiert:    {stats['triples_extracted']}
Verarbeitungszeit:     {stats['processing_time_ms']/1000:.2f}s

VALIDIERUNGSERGEBNISSE
----------------------
Akzeptiert:    {stats['triples_accepted']} ({stats['triples_accepted']/max(stats['triples_extracted'],1)*100:.1f}%)
Abgelehnt:     {stats['triples_rejected']} ({stats['triples_rejected']/max(stats['triples_extracted'],1)*100:.1f}%)
Review nötig:  {stats['triples_review']} ({stats['triples_review']/max(stats['triples_extracted'],1)*100:.1f}%)

KNOWLEDGE GRAPH
---------------
Entitäten:    {stats['entities_created']}
Relationen:   {stats['relations_created']}

ORCHESTRATOR STATISTIKEN
------------------------
Acceptance Rate:   {orch_stats.get('acceptance_rate', 'N/A')}
Escalation Rate:   {orch_stats.get('escalation_rate', 'N/A')}
Stufe 1 bestanden: {orch_stats.get('stage1_passed', 'N/A')}
Stufe 2 benötigt:  {orch_stats.get('stage2_required', 'N/A')}
Stufe 3 benötigt:  {orch_stats.get('stage3_required', 'N/A')}

================================================================================
"""
        return summary

    def run(
        self,
        benchmark: str = "hotpotqa",
        sample_size: int = 500
    ) -> Dict[str, Any]:
        """
        Führt die vollständige Evaluation durch.

        Args:
            benchmark: "hotpotqa" oder "musique"
            sample_size: Anzahl der zu verarbeitenden Beispiele

        Returns:
            Dictionary mit allen Ergebnissen und Pfaden
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTE EXTRINSISCHE EVALUATION")
        logger.info(f"Benchmark: {benchmark}")
        logger.info(f"Sample Size: {sample_size}")
        logger.info(f"{'='*60}\n")

        result = {
            "benchmark": benchmark,
            "sample_size": sample_size,
            "success": False,
            "files": {},
            "neo4j_export": None,
            "statistics": None
        }

        # 1. Lade Benchmark-Daten
        logger.info("1. Lade Benchmark-Daten...")
        loader = BenchmarkLoader()

        if benchmark.lower() == "hotpotqa":
            examples = loader.load_hotpotqa(split="validation", sample_size=sample_size)
        elif benchmark.lower() == "musique":
            examples = loader.load_musique(split="validation", sample_size=sample_size)
        else:
            logger.error(f"Unbekannter Benchmark: {benchmark}")
            return result

        if not examples:
            logger.error("Keine Beispiele geladen!")
            return result

        logger.info(f"   Geladen: {len(examples)} Beispiele")

        # 2. Verarbeite Beispiele
        logger.info("\n2. Verarbeite Beispiele und baue Knowledge Graph...")
        self.process_examples(examples)

        # 3. Exportiere nach Neo4j
        logger.info("\n3. Exportiere nach Neo4j...")
        result["neo4j_export"] = self.export_to_neo4j()

        # 4. Generiere Visualisierungen
        logger.info("\n4. Generiere Visualisierungen...")
        viz_files = self.generate_visualizations()
        result["files"].update(viz_files)

        # 5. Speichere Ergebnisse
        logger.info("\n5. Speichere Ergebnisse...")
        saved_files = self.save_results()
        result["files"].update(saved_files)

        result["statistics"] = self.stats
        result["success"] = True

        # Finale Zusammenfassung
        logger.info("\n" + self._generate_summary())

        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION ABGESCHLOSSEN")
        logger.info(f"Ergebnisse in: {self.output_dir}")
        logger.info(f"{'='*60}\n")

        return result


def main():
    """Hauptfunktion mit CLI-Interface."""
    parser = argparse.ArgumentParser(
        description="Extrinsische Evaluation mit Multi-Hop QA Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python scripts/run_extrinsic_evaluation.py --benchmark hotpotqa --sample-size 500
  python scripts/run_extrinsic_evaluation.py --benchmark musique --sample-size 100
  python scripts/run_extrinsic_evaluation.py --sample-size 10 --skip-neo4j
        """
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["hotpotqa", "musique"],
        default="hotpotqa",
        help="Benchmark-Dataset (default: hotpotqa)"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Anzahl der zu verarbeitenden Beispiele (default: 500)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extrinsic",
        help="Ausgabeverzeichnis (default: results/extrinsic)"
    )

    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Neo4j Export überspringen"
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="LLM-Arbitration aktivieren (kostet Geld!)"
    )

    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Embedding-basierte Validierung deaktivieren"
    )

    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Nur Visualisierungen neu generieren (benötigt vorherige Ergebnisse)"
    )

    args = parser.parse_args()

    # Visualize-only Modus
    if args.visualize_only:
        logger.info("Visualize-only Modus: Generiere nur Visualisierungen")
        results_path = Path(args.output_dir) / "evaluation_results.json"
        if not results_path.exists():
            logger.error(f"Keine vorherigen Ergebnisse gefunden: {results_path}")
            sys.exit(1)

        with open(results_path) as f:
            results = json.load(f)

        from src.visualization import generate_all_visualizations
        # Für visualize-only brauchen wir einen leeren Repo (zeigt nur Metriken)
        repo = InMemoryGraphRepository()
        files = generate_all_visualizations(
            repo,
            results.get("orchestrator_metrics", {}),
            output_dir=args.output_dir
        )
        logger.info(f"Visualisierungen generiert: {files}")
        return

    # Vollständige Evaluation
    evaluator = ExtrinsicEvaluator(
        output_dir=args.output_dir,
        use_embeddings=not args.no_embeddings,
        use_llm=args.use_llm,
        skip_neo4j=args.skip_neo4j
    )

    result = evaluator.run(
        benchmark=args.benchmark,
        sample_size=args.sample_size
    )

    if result["success"]:
        print("\n" + "="*60)
        print("EVALUATION ERFOLGREICH")
        print("="*60)
        print(f"\nGenerierte Dateien:")
        for name, path in result["files"].items():
            print(f"  - {name}: {path}")

        if result["neo4j_export"]:
            print(f"\nNeo4j Export:")
            print(f"  - Entitäten: {result['neo4j_export']['export_stats']['entities_exported']}")
            print(f"  - Relationen: {result['neo4j_export']['export_stats']['relations_exported']}")

        print(f"\nNächste Schritte:")
        print(f"  1. Visualisierungen anschauen: {args.output_dir}/*.png")
        if not args.skip_neo4j:
            print(f"  2. Neo4j Browser öffnen: http://localhost:7474")
            print(f"     Cypher: MATCH (n) RETURN n LIMIT 100")
        print(f"  3. Ergebnisse lesen: {args.output_dir}/evaluation_summary.txt")
    else:
        print("\nEvaluation fehlgeschlagen!")
        sys.exit(1)


if __name__ == "__main__":
    main()
