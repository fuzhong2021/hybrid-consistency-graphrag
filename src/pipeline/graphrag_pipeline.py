# src/pipeline/graphrag_pipeline.py
"""
Vollständige GraphRAG Pipeline.

Implementiert den End-to-End Workflow:
1. Dokument-Ingestion → Triple-Extraktion
2. Konsistenzprüfung → Knowledge Graph
3. Query → Graph-Retrieval → LLM Answer Generation

Basiert auf:
- Microsoft GraphRAG Architektur
- Consistency-aware Knowledge Integration (unsere Erweiterung)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.models.entities import Entity, Triple, ValidationStatus
from src.consistency.base import ConsistencyConfig
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.metrics import ConsistencyMetrics
from src.extraction.triple_extractor import TripleExtractor, ExtractionConfig, ExtractionResult

logger = logging.getLogger(__name__)


# =============================================================================
# Konfiguration
# =============================================================================

@dataclass
class PipelineConfig:
    """Konfiguration für die GraphRAG Pipeline."""

    # Extraction
    extraction_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"

    # Consistency
    enable_consistency: bool = True
    high_confidence_threshold: float = 0.9
    medium_confidence_threshold: float = 0.7

    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # LLM
    llm_temperature: float = 0.0

    # Answer Generation
    answer_model: str = "gpt-4-turbo-preview"
    max_context_triples: int = 20
    answer_temperature: float = 0.0
    answer_max_tokens: int = 500

    # Schema
    entity_types: List[str] = field(default_factory=lambda: [
        "Person", "Organisation", "Ort", "Ereignis", "Dokument", "Konzept"
    ])
    relation_types: List[str] = field(default_factory=lambda: [
        "GEBOREN_IN", "GESTORBEN_IN", "WOHNT_IN", "ARBEITET_BEI",
        "STUDIERT_AN", "LEITET", "TEIL_VON", "BEFINDET_SICH_IN",
        "VERHEIRATET_MIT", "KIND_VON", "KENNT",
        "ENTWICKELTE", "ERFAND", "SCHRIEB", "ERHIELT",
        "BETEILIGT_AN", "HAT_BEZIEHUNG_ZU"
    ])


@dataclass
class PipelineResult:
    """Ergebnis einer Pipeline-Operation."""

    # Für Ingestion
    entities_extracted: int = 0
    triples_extracted: int = 0
    triples_accepted: int = 0
    triples_rejected: int = 0

    # Für Query
    answer: str = ""
    context_triples: List[Triple] = field(default_factory=list)
    confidence: float = 0.0

    # Ingestion details
    chunks_processed: int = 0
    triples_review: int = 0
    document_id: str = ""

    # Performance
    extraction_time_ms: float = 0.0
    consistency_time_ms: float = 0.0
    answer_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Details
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities_extracted": self.entities_extracted,
            "triples_extracted": self.triples_extracted,
            "triples_accepted": self.triples_accepted,
            "triples_rejected": self.triples_rejected,
            "chunks_processed": self.chunks_processed,
            "triples_review": self.triples_review,
            "document_id": self.document_id,
            "answer": self.answer,
            "confidence": self.confidence,
            "extraction_time_ms": self.extraction_time_ms,
            "consistency_time_ms": self.consistency_time_ms,
            "answer_time_ms": self.answer_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms": self.total_time_ms,
        }


# =============================================================================
# In-Memory Graph Store (für Demo/Test)
# =============================================================================

class InMemoryGraphStore:
    """
    Einfacher In-Memory Graph Store.

    In Produktion würde dies durch Neo4j ersetzt.
    """

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.triples: List[Triple] = []
        self._entity_index: Dict[str, List[str]] = {}  # name_lower -> [entity_ids]

    def add_entity(self, entity: Entity):
        """Fügt eine Entität hinzu."""
        self.entities[entity.id] = entity

        # Index aktualisieren
        key = entity.name.lower()
        if key not in self._entity_index:
            self._entity_index[key] = []
        self._entity_index[key].append(entity.id)

    def add_triple(self, triple: Triple):
        """Fügt ein Triple hinzu."""
        self.triples.append(triple)

        # Entitäten auch hinzufügen wenn nicht vorhanden
        if triple.subject.id not in self.entities:
            self.add_entity(triple.subject)
        if triple.object.id not in self.entities:
            self.add_entity(triple.object)

    def find_entity(self, name: str) -> Optional[Entity]:
        """Findet eine Entität nach Name."""
        key = name.lower()
        if key in self._entity_index:
            entity_id = self._entity_index[key][0]
            return self.entities.get(entity_id)
        return None

    def find_by_name(self, prefix: str, entity_type=None) -> List[Entity]:
        """Findet Entitäten nach Namens-Präfix."""
        results = []
        prefix = prefix.lower()
        for key, entity_ids in self._entity_index.items():
            if key.startswith(prefix):
                for eid in entity_ids:
                    entity = self.entities.get(eid)
                    if entity and (entity_type is None or entity.entity_type == entity_type):
                        results.append(entity)
        return results

    def find_relations(self, source_id: str = None, target_id: str = None) -> List[Dict]:
        """Findet Relationen."""
        results = []
        for triple in self.triples:
            if source_id and triple.subject.id == source_id:
                results.append({
                    "source": {"id": triple.subject.id, "name": triple.subject.name},
                    "rel_type": triple.predicate,
                    "target": {"id": triple.object.id, "name": triple.object.name}
                })
            elif target_id and triple.object.id == target_id:
                results.append({
                    "source": {"id": triple.subject.id, "name": triple.subject.name},
                    "rel_type": triple.predicate,
                    "target": {"id": triple.object.id, "name": triple.object.name}
                })
        return results

    def find_relations_by_type(self, rel_type: str) -> List[Dict]:
        """Findet Relationen nach Typ."""
        results = []
        for triple in self.triples:
            if triple.predicate.upper() == rel_type.upper():
                results.append({
                    "source": {"id": triple.subject.id, "name": triple.subject.name},
                    "rel_type": triple.predicate,
                    "target": {"id": triple.object.id, "name": triple.object.name}
                })
        return results

    def search(self, query: str, limit: int = 20) -> List[Triple]:
        """
        Sucht relevante Tripel für eine Query.

        Einfache Keyword-Suche - in Produktion würde Embedding-basierte
        Suche verwendet.
        """
        query_words = set(query.lower().split())
        scored_triples = []

        for triple in self.triples:
            if triple.validation_status != ValidationStatus.ACCEPTED:
                continue

            # Score basierend auf Wort-Überlappung
            triple_text = f"{triple.subject.name} {triple.predicate} {triple.object.name}".lower()
            triple_words = set(triple_text.split())

            overlap = len(query_words & triple_words)
            if overlap > 0:
                scored_triples.append((triple, overlap))

        # Sortieren nach Score
        scored_triples.sort(key=lambda x: x[1], reverse=True)

        return [t for t, _ in scored_triples[:limit]]

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über den Graph zurück."""
        accepted = sum(1 for t in self.triples if t.validation_status == ValidationStatus.ACCEPTED)
        rejected = sum(1 for t in self.triples if t.validation_status == ValidationStatus.REJECTED)

        return {
            "num_entities": len(self.entities),
            "num_triples": len(self.triples),
            "triples_accepted": accepted,
            "triples_rejected": rejected,
        }


# =============================================================================
# GraphRAG Pipeline
# =============================================================================

class GraphRAGPipeline:
    """
    Vollständige GraphRAG Pipeline mit Konsistenzprüfung.

    Komponenten:
    - TripleExtractor: Extrahiert Wissen aus Text
    - ConsistencyOrchestrator: Validiert Tripel
    - GraphStore: Speichert Knowledge Graph
    - AnswerGenerator: Generiert Antworten
    """

    def __init__(
        self,
        config: PipelineConfig = None,
        llm_client: Any = None,
        embedding_model: Any = None,
        graph_store: Any = None,
        use_neo4j: bool = None
    ):
        """
        Args:
            config: Pipeline-Konfiguration
            llm_client: OpenAI-kompatibles LLM
            embedding_model: Embedding-Modell
            graph_store: Graph-Speicher (default: InMemory)
            use_neo4j: True für Neo4j via Repository Factory,
                       False/None für InMemory (ignoriert wenn graph_store gegeben)
        """
        self.config = config or PipelineConfig()
        self.llm_client = llm_client
        self.embedding_model = embedding_model

        # Graph Store
        if graph_store is not None:
            self.graph_store = graph_store
        elif use_neo4j is not None:
            from src.graph.repository_factory import create_repository
            self.graph_store = create_repository(use_neo4j=use_neo4j)
        else:
            self.graph_store = InMemoryGraphStore()

        # Statistiken
        self._total_documents = 0
        self._total_triples = 0

        # Triple Extractor
        extraction_config = ExtractionConfig(
            model=self.config.extraction_model,
            entity_types=self.config.entity_types,
            relation_types=self.config.relation_types
        )
        self.extractor = TripleExtractor(extraction_config, llm_client)

        # Consistency Orchestrator
        if self.config.enable_consistency:
            consistency_config = ConsistencyConfig(
                valid_entity_types=self.config.entity_types,
                valid_relation_types=self.config.relation_types,
                high_confidence_threshold=self.config.high_confidence_threshold,
                medium_confidence_threshold=self.config.medium_confidence_threshold,
                llm_model=self.config.extraction_model
            )
            self.consistency = ConsistencyOrchestrator(
                config=consistency_config,
                graph_repo=self.graph_store,
                embedding_model=embedding_model,
                llm_client=llm_client,
                enable_metrics=True
            )
        else:
            self.consistency = None

        logger.info("GraphRAG Pipeline initialisiert")
        logger.info(f"  → Konsistenzprüfung: {'aktiviert' if self.config.enable_consistency else 'deaktiviert'}")

    def ingest(
        self,
        text: str,
        document_id: str = None,
        metadata: Dict = None
    ) -> PipelineResult:
        """
        Nimmt ein Dokument auf und integriert es in den Knowledge Graph.

        Args:
            text: Dokumenttext
            document_id: Optionale Dokument-ID
            metadata: Optionale Metadaten

        Returns:
            PipelineResult mit Statistiken
        """
        start_time = time.time()
        result = PipelineResult()
        result.document_id = document_id or f"doc_{int(time.time())}"

        # 1. Extraktion
        extraction_start = time.time()
        extraction_result = self.extractor.extract(text, document_id, metadata)
        result.extraction_time_ms = (time.time() - extraction_start) * 1000

        result.entities_extracted = len(extraction_result.entities)
        result.triples_extracted = len(extraction_result.triples)
        result.chunks_processed = getattr(extraction_result, 'chunks_processed', 1)

        logger.info(f"Extraktion: {result.entities_extracted} Entitäten, "
                   f"{result.triples_extracted} Tripel")

        # 2. Konsistenzprüfung und Integration
        consistency_start = time.time()

        for triple in extraction_result.triples:
            if self.consistency:
                validated = self.consistency.process(triple)
            else:
                validated = triple
                validated.validation_status = ValidationStatus.ACCEPTED

            # In Graph speichern
            if validated.validation_status == ValidationStatus.ACCEPTED:
                self.graph_store.add_triple(validated)
                result.triples_accepted += 1
            elif validated.validation_status == ValidationStatus.REJECTED:
                result.triples_rejected += 1
            else:
                result.triples_review += 1

        result.consistency_time_ms = (time.time() - consistency_start) * 1000

        result.total_time_ms = (time.time() - start_time) * 1000

        self._total_documents += 1
        self._total_triples += result.triples_accepted

        logger.info(f"Integration: {result.triples_accepted} akzeptiert, "
                   f"{result.triples_rejected} abgelehnt, "
                   f"{result.triples_review} zur Review")

        return result

    def query(
        self,
        question: str,
        context: str = None
    ) -> PipelineResult:
        """
        Beantwortet eine Frage basierend auf dem Knowledge Graph.

        Args:
            question: Die Frage
            context: Optionaler zusätzlicher Kontext

        Returns:
            PipelineResult mit Antwort
        """
        start_time = time.time()
        result = PipelineResult()

        # 1. Relevante Tripel suchen
        retrieval_start = time.time()
        relevant_triples = self.graph_store.search(
            question,
            limit=self.config.max_context_triples
        )
        result.retrieval_time_ms = (time.time() - retrieval_start) * 1000

        result.context_triples = relevant_triples

        # 2. Kontext aufbauen
        graph_context = self._build_context(relevant_triples)

        # Zusätzlichen Kontext hinzufügen wenn vorhanden
        full_context = graph_context
        if context:
            full_context = f"Zusätzlicher Kontext:\n{context}\n\n{graph_context}"

        # 3. Antwort generieren
        answer_start = time.time()
        result.answer, result.confidence = self._generate_answer(question, full_context)
        result.answer_time_ms = (time.time() - answer_start) * 1000
        result.generation_time_ms = result.answer_time_ms

        result.total_time_ms = (time.time() - start_time) * 1000

        result.details = {
            "num_context_triples": len(relevant_triples),
            "graph_context_length": len(graph_context),
        }

        return result

    def _build_context(self, triples: List[Triple]) -> str:
        """Baut Kontext aus Tripeln für die Antwortgenerierung."""
        if not triples:
            return "Keine relevanten Informationen im Knowledge Graph gefunden."

        lines = ["Bekannte Fakten aus dem Knowledge Graph:"]

        for triple in triples:
            line = f"- {triple.subject.name} {triple.predicate.replace('_', ' ').lower()} {triple.object.name}"

            # Quelle hinzufügen wenn vorhanden
            if triple.source_text:
                line += f" (Quelle: \"{triple.source_text[:100]}...\")"

            lines.append(line)

        return "\n".join(lines)

    def _generate_answer(
        self,
        question: str,
        context: str
    ) -> Tuple[str, float]:
        """Generiert eine Antwort mit dem LLM."""
        if not self.llm_client:
            # Mock-Antwort
            return self._mock_answer(question, context), 0.5

        try:
            prompt = f"""Beantworte die folgende Frage basierend auf dem gegebenen Kontext.
Wenn die Antwort nicht im Kontext zu finden ist, sage "Ich weiß es nicht basierend auf den verfügbaren Informationen."

Kontext:
{context}

Frage: {question}

Antwort (kurz und präzise):"""

            response = self.llm_client.chat.completions.create(
                model=self.config.answer_model,
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreicher Assistent, der Fragen basierend auf einem Knowledge Graph beantwortet."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.answer_temperature,
                max_tokens=self.config.answer_max_tokens
            )

            answer = response.choices[0].message.content.strip()

            # Konfidenz schätzen (basierend auf Anzahl relevanter Tripel)
            confidence = min(0.9, 0.5 + len(context) / 2000)

            return answer, confidence

        except Exception as e:
            logger.error(f"Antwortgenerierung fehlgeschlagen: {e}")
            return "Fehler bei der Antwortgenerierung.", 0.0

    def _mock_answer(self, question: str, context: str) -> str:
        """Mock-Antwort für Tests ohne LLM."""
        # Versuche einfache Antwort aus Kontext zu extrahieren
        question_words = question.lower().split()

        for line in context.split('\n'):
            line_lower = line.lower()
            matches = sum(1 for word in question_words if word in line_lower)
            if matches >= 2:
                # Extrahiere mögliche Antwort
                if '-' in line:
                    return line.split('-', 1)[1].strip()

        return "Basierend auf dem Knowledge Graph kann ich diese Frage nicht beantworten."

    def ingest_document(
        self,
        file_path: str,
        document_id: str = None
    ) -> PipelineResult:
        """
        Nimmt ein Dokument auf und integriert es in den Knowledge Graph.

        Unterstützte Formate:
        - .txt: Plain Text
        - .pdf: PDF (benötigt pypdf)
        - .md: Markdown

        Args:
            file_path: Pfad zum Dokument
            document_id: Optionale Dokument-ID

        Returns:
            PipelineResult mit Statistiken
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dokument nicht gefunden: {file_path}")

        doc_id = document_id or path.stem
        text = self._load_document(path)

        return self.ingest(text, document_id=doc_id)

    def _load_document(self, path: Path) -> str:
        """Lädt ein Dokument basierend auf dem Dateityp."""
        suffix = path.suffix.lower()

        if suffix in [".txt", ".md"]:
            return path.read_text(encoding="utf-8")

        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                raise ImportError("pypdf benötigt für PDF-Verarbeitung: pip install pypdf")

        else:
            # Versuche als Text zu laden
            return path.read_text(encoding="utf-8")

    def print_summary(self):
        """Gibt eine formatierte Zusammenfassung der Pipeline aus."""
        if self.consistency:
            self.consistency.print_summary()
        else:
            stats = self.get_graph_statistics()
            print("\n" + "=" * 60)
            print("GRAPHRAG PIPELINE - SUMMARY")
            print("=" * 60)
            print(f"  Dokumente verarbeitet: {self._total_documents}")
            print(f"  Tripel akzeptiert: {self._total_triples}")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über den Knowledge Graph zurück."""
        graph_stats = self.graph_store.get_statistics()

        if self.consistency:
            consistency_stats = self.consistency.get_statistics()
            graph_stats["consistency"] = consistency_stats

        return graph_stats

    def export_metrics(self, path: str, format: str = "json"):
        """Exportiert Metriken."""
        if self.consistency:
            self.consistency.export_metrics(path, format)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_pipeline(
    openai_api_key: str = None,
    enable_consistency: bool = True
) -> GraphRAGPipeline:
    """
    Erstellt eine GraphRAG Pipeline mit OpenAI.

    Args:
        openai_api_key: OpenAI API Key
        enable_consistency: Konsistenzprüfung aktivieren

    Returns:
        Konfigurierte Pipeline
    """
    llm_client = None

    if openai_api_key:
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=openai_api_key)
        except ImportError:
            logger.warning("openai Paket nicht installiert")

    config = PipelineConfig(enable_consistency=enable_consistency)

    return GraphRAGPipeline(config, llm_client)


def main():
    """Hauptfunktion für CLI-Nutzung."""
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG Pipeline")
    parser.add_argument("--mode", choices=["demo", "ingest", "query"], default="demo")
    parser.add_argument("--file", help="Datei für Ingestion")
    parser.add_argument("--question", help="Frage für Query")
    parser.add_argument("--neo4j", action="store_true", help="Neo4j verwenden")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("GraphRAG Pipeline")
    print("=" * 60)

    # Pipeline erstellen
    pipeline = GraphRAGPipeline(use_neo4j=args.neo4j if args.neo4j else None)

    if args.mode == "demo":
        # Demo-Durchlauf
        test_doc = """
        Albert Einstein wurde am 14. März 1879 in Ulm geboren.
        Er studierte Physik an der ETH Zürich in der Schweiz.
        1905 veröffentlichte er die spezielle Relativitätstheorie.
        Einstein arbeitete später am Institute for Advanced Study in Princeton.
        Er erhielt 1921 den Nobelpreis für Physik.
        Marie Curie war eine Freundin von Einstein.
        Sie erhielt den Nobelpreis für Physik und Chemie.
        """

        print("\n--- Dokument-Ingestion ---")
        result = pipeline.ingest(test_doc, "demo_doc")
        print(f"Extrahiert: {result.entities_extracted} Entitäten, {result.triples_extracted} Tripel")
        print(f"Akzeptiert: {result.triples_accepted}, Abgelehnt: {result.triples_rejected}")
        print(f"Zeit: {result.total_time_ms:.0f}ms")

        print("\n--- Frage-Antwort ---")
        questions = [
            "Wo wurde Einstein geboren?",
            "Wer erhielt den Nobelpreis für Physik?",
            "Wo studierte Einstein?",
        ]

        for q in questions:
            result = pipeline.query(q)
            print(f"\nFrage: {q}")
            print(f"Antwort: {result.answer}")
            print(f"Kontext-Tripel: {len(result.context_triples)}")

        print("\n--- Statistiken ---")
        pipeline.print_summary()

    elif args.mode == "ingest" and args.file:
        result = pipeline.ingest_document(args.file)
        print(f"Ingestion-Ergebnis: {result.to_dict()}")

    elif args.mode == "query" and args.question:
        result = pipeline.query(args.question)
        print(f"Antwort: {result.answer}")
        print(f"Konfidenz: {result.confidence:.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
