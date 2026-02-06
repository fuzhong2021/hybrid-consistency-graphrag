# src/extraction/triple_extractor.py
"""
LLM-basierter Triple-Extractor für Knowledge Graph Aufbau.

Implementiert einen mehrstufigen Extraktionsprozess basierend auf:
- Microsoft GraphRAG (2024): Hierarchische Extraktion
- iText2KG (Lairgi et al., 2024): Iterative Verfeinerung mit Schema
- Graphiti (Rasmussen et al., 2025): Bi-temporale Awareness

Pipeline:
1. Document Chunking
2. Entity Extraction (pro Chunk)
3. Relation Extraction
4. Entity Resolution (Deduplizierung)
5. Konsistenzprüfung
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from src.models.entities import (
    Entity, EntityType, Triple, ValidationStatus,
    EntityResolutionResult, MergeStrategy
)
from src.extraction.prompts import PromptBuilder, SYSTEM_PROMPT_EXTRACTION
from src.extraction.chunking import DocumentChunker, ChunkingConfig, ChunkingStrategy, TextChunk

logger = logging.getLogger(__name__)


# =============================================================================
# Konfiguration
# =============================================================================

@dataclass
class ExtractionConfig:
    """Konfiguration für die Triple-Extraktion."""

    # LLM-Einstellungen
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.0
    max_tokens: int = 4000

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

    # Chunking
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # Extraktion
    extraction_mode: str = "combined"  # "combined", "two_pass"
    use_few_shot: bool = True
    min_confidence: float = 0.5

    # Verifikation
    verify_extractions: bool = True
    use_coreference: bool = True


@dataclass
class ExtractionResult:
    """Ergebnis einer Extraktion."""
    entities: List[Entity] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)

    # Statistiken
    chunks_processed: int = 0
    extraction_time_ms: float = 0.0
    llm_calls: int = 0
    tokens_used: int = 0

    # Rohdaten (für Debugging)
    raw_extractions: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_entities": len(self.entities),
            "num_triples": len(self.triples),
            "chunks_processed": self.chunks_processed,
            "extraction_time_ms": self.extraction_time_ms,
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used,
        }


# =============================================================================
# Triple Extractor
# =============================================================================

class TripleExtractor:
    """
    LLM-basierter Extractor für Knowledge Graph Tripel.

    Workflow:
    1. Text in Chunks aufteilen
    2. Pro Chunk: Entitäten und Relationen extrahieren
    3. Entitäten über Chunks hinweg auflösen (Deduplizierung)
    4. Tripel erstellen und validieren
    """

    def __init__(
        self,
        config: ExtractionConfig = None,
        llm_client: Any = None
    ):
        """
        Args:
            config: Extraktions-Konfiguration
            llm_client: OpenAI-kompatibles LLM (mit .chat.completions.create)
        """
        self.config = config or ExtractionConfig()
        self.llm_client = llm_client

        # Prompt Builder
        self.prompt_builder = PromptBuilder(
            entity_types=self.config.entity_types,
            relation_types=self.config.relation_types,
            use_few_shot=self.config.use_few_shot
        )

        # Document Chunker
        chunking_config = ChunkingConfig(
            strategy=self.config.chunking_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.chunker = DocumentChunker(chunking_config)

        # Statistiken
        self._total_llm_calls = 0
        self._total_tokens = 0

        logger.info(f"TripleExtractor initialisiert (Model: {self.config.model})")

    def extract(
        self,
        text: str,
        document_id: str = None,
        metadata: Dict = None
    ) -> ExtractionResult:
        """
        Extrahiert Wissen aus einem Text.

        Args:
            text: Zu analysierender Text
            document_id: Optionale Dokument-ID
            metadata: Optionale Metadaten

        Returns:
            ExtractionResult mit Entitäten und Tripeln
        """
        start_time = time.time()
        result = ExtractionResult()

        if not text or not text.strip():
            logger.warning("Leerer Text - keine Extraktion")
            return result

        # 1. Chunking
        chunks = self.chunker.chunk(text, document_id)
        logger.info(f"Text in {len(chunks)} Chunks aufgeteilt")

        # 2. Extraktion pro Chunk
        all_entities: Dict[str, Entity] = {}  # name -> Entity
        all_relations: List[Dict] = []

        for i, chunk in enumerate(chunks):
            logger.debug(f"Verarbeite Chunk {i+1}/{len(chunks)}")

            chunk_result = self._extract_from_chunk(chunk)

            if chunk_result:
                result.raw_extractions.append(chunk_result)

                # Entitäten sammeln (mit Deduplizierung)
                for entity_data in chunk_result.get("entities", []):
                    entity = self._create_entity(entity_data, document_id)
                    if entity:
                        # Einfache Deduplizierung nach Name
                        key = entity.name.lower().strip()
                        if key not in all_entities:
                            all_entities[key] = entity
                        else:
                            # Merge: Höhere Konfidenz, mehr Aliases
                            existing = all_entities[key]
                            if entity.confidence > existing.confidence:
                                entity.aliases = list(set(entity.aliases + existing.aliases))
                                all_entities[key] = entity

                # Relationen sammeln
                for rel in chunk_result.get("relations", []):
                    rel["chunk_id"] = chunk.chunk_id
                    all_relations.append(rel)

            result.chunks_processed += 1

        # 3. Tripel erstellen
        result.entities = list(all_entities.values())
        result.triples = self._create_triples(result.entities, all_relations, text)

        # Statistiken
        result.extraction_time_ms = (time.time() - start_time) * 1000
        result.llm_calls = self._total_llm_calls
        result.tokens_used = self._total_tokens

        logger.info(f"Extraktion abgeschlossen: {len(result.entities)} Entitäten, "
                   f"{len(result.triples)} Tripel in {result.extraction_time_ms:.0f}ms")

        return result

    def _extract_from_chunk(self, chunk: TextChunk) -> Optional[Dict]:
        """Extrahiert Entitäten und Relationen aus einem Chunk."""
        if not self.llm_client:
            logger.warning("Kein LLM-Client - verwende Mock-Extraktion")
            return self._mock_extraction(chunk.text)

        try:
            # Prompt erstellen
            messages = self.prompt_builder.build_extraction_prompt(
                chunk.text,
                mode=self.config.extraction_mode
            )

            # LLM aufrufen
            response = self.llm_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )

            # Token-Tracking
            self._total_llm_calls += 1
            if hasattr(response, 'usage'):
                self._total_tokens += response.usage.total_tokens

            # Response parsen
            content = response.choices[0].message.content
            result = json.loads(content)

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Fehler: {e}")
            return None
        except Exception as e:
            logger.error(f"Extraktion fehlgeschlagen: {e}")
            return None

    def _mock_extraction(self, text: str) -> Dict:
        """
        Mock-Extraktion für Tests ohne LLM.

        Verwendet erweiterte Heuristiken zur Demonstration.
        """
        import re

        entities = []
        relations = []
        text_lower = text.lower()

        # Pattern für verschiedene Relationen
        relation_patterns = [
            # Person geboren in Ort
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was\s+)?born\s+(?:on\s+[\w\s,]+\s+)?in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             "GEBOREN_IN", "Person", "Ort"),
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+wurde\s+(?:am\s+[\w\s.]+\s+)?in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+geboren",
             "GEBOREN_IN", "Person", "Ort"),

            # Person studierte an Organisation
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:studied|studierte)\s+(?:\w+\s+)?(?:at|an)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             "STUDIERT_AN", "Person", "Organisation"),

            # Person arbeitet bei Organisation
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:worked|arbeitete)\s+(?:at|bei)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             "ARBEITET_BEI", "Person", "Organisation"),

            # Person erhielt Auszeichnung
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:received|erhielt|won)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Prize)?)",
             "ERHIELT", "Person", "Ereignis"),

            # Person kennt Person (friend)
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+a\s+friend\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             "KENNT", "Person", "Person"),
        ]

        seen_entities = {}  # name_lower -> entity

        def add_entity(name: str, entity_type: str) -> Dict:
            """Fügt Entity hinzu oder gibt existierende zurück."""
            key = name.lower()
            if key not in seen_entities:
                seen_entities[key] = {
                    "name": name,
                    "type": entity_type,
                    "description": "",
                    "confidence": 0.7
                }
            return seen_entities[key]

        # Relationen extrahieren
        for pattern, rel_type, source_type, target_type in relation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                source_name, target_name = match[0], match[1]

                # Entitäten erstellen
                source_entity = add_entity(source_name, source_type)
                target_entity = add_entity(target_name, target_type)

                # Relation erstellen
                relations.append({
                    "source": source_entity["name"],
                    "relation": rel_type,
                    "target": target_entity["name"],
                    "evidence": f"Pattern Match: {rel_type}",
                    "confidence": 0.7
                })

        # Zusätzliche Entitäten aus Großbuchstaben (ohne Duplikate)
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        for name in capitalized:
            if name.lower() not in seen_entities:
                # Typ erraten
                entity_type = "Konzept"
                if any(word in name.lower() for word in ["institute", "university", "eth", "company"]):
                    entity_type = "Organisation"
                elif any(word in name.lower() for word in ["prize", "award", "nobel"]):
                    entity_type = "Ereignis"
                elif len(name.split()) == 2:  # Zwei Wörter = wahrscheinlich Person
                    entity_type = "Person"

                add_entity(name, entity_type)

        entities = list(seen_entities.values())

        return {"entities": entities, "relations": relations}

    def _create_entity(
        self,
        entity_data: Dict,
        document_id: str = None
    ) -> Optional[Entity]:
        """Erstellt eine Entity aus extrahierten Daten."""
        try:
            name = entity_data.get("name", "").strip()
            if not name:
                return None

            # Typ parsen
            type_str = entity_data.get("type", "Konzept")
            entity_type = EntityType.from_string(type_str)

            # Temporal-Daten parsen
            temporal = entity_data.get("temporal", {}) or {}
            valid_from = self._parse_date(temporal.get("valid_from"))
            valid_until = self._parse_date(temporal.get("valid_until"))

            entity = Entity(
                name=name,
                entity_type=entity_type,
                description=entity_data.get("description", ""),
                aliases=entity_data.get("aliases", []),
                confidence=entity_data.get("confidence", 0.8),
                valid_from=valid_from,
                valid_until=valid_until,
                source_document=document_id,
                validation_status=ValidationStatus.PENDING
            )

            return entity

        except Exception as e:
            logger.error(f"Entity-Erstellung fehlgeschlagen: {e}")
            return None

    def _create_triples(
        self,
        entities: List[Entity],
        relations: List[Dict],
        source_text: str
    ) -> List[Triple]:
        """Erstellt Triple-Objekte aus extrahierten Relationen."""
        triples = []

        # Entity-Lookup nach Name
        entity_lookup = {e.name.lower(): e for e in entities}

        for rel in relations:
            source_name = (rel.get("source") or "").lower()
            target_name = (rel.get("target") or "").lower()

            source_entity = entity_lookup.get(source_name)
            target_entity = entity_lookup.get(target_name)

            if not source_entity or not target_entity:
                logger.debug(f"Entität nicht gefunden: {source_name} oder {target_name}")
                continue

            # Predicate normalisieren
            predicate = rel.get("relation", "HAT_BEZIEHUNG_ZU")
            predicate = predicate.upper().replace(" ", "_")

            # Konfidenz prüfen
            confidence = rel.get("confidence", 0.8)
            if confidence < self.config.min_confidence:
                logger.debug(f"Überspringe Relation mit niedriger Konfidenz: {confidence}")
                continue

            # Temporal-Daten
            temporal = rel.get("temporal", {}) or {}

            triple = Triple(
                subject=source_entity,
                predicate=predicate,
                object=target_entity,
                source_text=rel.get("evidence", "")[:500],
                source_chunk_id=rel.get("chunk_id"),
                extraction_confidence=confidence,
                validation_status=ValidationStatus.PENDING
            )

            triples.append(triple)

        return triples

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parst ein Datum aus String."""
        if not date_str:
            return None

        try:
            # Verschiedene Formate probieren
            for fmt in ["%Y-%m-%d", "%Y-%m", "%Y", "%d.%m.%Y", "%d/%m/%Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def extract_and_validate(
        self,
        text: str,
        consistency_orchestrator: Any,
        document_id: str = None
    ) -> Tuple[ExtractionResult, List[Triple]]:
        """
        Extrahiert und validiert Tripel in einem Durchgang.

        Args:
            text: Zu analysierender Text
            consistency_orchestrator: Konsistenz-Orchestrator
            document_id: Optionale Dokument-ID

        Returns:
            Tuple von (ExtractionResult, validierte_triples)
        """
        # Extraktion
        result = self.extract(text, document_id)

        # Validierung
        validated_triples = []
        for triple in result.triples:
            validated = consistency_orchestrator.process(triple)
            validated_triples.append(validated)

        return result, validated_triples


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_knowledge(
    text: str,
    llm_client: Any = None,
    entity_types: List[str] = None,
    relation_types: List[str] = None
) -> ExtractionResult:
    """
    Convenience-Funktion für Wissensextraktion.

    Args:
        text: Zu analysierender Text
        llm_client: Optionaler LLM-Client
        entity_types: Optionale Entity-Typen
        relation_types: Optionale Relation-Typen

    Returns:
        ExtractionResult
    """
    config = ExtractionConfig()
    if entity_types:
        config.entity_types = entity_types
    if relation_types:
        config.relation_types = relation_types

    extractor = TripleExtractor(config, llm_client)
    return extractor.extract(text)


if __name__ == "__main__":
    # Test ohne LLM
    logging.basicConfig(level=logging.INFO)

    test_text = """
    Albert Einstein wurde am 14. März 1879 in Ulm geboren.
    Er studierte Physik an der ETH Zürich in der Schweiz.
    1905 veröffentlichte er die spezielle Relativitätstheorie.
    Einstein arbeitete später am Institute for Advanced Study in Princeton.
    Er erhielt 1921 den Nobelpreis für Physik.
    Einstein starb am 18. April 1955 in Princeton.
    """

    print("\n=== Triple Extractor Test (Mock) ===\n")

    extractor = TripleExtractor()
    result = extractor.extract(test_text, "test_doc")

    print(f"Entitäten ({len(result.entities)}):")
    for e in result.entities:
        print(f"  - {e.name} ({e.entity_type.value})")

    print(f"\nTripel ({len(result.triples)}):")
    for t in result.triples:
        print(f"  - {t}")

    print(f"\nStatistiken: {result.to_dict()}")
