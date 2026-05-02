#!/usr/bin/env python3
# evaluation/streaming/conflict_taxonomy.py
"""
Wissenschaftliche Taxonomie für Knowledge Graph Konflikte.

Diese Taxonomie klassifiziert alle relevanten Konflikt-Typen für
die Konsistenzprüfung in Knowledge Graphs. Jeder Typ ist:
- Formal definiert
- Mit wissenschaftlicher Referenz versehen
- Mit Ground Truth Kriterien spezifiziert
- Mit Erkennungsmethode annotiert

═══════════════════════════════════════════════════════════════════════════════
TAXONOMIE DER KG-KONFLIKTE (10 Hauptkategorien)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 1: FAKTISCHE WIDERSPRÜCHE (Factual Contradictions)               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Zwei Aussagen die nicht gleichzeitig wahr sein können         │
│ Beispiel:    (Einstein, birthPlace, Ulm) vs (Einstein, birthPlace, Munich) │
│ Referenz:    Thorne et al. (2018) - FEVER Dataset                          │
│ Erkennung:   NLI (Contradiction) + Kardinalität                            │
│ Ground Truth: REJECT das später ankommende Triple                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 2: TEMPORALE KONFLIKTE (Temporal Conflicts)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Aussagen die zu verschiedenen Zeitpunkten gültig sind         │
│ Beispiel:    (Einstein, profession, Professor@Berlin[1914-1932])           │
│              (Einstein, profession, Professor@Princeton[1933-1955])        │
│ Referenz:    Leblay & Chekol (2018) - Temporal Knowledge Graphs            │
│              Lacroix et al. (2020) - Tensor Decomposition for TKG          │
│ Erkennung:   Temporal Reasoning + Intervall-Überlappung                    │
│ Ground Truth: ACCEPT beide wenn Intervalle disjunkt, sonst CONFLICT        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 3: GRANULARITÄTS-DIFFERENZEN (Granularity Mismatches)            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Aussagen auf verschiedenen Abstraktionsebenen                 │
│ Beispiel:    (Einstein, birthPlace, Germany) [coarse]                      │
│              (Einstein, birthPlace, Ulm) [fine]                            │
│ Referenz:    Hobbs (1985) - Granularity in AI                              │
│              Bittner & Smith (2003) - Granular Partitions                  │
│ Erkennung:   Ontologie-Hierarchie (Ulm ⊂ Germany)                          │
│ Ground Truth: ACCEPT beide, MERGE zu feinerem Level                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 4: ENTITY-VARIANTEN (Entity Coreference)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Verschiedene Namen für dieselbe Entität                       │
│ Beispiel:    "Albert Einstein" vs "A. Einstein" vs "Prof. Einstein"        │
│ Referenz:    Lairgi et al. (2024) - iText2KG Entity Resolution             │
│              Shen et al. (2015) - Entity Linking Survey                    │
│ Erkennung:   Embedding Similarity + String Matching (α=0.6/0.4)            │
│ Ground Truth: MERGE zu kanonischer Entity                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 5: IMPLIZITE WIDERSPRÜCHE (Implicit Contradictions)              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Widersprüche die Weltwissen/Inferenz erfordern                │
│ Beispiel:    (Einstein, hasChild, none) vs (Hans_Albert, father, Einstein) │
│ Referenz:    Razniewski et al. (2016) - Negative Statements in KGs         │
│              Arnaout et al. (2022) - Negative Knowledge in Wikidata        │
│ Erkennung:   LLM-basierte Inferenz + Regelbasiert                          │
│ Ground Truth: REJECT das inkonsistente Triple                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 6: NEGATIONS-KONFLIKTE (Negation Conflicts)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Direkte Negation einer Aussage                                │
│ Beispiel:    (Einstein, wonAward, NobelPrize) vs                           │
│              (Einstein, NOT_wonAward, NobelPrize)                          │
│ Referenz:    Arnaout et al. (2021) - Negative Statements                   │
│ Erkennung:   Pattern Matching + NLI                                        │
│ Ground Truth: REJECT die falsche Aussage                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 7: MODALITÄTS-KONFLIKTE (Modality Conflicts)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Unterschiedliche Gewissheitsgrade                             │
│ Beispiel:    "Einstein was definitely born in Ulm"                         │
│              "Einstein was possibly born in Munich"                        │
│ Referenz:    Safavi et al. (2020) - Uncertain Knowledge Graphs             │
│              Chen et al. (2019) - Probabilistic Knowledge Graphs           │
│ Erkennung:   Modal Marker Detection + Confidence Adjustment                │
│ Ground Truth: ACCEPT mit angepasster Konfidenz                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 8: SOURCE-QUALITÄTS-KONFLIKTE (Source Reliability)               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Gleiche Aussage aus unterschiedlich vertrauenswürdigen Quellen│
│ Beispiel:    Wikipedia (reliability=0.85) vs Twitter (reliability=0.3)     │
│ Referenz:    Dong et al. (2015) - Knowledge Vault Source Quality           │
│              Pasternack & Roth (2013) - Source Reliability                 │
│ Erkennung:   Source Credibility Scoring                                    │
│ Ground Truth: WEIGHT by source reliability                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 9: SCHEMA-HETEROGENITÄT (Schema Conflicts)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Gleiche Semantik, verschiedene Schemata                       │
│ Beispiel:    schema:birthPlace vs dbo:placeOfBirth vs wdt:P19              │
│ Referenz:    Paulheim (2017) - Knowledge Graph Refinement Survey           │
│              Euzenat & Shvaiko (2013) - Ontology Matching                  │
│ Erkennung:   Schema Alignment + Ontology Matching                          │
│ Ground Truth: MERGE unter kanonischem Schema                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ KATEGORIE 10: NUMERISCHE PRÄZISION (Numerical Precision)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Definition:  Numerische Werte mit unterschiedlicher Präzision              │
│ Beispiel:    (Einstein, birthYear, 1879) vs (Einstein, birthYear, "1870s") │
│ Referenz:    Hoffart et al. (2013) - YAGO2 Temporal Facts                  │
│ Erkennung:   Numerical Range Overlap                                       │
│ Ground Truth: MERGE zu präziserem Wert wenn konsistent                     │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

Autor: Masterarbeit GraphRAG Konsistenzprüfung
Basierend auf systematischer Literaturanalyse
"""

import logging
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS UND DATENKLASSEN
# =============================================================================

class ConflictType(Enum):
    """Die 10 Hauptkategorien von KG-Konflikten."""
    FACTUAL = "factual"                    # Kategorie 1
    TEMPORAL = "temporal"                  # Kategorie 2
    GRANULARITY = "granularity"            # Kategorie 3
    ENTITY_VARIANT = "entity_variant"      # Kategorie 4
    IMPLICIT = "implicit"                  # Kategorie 5
    NEGATION = "negation"                  # Kategorie 6
    MODALITY = "modality"                  # Kategorie 7
    SOURCE_QUALITY = "source_quality"      # Kategorie 8
    SCHEMA = "schema"                      # Kategorie 9
    NUMERICAL = "numerical"                # Kategorie 10


class GroundTruthAction(Enum):
    """Mögliche Ground Truth Aktionen."""
    ACCEPT = "accept"           # Triple akzeptieren
    REJECT = "reject"           # Triple ablehnen
    MERGE = "merge"             # Mit existierendem Triple mergen
    CONFLICT = "conflict"       # Als Konflikt markieren (manuell prüfen)
    WEIGHT = "weight"           # Mit Konfidenz gewichten


class DetectionMethod(Enum):
    """Erkennungsmethoden für Konflikte."""
    NLI = "nli"                           # Natural Language Inference
    CARDINALITY = "cardinality"           # Kardinalitätsregeln
    EMBEDDING = "embedding"               # Embedding-Similarität
    TEMPORAL_REASONING = "temporal"       # Temporales Reasoning
    ONTOLOGY = "ontology"                 # Ontologie-Hierarchie
    LLM_INFERENCE = "llm"                 # LLM-basierte Inferenz
    PATTERN_MATCHING = "pattern"          # Regelbasiertes Pattern Matching
    NUMERICAL = "numerical"               # Numerischer Vergleich
    SOURCE_SCORING = "source"             # Quellen-Bewertung
    SCHEMA_ALIGNMENT = "schema"           # Schema-Alignment


@dataclass
class ScientificReference:
    """Wissenschaftliche Referenz für eine Konflikt-Kategorie."""
    authors: str
    year: int
    title: str
    venue: str
    relevance: str  # Warum diese Referenz relevant ist

    def __str__(self) -> str:
        return f"{self.authors} ({self.year}): {self.title}. {self.venue}"


@dataclass
class ConflictCategory:
    """
    Definition einer Konflikt-Kategorie.

    Enthält alle wissenschaftlich relevanten Informationen:
    - Formale Definition
    - Beispiele
    - Erkennungsmethoden
    - Ground Truth Kriterien
    - Wissenschaftliche Referenzen
    """
    conflict_type: ConflictType
    name: str
    definition: str
    examples: List[Tuple[str, str]]  # Paare von widersprüchlichen Aussagen
    detection_methods: List[DetectionMethod]
    ground_truth_action: GroundTruthAction
    ground_truth_condition: str  # Wann welche Aktion
    references: List[ScientificReference]
    requires_world_knowledge: bool = False
    requires_temporal_data: bool = False
    requires_ontology: bool = False
    difficulty: str = "medium"  # easy, medium, hard

    @property
    def primary_detection_method(self) -> DetectionMethod:
        return self.detection_methods[0] if self.detection_methods else DetectionMethod.NLI


# =============================================================================
# TAXONOMIE-DEFINITION
# =============================================================================

CONFLICT_TAXONOMY: Dict[ConflictType, ConflictCategory] = {

    ConflictType.FACTUAL: ConflictCategory(
        conflict_type=ConflictType.FACTUAL,
        name="Faktische Widersprüche",
        definition="Zwei Aussagen die nicht gleichzeitig wahr sein können. "
                   "Verletzt das Prinzip des ausgeschlossenen Widerspruchs.",
        examples=[
            ("Einstein was born in Ulm", "Einstein was born in Munich"),
            ("The Beatles formed in 1960", "The Beatles formed in 1965"),
            ("Paris is the capital of France", "Lyon is the capital of France"),
        ],
        detection_methods=[DetectionMethod.NLI, DetectionMethod.CARDINALITY],
        ground_truth_action=GroundTruthAction.REJECT,
        ground_truth_condition="REJECT das Triple das dem etablierten Wissen widerspricht",
        references=[
            ScientificReference(
                authors="Thorne, Vlachos, Christodoulopoulos & Mittal",
                year=2018,
                title="FEVER: A Large-scale Dataset for Fact Extraction and VERification",
                venue="NAACL",
                relevance="Definiert SUPPORTS/REFUTES Labels für Fact Verification"
            ),
            ScientificReference(
                authors="Bowman, Angeli, Potts & Manning",
                year=2015,
                title="A large annotated corpus for learning natural language inference",
                venue="EMNLP",
                relevance="SNLI Dataset für NLI-Training"
            ),
        ],
        difficulty="easy",
    ),

    ConflictType.TEMPORAL: ConflictCategory(
        conflict_type=ConflictType.TEMPORAL,
        name="Temporale Konflikte",
        definition="Aussagen die zu verschiedenen, nicht-überlappenden Zeitpunkten gültig sind. "
                   "Erfordert Temporal Reasoning zur korrekten Interpretation.",
        examples=[
            ("Einstein was professor at Berlin [1914-1932]",
             "Einstein was professor at Princeton [1933-1955]"),
            ("Germany's capital was Bonn [1949-1990]",
             "Germany's capital is Berlin [1990-present]"),
            ("Trump was president [2017-2021]",
             "Biden is president [2021-present]"),
        ],
        detection_methods=[DetectionMethod.TEMPORAL_REASONING, DetectionMethod.NLI],
        ground_truth_action=GroundTruthAction.ACCEPT,
        ground_truth_condition="ACCEPT beide wenn Zeitintervalle disjunkt; "
                               "CONFLICT wenn Intervalle überlappen",
        references=[
            ScientificReference(
                authors="Leblay & Chekol",
                year=2018,
                title="Deriving Validity Time in Knowledge Graph",
                venue="WWW",
                relevance="Temporal Knowledge Graph Reasoning"
            ),
            ScientificReference(
                authors="Lacroix, Obozinski & Usunier",
                year=2020,
                title="Tensor Decompositions for Temporal Knowledge Base Completion",
                venue="ICLR",
                relevance="Temporal KG Completion mit Tensor-Methoden"
            ),
            ScientificReference(
                authors="García-Durán, Dumančić & Niepert",
                year=2018,
                title="Learning Sequence Encoders for Temporal Knowledge Graph Completion",
                venue="EMNLP",
                relevance="Sequenzbasierte Temporal KG Methoden"
            ),
        ],
        requires_temporal_data=True,
        difficulty="medium",
    ),

    ConflictType.GRANULARITY: ConflictCategory(
        conflict_type=ConflictType.GRANULARITY,
        name="Granularitäts-Differenzen",
        definition="Aussagen auf verschiedenen Abstraktionsebenen einer Hierarchie. "
                   "Keine echten Widersprüche, aber unterschiedliche Spezifizität.",
        examples=[
            ("Einstein was born in Germany", "Einstein was born in Ulm"),
            ("Einstein worked in Europe", "Einstein worked in Berlin"),
            ("The event happened in 2020", "The event happened on March 15, 2020"),
        ],
        detection_methods=[DetectionMethod.ONTOLOGY, DetectionMethod.EMBEDDING],
        ground_truth_action=GroundTruthAction.MERGE,
        ground_truth_condition="MERGE zu feinerem Granularitätslevel wenn konsistent; "
                               "CONFLICT wenn inkonsistent (Ulm ⊄ France)",
        references=[
            ScientificReference(
                authors="Hobbs",
                year=1985,
                title="Granularity",
                venue="IJCAI",
                relevance="Foundational work on granularity in AI"
            ),
            ScientificReference(
                authors="Bittner & Smith",
                year=2003,
                title="A Theory of Granular Partitions",
                venue="Foundations of Geographic Information Science",
                relevance="Formale Theorie zu Granularität"
            ),
        ],
        requires_ontology=True,
        difficulty="medium",
    ),

    ConflictType.ENTITY_VARIANT: ConflictCategory(
        conflict_type=ConflictType.ENTITY_VARIANT,
        name="Entity-Varianten (Koreferenz)",
        definition="Verschiedene Oberflächenformen die dieselbe Entität referenzieren. "
                   "Erfordert Entity Resolution/Linking.",
        examples=[
            ("Albert Einstein", "A. Einstein"),
            ("The United States", "USA"),
            ("Prof. Dr. Müller", "Hans Müller"),
        ],
        detection_methods=[DetectionMethod.EMBEDDING, DetectionMethod.PATTERN_MATCHING],
        ground_truth_action=GroundTruthAction.MERGE,
        ground_truth_condition="MERGE zur kanonischen Entity-Form",
        references=[
            ScientificReference(
                authors="Lairgi, Loukili, Mouatadid & Bououd",
                year=2024,
                title="iText2KG: Incremental Knowledge Graphs Construction",
                venue="arXiv",
                relevance="Entity Resolution mit α=0.6 Name, 0.4 Embedding"
            ),
            ScientificReference(
                authors="Shen, Wang, Liu & Wang",
                year=2015,
                title="Entity Linking with a Knowledge Base: Issues, Techniques, and Solutions",
                venue="TKDE",
                relevance="Umfassender Survey zu Entity Linking"
            ),
        ],
        difficulty="easy",
    ),

    ConflictType.IMPLICIT: ConflictCategory(
        conflict_type=ConflictType.IMPLICIT,
        name="Implizite Widersprüche",
        definition="Widersprüche die nicht direkt sichtbar sind, sondern "
                   "Weltwissen oder logische Inferenz erfordern.",
        examples=[
            ("Einstein had no children", "Hans Albert Einstein's father was Albert Einstein"),
            ("X is a bachelor", "X is married"),
            ("The restaurant is vegan", "The restaurant serves steak"),
        ],
        detection_methods=[DetectionMethod.LLM_INFERENCE, DetectionMethod.NLI],
        ground_truth_action=GroundTruthAction.REJECT,
        ground_truth_condition="REJECT das Triple das dem inferierten Wissen widerspricht",
        references=[
            ScientificReference(
                authors="Razniewski, Suchanek & Nutt",
                year=2016,
                title="But What Do We Actually Know?",
                venue="AKBC",
                relevance="Negative Statements und Closed World Assumption"
            ),
            ScientificReference(
                authors="Arnaout, Razniewski & Weikum",
                year=2022,
                title="Negative Knowledge in Wikidata",
                venue="WWW",
                relevance="Wie Wikidata negative Aussagen handhabt"
            ),
        ],
        requires_world_knowledge=True,
        difficulty="hard",
    ),

    ConflictType.NEGATION: ConflictCategory(
        conflict_type=ConflictType.NEGATION,
        name="Negations-Konflikte",
        definition="Direkte Negation einer Aussage durch explizite Verneinung.",
        examples=[
            ("Einstein won the Nobel Prize", "Einstein did not win the Nobel Prize"),
            ("X is true", "X is false"),
            ("The film was released", "The film was never released"),
        ],
        detection_methods=[DetectionMethod.PATTERN_MATCHING, DetectionMethod.NLI],
        ground_truth_action=GroundTruthAction.REJECT,
        ground_truth_condition="REJECT die faktisch falsche Aussage",
        references=[
            ScientificReference(
                authors="Arnaout, Razniewski, Weikum & Ponzetto",
                year=2021,
                title="Negative Statements Considered Useful",
                venue="JAIR",
                relevance="Systematik zu negativen Aussagen in KGs"
            ),
        ],
        difficulty="easy",
    ),

    ConflictType.MODALITY: ConflictCategory(
        conflict_type=ConflictType.MODALITY,
        name="Modalitäts-Konflikte",
        definition="Aussagen mit unterschiedlichen Gewissheitsgraden oder "
                   "epistemischen Modalitäten (definitiv, möglicherweise, etc.).",
        examples=[
            ("Einstein was definitely born in Ulm", "Einstein was possibly born in Munich"),
            ("The experiment will succeed", "The experiment might succeed"),
            ("X is certainly true", "X is probably true"),
        ],
        detection_methods=[DetectionMethod.PATTERN_MATCHING, DetectionMethod.LLM_INFERENCE],
        ground_truth_action=GroundTruthAction.WEIGHT,
        ground_truth_condition="ACCEPT mit angepasster Konfidenz basierend auf Modalität",
        references=[
            ScientificReference(
                authors="Safavi, Koutra & Meij",
                year=2020,
                title="Evaluating the Calibration of Knowledge Graph Embeddings",
                venue="EMNLP",
                relevance="Unsicherheit in KG Embeddings"
            ),
            ScientificReference(
                authors="Chen, Hu, Sun & Xu",
                year=2019,
                title="Embedding Uncertain Knowledge Graphs",
                venue="AAAI",
                relevance="Probabilistische Knowledge Graphs"
            ),
        ],
        difficulty="medium",
    ),

    ConflictType.SOURCE_QUALITY: ConflictCategory(
        conflict_type=ConflictType.SOURCE_QUALITY,
        name="Source-Qualitäts-Konflikte",
        definition="Gleiche oder widersprüchliche Aussagen aus Quellen mit "
                   "unterschiedlicher Vertrauenswürdigkeit.",
        examples=[
            ("Wikipedia says X [reliability=0.85]", "Random blog says Y [reliability=0.2]"),
            ("Scientific paper states X", "Twitter post claims not-X"),
            ("Official government source", "Anonymous forum post"),
        ],
        detection_methods=[DetectionMethod.SOURCE_SCORING],
        ground_truth_action=GroundTruthAction.WEIGHT,
        ground_truth_condition="WEIGHT Konfidenz basierend auf Source Reliability Score",
        references=[
            ScientificReference(
                authors="Dong, Gabrilovich, Heitz, Horn, Murphy, Sun & Zhang",
                year=2015,
                title="From Data Fusion to Knowledge Fusion",
                venue="VLDB",
                relevance="Knowledge Vault Source Quality Estimation"
            ),
            ScientificReference(
                authors="Pasternack & Roth",
                year=2013,
                title="Latent Credibility Analysis",
                venue="WWW",
                relevance="Source Credibility Modeling"
            ),
        ],
        difficulty="medium",
    ),

    ConflictType.SCHEMA: ConflictCategory(
        conflict_type=ConflictType.SCHEMA,
        name="Schema-Heterogenität",
        definition="Semantisch äquivalente Aussagen mit unterschiedlichen "
                   "Schema-Repräsentationen (Prädikaten, Ontologien).",
        examples=[
            ("schema:birthPlace", "dbo:placeOfBirth", "wdt:P19"),
            ("foaf:name", "rdfs:label"),
            ("married_to", "spouse", "hasSpouse"),
        ],
        detection_methods=[DetectionMethod.SCHEMA_ALIGNMENT, DetectionMethod.EMBEDDING],
        ground_truth_action=GroundTruthAction.MERGE,
        ground_truth_condition="MERGE unter kanonischem Schema",
        references=[
            ScientificReference(
                authors="Paulheim",
                year=2017,
                title="Knowledge Graph Refinement: A Survey of Approaches and Evaluation Methods",
                venue="Semantic Web Journal",
                relevance="Umfassender Survey inkl. Schema Alignment"
            ),
            ScientificReference(
                authors="Euzenat & Shvaiko",
                year=2013,
                title="Ontology Matching",
                venue="Springer",
                relevance="Standardwerk zu Ontology Matching"
            ),
        ],
        requires_ontology=True,
        difficulty="medium",
    ),

    ConflictType.NUMERICAL: ConflictCategory(
        conflict_type=ConflictType.NUMERICAL,
        name="Numerische Präzision",
        definition="Numerische Werte mit unterschiedlicher Präzision oder "
                   "verschiedenen Einheiten.",
        examples=[
            ("Einstein was born in 1879", "Einstein was born in the 1870s"),
            ("The distance is 100km", "The distance is approximately 100km"),
            ("Population: 1,000,000", "Population: about 1 million"),
        ],
        detection_methods=[DetectionMethod.NUMERICAL, DetectionMethod.PATTERN_MATCHING],
        ground_truth_action=GroundTruthAction.MERGE,
        ground_truth_condition="MERGE zu präziserem Wert wenn Range konsistent; "
                               "CONFLICT wenn numerisch inkonsistent",
        references=[
            ScientificReference(
                authors="Hoffart, Suchanek, Berberich & Weikum",
                year=2013,
                title="YAGO2: A Spatially and Temporally Enhanced Knowledge Base from Wikipedia",
                venue="AIJ",
                relevance="Temporale und numerische Fakten in YAGO"
            ),
        ],
        difficulty="easy",
    ),
}


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def get_all_conflict_types() -> List[ConflictType]:
    """Gibt alle Konflikt-Typen zurück."""
    return list(ConflictType)


def get_category(conflict_type: ConflictType) -> ConflictCategory:
    """Gibt die Kategorie-Definition für einen Konflikt-Typ zurück."""
    return CONFLICT_TAXONOMY[conflict_type]


def get_detection_methods(conflict_type: ConflictType) -> List[DetectionMethod]:
    """Gibt die Erkennungsmethoden für einen Konflikt-Typ zurück."""
    return CONFLICT_TAXONOMY[conflict_type].detection_methods


def get_ground_truth_action(conflict_type: ConflictType) -> GroundTruthAction:
    """Gibt die Ground Truth Aktion für einen Konflikt-Typ zurück."""
    return CONFLICT_TAXONOMY[conflict_type].ground_truth_action


def get_references(conflict_type: ConflictType) -> List[ScientificReference]:
    """Gibt die wissenschaftlichen Referenzen für einen Konflikt-Typ zurück."""
    return CONFLICT_TAXONOMY[conflict_type].references


def get_all_references() -> List[ScientificReference]:
    """Gibt alle wissenschaftlichen Referenzen zurück (dedupliziert)."""
    all_refs = []
    seen = set()
    for category in CONFLICT_TAXONOMY.values():
        for ref in category.references:
            key = (ref.authors, ref.year, ref.title)
            if key not in seen:
                seen.add(key)
                all_refs.append(ref)
    return sorted(all_refs, key=lambda r: r.year)


def get_conflicts_requiring_method(method: DetectionMethod) -> List[ConflictType]:
    """Gibt alle Konflikt-Typen zurück die eine bestimmte Methode benötigen."""
    return [
        ct for ct, cat in CONFLICT_TAXONOMY.items()
        if method in cat.detection_methods
    ]


def get_difficulty_distribution() -> Dict[str, List[ConflictType]]:
    """Gruppiert Konflikt-Typen nach Schwierigkeit."""
    distribution = {"easy": [], "medium": [], "hard": []}
    for ct, cat in CONFLICT_TAXONOMY.items():
        distribution[cat.difficulty].append(ct)
    return distribution


def print_taxonomy_summary():
    """Gibt eine Zusammenfassung der Taxonomie aus."""
    print("=" * 80)
    print("KONFLIKT-TAXONOMIE FÜR KNOWLEDGE GRAPH KONSISTENZPRÜFUNG")
    print("=" * 80)

    for ct, cat in CONFLICT_TAXONOMY.items():
        print(f"\n{ct.value.upper()}: {cat.name}")
        print("-" * 40)
        print(f"Definition: {cat.definition[:80]}...")
        print(f"Erkennung: {', '.join(m.value for m in cat.detection_methods)}")
        print(f"Ground Truth: {cat.ground_truth_action.value}")
        print(f"Schwierigkeit: {cat.difficulty}")
        print(f"Referenzen: {len(cat.references)}")

    print("\n" + "=" * 80)
    print(f"Gesamt: {len(CONFLICT_TAXONOMY)} Konflikt-Kategorien")
    print(f"Referenzen: {len(get_all_references())} Paper")
    print("=" * 80)


# =============================================================================
# MAIN (Test)
# =============================================================================

if __name__ == "__main__":
    print_taxonomy_summary()
