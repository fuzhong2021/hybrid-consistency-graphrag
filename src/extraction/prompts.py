# src/extraction/prompts.py
"""
Prompt-Templates für LLM-basierte Wissensextraktion.

Basiert auf aktuellen Forschungsansätzen:
- Microsoft GraphRAG (2024): Strukturierte Entity/Relation Extraktion
- iText2KG (Lairgi et al., 2024): Schema-guided Extraction
- Zero-shot und Few-shot Prompting Best Practices
"""

from typing import List, Dict, Any


# =============================================================================
# System-Prompts
# =============================================================================

SYSTEM_PROMPT_EXTRACTION = """Du bist ein Experte für Wissensextraktion aus Text.
Deine Aufgabe ist es, strukturiertes Wissen in Form von Entitäten und Relationen zu extrahieren.

WICHTIG:
- Extrahiere NUR explizit im Text genannte Fakten
- Keine Inferenzen oder Annahmen
- Jede Entität braucht einen eindeutigen Namen und Typ
- Jede Relation verbindet genau zwei Entitäten
- Gib Konfidenzwerte basierend auf der Klarheit der Information"""


# =============================================================================
# Entity Extraction (Schritt 1 - nach Microsoft GraphRAG)
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """Extrahiere alle Entitäten aus dem folgenden Text.

## Erlaubte Entitätstypen
{entity_types}

## Text
{text}

## Aufgabe
Identifiziere alle benannten Entitäten und klassifiziere sie.

## Ausgabeformat (JSON)
{{
    "entities": [
        {{
            "name": "Vollständiger Name der Entität",
            "type": "Einer der erlaubten Typen",
            "description": "Kurze Beschreibung basierend auf Kontext",
            "aliases": ["Alternative Namen/Schreibweisen"],
            "confidence": 0.0-1.0
        }}
    ]
}}

Antworte NUR mit validem JSON."""


# =============================================================================
# Relation Extraction (Schritt 2 - nach Microsoft GraphRAG)
# =============================================================================

RELATION_EXTRACTION_PROMPT = """Extrahiere alle Relationen zwischen den gegebenen Entitäten.

## Erlaubte Relationstypen
{relation_types}

## Entitäten im Text
{entities}

## Text
{text}

## Aufgabe
Identifiziere alle Beziehungen zwischen den Entitäten.

## Ausgabeformat (JSON)
{{
    "relations": [
        {{
            "source": "Name der Quell-Entität",
            "relation": "Einer der erlaubten Relationstypen",
            "target": "Name der Ziel-Entität",
            "evidence": "Textstelle, die diese Relation belegt",
            "confidence": 0.0-1.0
        }}
    ]
}}

Antworte NUR mit validem JSON."""


# =============================================================================
# Combined Extraction (Single-Pass - für Effizienz)
# =============================================================================

COMBINED_EXTRACTION_PROMPT = """Extrahiere Wissen aus dem folgenden Text als Knowledge Graph.

## Schema

### Erlaubte Entitätstypen
{entity_types}

### Erlaubte Relationstypen
{relation_types}

## Text
{text}

## Aufgabe
1. Identifiziere alle benannten Entitäten
2. Extrahiere Relationen zwischen Entitäten
3. Gib Konfidenzwerte basierend auf Explizitheit der Information

## Regeln
- NUR explizit genannte Fakten extrahieren
- Keine Inferenzen oder Weltwissen hinzufügen
- Bei Unsicherheit: niedrigere Konfidenz, nicht weglassen
- Temporale Informationen (Datum, Zeitraum) wenn vorhanden erfassen

## Ausgabeformat (JSON)
{{
    "entities": [
        {{
            "name": "Name",
            "type": "Typ",
            "description": "Beschreibung aus Kontext",
            "temporal": {{
                "valid_from": "YYYY-MM-DD oder null",
                "valid_until": "YYYY-MM-DD oder null"
            }},
            "confidence": 0.0-1.0
        }}
    ],
    "relations": [
        {{
            "source": "Quell-Entität",
            "relation": "Relationstyp",
            "target": "Ziel-Entität",
            "evidence": "Belegstelle",
            "temporal": {{
                "valid_from": "YYYY-MM-DD oder null",
                "valid_until": "YYYY-MM-DD oder null"
            }},
            "confidence": 0.0-1.0
        }}
    ]
}}

Antworte NUR mit validem JSON."""


# =============================================================================
# Few-Shot Examples (nach iText2KG-Ansatz)
# =============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "text": "Albert Einstein wurde am 14. März 1879 in Ulm geboren. Er arbeitete später am Patentamt in Bern und entwickelte die Relativitätstheorie.",
        "output": {
            "entities": [
                {
                    "name": "Albert Einstein",
                    "type": "Person",
                    "description": "Physiker, Entwickler der Relativitätstheorie",
                    "temporal": {"valid_from": "1879-03-14", "valid_until": None},
                    "confidence": 0.95
                },
                {
                    "name": "Ulm",
                    "type": "Ort",
                    "description": "Geburtsort von Albert Einstein",
                    "temporal": None,
                    "confidence": 0.95
                },
                {
                    "name": "Patentamt Bern",
                    "type": "Organisation",
                    "description": "Arbeitgeber von Einstein",
                    "temporal": None,
                    "confidence": 0.90
                },
                {
                    "name": "Relativitätstheorie",
                    "type": "Konzept",
                    "description": "Von Einstein entwickelte physikalische Theorie",
                    "temporal": None,
                    "confidence": 0.95
                }
            ],
            "relations": [
                {
                    "source": "Albert Einstein",
                    "relation": "GEBOREN_IN",
                    "target": "Ulm",
                    "evidence": "wurde am 14. März 1879 in Ulm geboren",
                    "temporal": {"valid_from": "1879-03-14", "valid_until": "1879-03-14"},
                    "confidence": 0.95
                },
                {
                    "source": "Albert Einstein",
                    "relation": "ARBEITET_BEI",
                    "target": "Patentamt Bern",
                    "evidence": "arbeitete später am Patentamt in Bern",
                    "temporal": None,
                    "confidence": 0.85
                },
                {
                    "source": "Albert Einstein",
                    "relation": "ENTWICKELTE",
                    "target": "Relativitätstheorie",
                    "evidence": "entwickelte die Relativitätstheorie",
                    "temporal": None,
                    "confidence": 0.90
                }
            ]
        }
    },
    {
        "text": "Marie Curie, geborene Skłodowska, war eine polnisch-französische Physikerin. Sie erhielt 1903 zusammen mit ihrem Ehemann Pierre Curie den Nobelpreis für Physik.",
        "output": {
            "entities": [
                {
                    "name": "Marie Curie",
                    "type": "Person",
                    "description": "Polnisch-französische Physikerin, Nobelpreisträgerin",
                    "temporal": None,
                    "confidence": 0.95
                },
                {
                    "name": "Pierre Curie",
                    "type": "Person",
                    "description": "Ehemann von Marie Curie, Nobelpreisträger",
                    "temporal": None,
                    "confidence": 0.95
                },
                {
                    "name": "Nobelpreis für Physik",
                    "type": "Ereignis",
                    "description": "Auszeichnung 1903",
                    "temporal": {"valid_from": "1903-01-01", "valid_until": "1903-12-31"},
                    "confidence": 0.95
                }
            ],
            "relations": [
                {
                    "source": "Marie Curie",
                    "relation": "VERHEIRATET_MIT",
                    "target": "Pierre Curie",
                    "evidence": "ihrem Ehemann Pierre Curie",
                    "temporal": None,
                    "confidence": 0.95
                },
                {
                    "source": "Marie Curie",
                    "relation": "ERHIELT",
                    "target": "Nobelpreis für Physik",
                    "evidence": "erhielt 1903 zusammen mit ihrem Ehemann Pierre Curie den Nobelpreis",
                    "temporal": {"valid_from": "1903-01-01", "valid_until": "1903-12-31"},
                    "confidence": 0.95
                },
                {
                    "source": "Pierre Curie",
                    "relation": "ERHIELT",
                    "target": "Nobelpreis für Physik",
                    "evidence": "erhielt 1903 zusammen mit ihrem Ehemann Pierre Curie den Nobelpreis",
                    "temporal": {"valid_from": "1903-01-01", "valid_until": "1903-12-31"},
                    "confidence": 0.90
                }
            ]
        }
    }
]


def format_few_shot_prompt(examples: List[Dict] = None) -> str:
    """Formatiert Few-Shot Beispiele für den Prompt."""
    if examples is None:
        examples = FEW_SHOT_EXAMPLES

    formatted = "## Beispiele\n\n"

    for i, ex in enumerate(examples, 1):
        formatted += f"### Beispiel {i}\n"
        formatted += f"**Text:** {ex['text']}\n\n"
        formatted += f"**Ausgabe:**\n```json\n"

        import json
        formatted += json.dumps(ex['output'], indent=2, ensure_ascii=False)
        formatted += "\n```\n\n"

    return formatted


# =============================================================================
# Coreference Resolution (für Entity Linking)
# =============================================================================

COREFERENCE_PROMPT = """Analysiere den Text und identifiziere, welche Erwähnungen sich auf dieselbe Entität beziehen.

## Text
{text}

## Bereits extrahierte Entitäten
{entities}

## Aufgabe
Finde Pronomen und alternative Bezeichnungen, die sich auf die Entitäten beziehen.

## Ausgabeformat (JSON)
{{
    "coreferences": [
        {{
            "entity": "Kanonischer Name",
            "mentions": ["Erwähnung 1", "Erwähnung 2", "er", "sie", ...]
        }}
    ]
}}

Antworte NUR mit validem JSON."""


# =============================================================================
# Temporal Extraction (für Graphiti bi-temporal)
# =============================================================================

TEMPORAL_EXTRACTION_PROMPT = """Extrahiere temporale Informationen aus dem Text.

## Text
{text}

## Aufgabe
Identifiziere:
1. Explizite Datumsangaben
2. Relative Zeitangaben ("später", "danach", "während")
3. Zeiträume ("von X bis Y", "seit X")

## Ausgabeformat (JSON)
{{
    "temporal_expressions": [
        {{
            "text": "Originaltext der Zeitangabe",
            "type": "date|period|relative",
            "normalized": {{
                "start": "YYYY-MM-DD oder null",
                "end": "YYYY-MM-DD oder null"
            }},
            "related_entities": ["Entität 1", "Entität 2"]
        }}
    ]
}}

Antworte NUR mit validem JSON."""


# =============================================================================
# Fact Verification (Zusätzlich für Konsistenzprüfung)
# =============================================================================

FACT_VERIFICATION_PROMPT = """Prüfe die Konsistenz der extrahierten Fakten.

## Extrahierte Tripel
{triples}

## Originaltext
{text}

## Aufgabe
Für jedes Tripel:
1. Ist es direkt im Text belegt?
2. Widerspricht es anderen extrahierten Fakten?
3. Wie hoch ist die Konfidenz?

## Ausgabeformat (JSON)
{{
    "verified_triples": [
        {{
            "triple": {{"source": "...", "relation": "...", "target": "..."}},
            "is_supported": true/false,
            "evidence": "Belegstelle oder null",
            "conflicts_with": ["Index anderer Tripel"] oder [],
            "confidence": 0.0-1.0,
            "reasoning": "Begründung"
        }}
    ]
}}

Antworte NUR mit validem JSON."""


# =============================================================================
# Prompt Builder
# =============================================================================

class PromptBuilder:
    """Baut Prompts für verschiedene Extraktionsaufgaben."""

    def __init__(
        self,
        entity_types: List[str],
        relation_types: List[str],
        use_few_shot: bool = True,
        language: str = "de"
    ):
        self.entity_types = entity_types
        self.relation_types = relation_types
        self.use_few_shot = use_few_shot
        self.language = language

    def build_extraction_prompt(
        self,
        text: str,
        mode: str = "combined"
    ) -> List[Dict[str, str]]:
        """
        Baut einen Extraktions-Prompt.

        Args:
            text: Zu analysierender Text
            mode: "combined", "entities", "relations"

        Returns:
            Liste von Messages für Chat-API
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION}
        ]

        # Few-shot Beispiele hinzufügen
        if self.use_few_shot:
            few_shot = format_few_shot_prompt(FEW_SHOT_EXAMPLES[:1])  # 1 Beispiel für Effizienz
            messages.append({
                "role": "user",
                "content": few_shot + "\nNun extrahiere aus folgendem Text:"
            })

        # Haupt-Prompt
        if mode == "combined":
            prompt = COMBINED_EXTRACTION_PROMPT.format(
                entity_types=self._format_types(self.entity_types),
                relation_types=self._format_types(self.relation_types),
                text=text
            )
        elif mode == "entities":
            prompt = ENTITY_EXTRACTION_PROMPT.format(
                entity_types=self._format_types(self.entity_types),
                text=text
            )
        elif mode == "relations":
            # Für Relations brauchen wir vorher extrahierte Entities
            raise ValueError("Für 'relations' mode bitte build_relation_prompt verwenden")
        else:
            raise ValueError(f"Unbekannter Mode: {mode}")

        messages.append({"role": "user", "content": prompt})

        return messages

    def build_relation_prompt(
        self,
        text: str,
        entities: List[Dict]
    ) -> List[Dict[str, str]]:
        """Baut einen Prompt für Relationsextraktion."""
        import json

        entities_str = json.dumps(
            [{"name": e["name"], "type": e["type"]} for e in entities],
            ensure_ascii=False,
            indent=2
        )

        prompt = RELATION_EXTRACTION_PROMPT.format(
            relation_types=self._format_types(self.relation_types),
            entities=entities_str,
            text=text
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
            {"role": "user", "content": prompt}
        ]

    def build_coreference_prompt(
        self,
        text: str,
        entities: List[Dict]
    ) -> List[Dict[str, str]]:
        """Baut einen Prompt für Coreference Resolution."""
        import json

        entities_str = json.dumps(
            [e["name"] for e in entities],
            ensure_ascii=False
        )

        prompt = COREFERENCE_PROMPT.format(
            text=text,
            entities=entities_str
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
            {"role": "user", "content": prompt}
        ]

    def build_verification_prompt(
        self,
        text: str,
        triples: List[Dict]
    ) -> List[Dict[str, str]]:
        """Baut einen Prompt für Faktenverifikation."""
        import json

        triples_str = json.dumps(triples, ensure_ascii=False, indent=2)

        prompt = FACT_VERIFICATION_PROMPT.format(
            triples=triples_str,
            text=text
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
            {"role": "user", "content": prompt}
        ]

    def _format_types(self, types: List[str]) -> str:
        """Formatiert Typen als Bullet-Liste."""
        return "\n".join(f"- {t}" for t in types)
