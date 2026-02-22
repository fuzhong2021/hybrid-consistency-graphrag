# src/consistency/rules/temporal_rules.py
"""
Temporale Regeln für Stage 1 Validierung.

Robuste Erkennung von temporalen Widersprüchen ohne LLM:
- Extraktion von Jahreszahlen aus Entity-Namen
- Aufbau eines temporalen Kontexts aus dem Graph
- Validierung neuer Triples gegen temporale Constraints

Beispiele für erkennbare Widersprüche:
- Person gestorben vor Geburt
- Preis gewonnen nach dem Tod
- Heirat nach dem Tod
- Ereignis in der Zukunft
"""

import re
import logging
from typing import Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime

from src.models.entities import Triple, ConflictSet, ConflictType, Entity

logger = logging.getLogger(__name__)

# Aktuelles Jahr für Plausibilitätsprüfungen
CURRENT_YEAR = datetime.now().year

# Minimales plausibles Geburtsjahr (für historische Personen)
MIN_BIRTH_YEAR = 1000

# Maximale Lebensspanne in Jahren
MAX_LIFESPAN = 130


@dataclass
class TemporalContext:
    """Temporaler Kontext einer Entity aus dem Graph."""
    entity_id: str
    entity_name: str

    # Lebensdaten
    birth_year: Optional[int] = None
    death_year: Optional[int] = None

    # Weitere temporale Daten
    events: Dict[str, int] = field(default_factory=dict)  # Relation -> Jahr

    @property
    def is_deceased(self) -> bool:
        """Prüft ob die Entity als verstorben bekannt ist."""
        return self.death_year is not None

    @property
    def lifespan(self) -> Optional[Tuple[int, int]]:
        """Gibt die Lebensspanne zurück wenn beide Daten bekannt sind."""
        if self.birth_year and self.death_year:
            return (self.birth_year, self.death_year)
        return None

    def is_year_in_lifetime(self, year: int) -> Optional[bool]:
        """
        Prüft ob ein Jahr innerhalb der Lebensspanne liegt.

        Returns:
            True: Jahr ist definitiv innerhalb der Lebensspanne
            False: Jahr ist definitiv außerhalb der Lebensspanne
            None: Kann nicht bestimmt werden (fehlende Daten)
        """
        if self.birth_year and year < self.birth_year:
            return False
        if self.death_year and year > self.death_year:
            return False
        if self.birth_year:
            return True  # Nach Geburt, Tod unbekannt oder Jahr davor
        return None


class YearExtractor:
    """Robuste Extraktion von Jahreszahlen aus Text und Entity-Namen."""

    # Regex-Patterns für Jahreszahlen
    PATTERNS = [
        # Explizite Jahreszahlen: "1955", "2020"
        r'\b(1[0-9]{3}|20[0-9]{2})\b',

        # Jahre in Klammern: "(1879-1955)"
        r'\((\d{4})[-–](\d{4})\)',

        # Einzelnes Jahr in Klammern: "(1955)"
        r'\((\d{4})\)',

        # Jahr am Ende: "Nobelpreis 1921"
        r'(\d{4})$',

        # Jahr mit Monat: "March 14, 1879"
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(\d{4})',

        # Deutsches Format: "14. März 1879"
        r'\d{1,2}\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+(\d{4})',

        # ISO-Format: "1879-03-14"
        r'^(\d{4})-\d{2}-\d{2}',
    ]

    @classmethod
    def extract_year(cls, text: str) -> Optional[int]:
        """
        Extrahiert das relevanteste Jahr aus einem Text.

        Priorisierung:
        1. Einzelnes Jahr (wenn nur eines vorhanden)
        2. Letztes Jahr im Text (oft das relevanteste)

        Returns:
            Jahr als int oder None
        """
        if not text:
            return None

        years = cls.extract_all_years(text)

        if not years:
            return None

        # Wenn nur ein Jahr gefunden, nimm es
        if len(years) == 1:
            return years[0]

        # Bei mehreren Jahren: nimm das letzte (oft das relevanteste)
        # z.B. "Nobelpreis Chemie 1960" -> 1960
        return years[-1]

    @classmethod
    def extract_all_years(cls, text: str) -> List[int]:
        """
        Extrahiert alle plausiblen Jahreszahlen aus einem Text.

        Returns:
            Liste von Jahren, sortiert nach Erscheinen im Text
        """
        if not text:
            return []

        years = []
        seen = set()

        for pattern in cls.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Hole alle Gruppen (manche Patterns haben mehrere)
                for group in match.groups():
                    if group and group.isdigit():
                        year = int(group)
                        if cls._is_plausible_year(year) and year not in seen:
                            years.append(year)
                            seen.add(year)

        return years

    @classmethod
    def extract_year_range(cls, text: str) -> Optional[Tuple[int, int]]:
        """
        Extrahiert einen Jahreszeitraum aus Text.

        Erkennt: "1879-1955", "(1879–1955)", "1879 - 1955"

        Returns:
            Tuple (start_year, end_year) oder None
        """
        if not text:
            return None

        # Pattern für Jahresbereiche
        range_patterns = [
            r'(\d{4})\s*[-–—]\s*(\d{4})',  # 1879-1955, 1879 – 1955
            r'\((\d{4})\s*[-–—]\s*(\d{4})\)',  # (1879-1955)
        ]

        for pattern in range_patterns:
            match = re.search(pattern, text)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                if cls._is_plausible_year(start) and cls._is_plausible_year(end):
                    return (start, end)

        return None

    @classmethod
    def _is_plausible_year(cls, year: int) -> bool:
        """Prüft ob eine Zahl ein plausibles Jahr ist."""
        return MIN_BIRTH_YEAR <= year <= CURRENT_YEAR + 10  # +10 für nahe Zukunft


# Temporale Relationstypen und ihre Semantik
TEMPORAL_RELATIONS = {
    # Lebensdaten
    "GEBOREN_AM": {"type": "birth", "sets": "birth_year"},
    "GEBOREN_IN": {"type": "birth", "sets": "birth_year"},
    "BIRTH_DATE": {"type": "birth", "sets": "birth_year"},
    "BORN": {"type": "birth", "sets": "birth_year"},
    "BORN_IN": {"type": "birth", "sets": "birth_year"},

    "GESTORBEN_AM": {"type": "death", "sets": "death_year"},
    "GESTORBEN_IN": {"type": "death", "sets": "death_year"},
    "DEATH_DATE": {"type": "death", "sets": "death_year"},
    "DIED": {"type": "death", "sets": "death_year"},
    "DIED_IN": {"type": "death", "sets": "death_year"},

    # Ereignisse die innerhalb der Lebensspanne sein müssen
    "GEWANN": {"type": "lifetime_event", "requires": "alive"},
    "ERHIELT": {"type": "lifetime_event", "requires": "alive"},
    "WON": {"type": "lifetime_event", "requires": "alive"},
    "AWARDED": {"type": "lifetime_event", "requires": "alive"},
    "RECEIVED": {"type": "lifetime_event", "requires": "alive"},

    "VERHEIRATET_MIT": {"type": "lifetime_event", "requires": "alive"},
    "MARRIED_TO": {"type": "lifetime_event", "requires": "alive"},
    "MARRIED": {"type": "lifetime_event", "requires": "alive"},
    "SPOUSE": {"type": "lifetime_event", "requires": "alive"},

    "GESCHIEDEN_VON": {"type": "lifetime_event", "requires": "alive"},
    "DIVORCED": {"type": "lifetime_event", "requires": "alive"},

    "ARBEITET_BEI": {"type": "lifetime_event", "requires": "alive"},
    "ARBEITETE_BEI": {"type": "lifetime_event", "requires": "alive"},
    "WORKS_AT": {"type": "lifetime_event", "requires": "alive"},
    "WORKED_AT": {"type": "lifetime_event", "requires": "alive"},
    "EMPLOYED_BY": {"type": "lifetime_event", "requires": "alive"},

    "GRUENDETE": {"type": "lifetime_event", "requires": "alive"},
    "FOUNDED": {"type": "lifetime_event", "requires": "alive"},

    "SCHRIEB": {"type": "lifetime_event", "requires": "alive"},
    "WROTE": {"type": "lifetime_event", "requires": "alive"},
    "AUTHORED": {"type": "lifetime_event", "requires": "alive"},

    "SPIELTE_IN": {"type": "lifetime_event", "requires": "alive"},
    "STARRED_IN": {"type": "lifetime_event", "requires": "alive"},
    "APPEARED_IN": {"type": "lifetime_event", "requires": "alive"},

    # Relationen ohne temporale Einschränkung (können posthum sein)
    "VATER_VON": {"type": "family", "requires": None},
    "MUTTER_VON": {"type": "family", "requires": None},
    "KIND_VON": {"type": "family", "requires": None},
    "PARENT_OF": {"type": "family", "requires": None},
    "CHILD_OF": {"type": "family", "requires": None},
}


class TemporalValidator:
    """
    Temporale Validierung für Knowledge Graph Triples.

    Baut temporalen Kontext aus dem Graph auf und validiert
    neue Triples gegen temporale Constraints.
    """

    def __init__(self, temporal_relations: Dict[str, Dict] = None):
        """
        Args:
            temporal_relations: Mapping von Relationstypen zu temporaler Semantik
        """
        self.temporal_relations = temporal_relations or TEMPORAL_RELATIONS
        self._context_cache: Dict[str, TemporalContext] = {}

    def clear_cache(self):
        """Leert den Context-Cache."""
        self._context_cache = {}

    def build_temporal_context(
        self,
        entity: Entity,
        graph_repo: Any
    ) -> TemporalContext:
        """
        Baut den temporalen Kontext für eine Entity aus dem Graph.

        Args:
            entity: Die Entity für die der Kontext gebaut werden soll
            graph_repo: Repository für Graph-Zugriff

        Returns:
            TemporalContext mit allen bekannten temporalen Daten
        """
        # Cache-Lookup
        if entity.id in self._context_cache:
            return self._context_cache[entity.id]

        context = TemporalContext(
            entity_id=entity.id,
            entity_name=entity.name
        )

        if not graph_repo:
            return context

        try:
            # Alle Relationen dieser Entity abrufen
            relations = graph_repo.find_relations(source_id=entity.id)

            for rel in relations:
                rel_type = rel.get("rel_type", "").upper().replace(" ", "_")
                target = rel.get("target", {})
                target_name = target.get("name", "")

                # Temporale Semantik prüfen
                temporal_info = self.temporal_relations.get(rel_type)
                if not temporal_info:
                    continue

                # Jahr aus Ziel-Entity extrahieren
                year = YearExtractor.extract_year(target_name)

                if year:
                    if temporal_info.get("sets") == "birth_year":
                        context.birth_year = year
                        logger.debug(f"  Temporal: {entity.name} geboren {year}")
                    elif temporal_info.get("sets") == "death_year":
                        context.death_year = year
                        logger.debug(f"  Temporal: {entity.name} gestorben {year}")
                    else:
                        context.events[rel_type] = year

        except Exception as e:
            logger.debug(f"Fehler beim Aufbau des temporalen Kontexts: {e}")

        # Cachen
        self._context_cache[entity.id] = context
        return context

    def validate(
        self,
        triple: Triple,
        graph_repo: Any = None
    ) -> Optional[ConflictSet]:
        """
        Validiert ein Triple gegen temporale Constraints.

        Prüfungen:
        1. Plausibilität: Jahr in sinnvollem Bereich
        2. Lebensspanne: Geburt < Tod
        3. Lifetime-Events: Ereignis während Lebensspanne
        4. Konsistenz: Keine widersprüchlichen Daten

        Args:
            triple: Das zu validierende Triple
            graph_repo: Repository für Graph-Zugriff

        Returns:
            ConflictSet bei temporalem Widerspruch, sonst None
        """
        predicate = triple.predicate.upper().replace(" ", "_")
        temporal_info = self.temporal_relations.get(predicate)

        # Jahr aus Object extrahieren
        object_year = YearExtractor.extract_year(triple.object.name)

        # === 1. Plausibilitätsprüfung ===
        if object_year:
            conflict = self._check_year_plausibility(object_year, triple)
            if conflict:
                return conflict

        # Ohne Graph keine weiteren Prüfungen möglich
        if not graph_repo:
            return None

        # Temporalen Kontext aufbauen
        subject_context = self.build_temporal_context(triple.subject, graph_repo)

        # === 2. Lebensspanne-Konsistenz ===
        if temporal_info:
            if temporal_info.get("sets") == "birth_year" and object_year:
                conflict = self._check_birth_consistency(
                    object_year, subject_context, triple
                )
                if conflict:
                    return conflict

            if temporal_info.get("sets") == "death_year" and object_year:
                conflict = self._check_death_consistency(
                    object_year, subject_context, triple
                )
                if conflict:
                    return conflict

        # === 3. Lifetime-Event-Prüfung ===
        if temporal_info and temporal_info.get("requires") == "alive":
            conflict = self._check_lifetime_event(
                object_year, subject_context, triple
            )
            if conflict:
                return conflict

        return None

    def _check_year_plausibility(
        self,
        year: int,
        triple: Triple
    ) -> Optional[ConflictSet]:
        """Prüft ob ein Jahr plausibel ist."""

        # Jahr in der Zukunft (mehr als 1 Jahr)
        if year > CURRENT_YEAR + 1:
            return ConflictSet(
                conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                description=f"Temporaler Fehler: Jahr {year} liegt in der Zukunft "
                           f"(aktuell: {CURRENT_YEAR})",
                severity=0.9
            )

        # Jahr zu weit in der Vergangenheit für bestimmte Relationen
        predicate = triple.predicate.upper()
        if predicate in ["GEWANN", "WON", "ERHIELT", "AWARDED"]:
            # Nobelpreis gibt es seit 1901
            if year < 1800:
                return ConflictSet(
                    conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                    description=f"Temporaler Fehler: Preis/Auszeichnung im Jahr {year} "
                               f"ist unplausibel (vor 1800)",
                    severity=0.8
                )

        return None

    def _check_birth_consistency(
        self,
        birth_year: int,
        context: TemporalContext,
        triple: Triple
    ) -> Optional[ConflictSet]:
        """Prüft Konsistenz eines neuen Geburtsdatums."""

        # Bereits bekanntes Geburtsjahr
        if context.birth_year and context.birth_year != birth_year:
            return ConflictSet(
                conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                description=f"Widersprüchliche Geburtsjahre: {triple.subject.name} "
                           f"bereits geboren {context.birth_year}, neues Triple sagt {birth_year}",
                severity=0.9
            )

        # Geburt nach Tod
        if context.death_year and birth_year > context.death_year:
            return ConflictSet(
                conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                description=f"Temporaler Widerspruch: {triple.subject.name} "
                           f"kann nicht {birth_year} geboren sein, "
                           f"da bereits als gestorben {context.death_year} bekannt",
                severity=0.95
            )

        # Unrealistische Lebensspanne
        if context.death_year:
            lifespan = context.death_year - birth_year
            if lifespan > MAX_LIFESPAN:
                return ConflictSet(
                    conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                    description=f"Unplausible Lebensspanne: {triple.subject.name} "
                               f"geboren {birth_year}, gestorben {context.death_year} "
                               f"= {lifespan} Jahre (max: {MAX_LIFESPAN})",
                    severity=0.8
                )
            if lifespan < 0:
                return ConflictSet(
                    conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                    description=f"Temporaler Widerspruch: {triple.subject.name} "
                               f"kann nicht {birth_year} geboren sein und "
                               f"{context.death_year} gestorben (negative Lebensspanne)",
                    severity=0.95
                )

        return None

    def _check_death_consistency(
        self,
        death_year: int,
        context: TemporalContext,
        triple: Triple
    ) -> Optional[ConflictSet]:
        """Prüft Konsistenz eines neuen Todesdatums."""

        # Bereits bekanntes Todesjahr
        if context.death_year and context.death_year != death_year:
            return ConflictSet(
                conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                description=f"Widersprüchliche Todesjahre: {triple.subject.name} "
                           f"bereits gestorben {context.death_year}, neues Triple sagt {death_year}",
                severity=0.9
            )

        # Tod vor Geburt
        if context.birth_year and death_year < context.birth_year:
            return ConflictSet(
                conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                description=f"Temporaler Widerspruch: {triple.subject.name} "
                           f"kann nicht {death_year} gestorben sein, "
                           f"da erst {context.birth_year} geboren",
                severity=0.95
            )

        # Unrealistische Lebensspanne
        if context.birth_year:
            lifespan = death_year - context.birth_year
            if lifespan > MAX_LIFESPAN:
                return ConflictSet(
                    conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                    description=f"Unplausible Lebensspanne: {triple.subject.name} "
                               f"geboren {context.birth_year}, gestorben {death_year} "
                               f"= {lifespan} Jahre (max: {MAX_LIFESPAN})",
                    severity=0.8
                )

        return None

    def _check_lifetime_event(
        self,
        event_year: Optional[int],
        context: TemporalContext,
        triple: Triple
    ) -> Optional[ConflictSet]:
        """
        Prüft ob ein Ereignis innerhalb der Lebensspanne liegt.

        Relevant für: GEWANN, VERHEIRATET_MIT, ARBEITET_BEI, etc.
        """
        if not event_year:
            # Ohne Jahr im Object: Prüfe ob Entity als verstorben bekannt ist
            # und das Prädikat impliziert Aktivität
            if context.is_deceased:
                predicate = triple.predicate.upper()
                # Präsens-Formen implizieren aktuelle Aktivität
                present_tense = ["ARBEITET_BEI", "WORKS_AT", "VERHEIRATET_MIT",
                                "MARRIED_TO", "IST_LEBENDIG", "IS_ALIVE"]
                if predicate in present_tense:
                    return ConflictSet(
                        conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                        description=f"Zustandswiderspruch: {triple.subject.name} "
                                   f"ist als verstorben ({context.death_year}) bekannt, "
                                   f"kann daher nicht '{predicate}' sein",
                        severity=0.9
                    )
            return None

        # Ereignis vor Geburt
        if context.birth_year and event_year < context.birth_year:
            return ConflictSet(
                conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                description=f"Temporaler Widerspruch: {triple.subject.name} "
                           f"kann {event_year} kein Ereignis haben, "
                           f"da erst {context.birth_year} geboren",
                severity=0.9
            )

        # Ereignis nach Tod
        if context.death_year and event_year > context.death_year:
            predicate = triple.predicate.upper()
            return ConflictSet(
                conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                description=f"Temporaler Widerspruch: {triple.subject.name} "
                           f"kann {event_year} '{predicate}' nicht ausführen, "
                           f"da bereits {context.death_year} gestorben",
                severity=0.95
            )

        return None


# Singleton-Instanz für einfache Nutzung
_temporal_validator: Optional[TemporalValidator] = None


def get_temporal_validator() -> TemporalValidator:
    """Gibt die Singleton-Instanz des TemporalValidators zurück."""
    global _temporal_validator
    if _temporal_validator is None:
        _temporal_validator = TemporalValidator()
    return _temporal_validator


def validate_temporal(
    triple: Triple,
    graph_repo: Any = None
) -> Optional[ConflictSet]:
    """
    Convenience-Funktion für temporale Validierung.

    Args:
        triple: Das zu validierende Triple
        graph_repo: Repository für Graph-Zugriff

    Returns:
        ConflictSet bei temporalem Widerspruch, sonst None
    """
    return get_temporal_validator().validate(triple, graph_repo)
