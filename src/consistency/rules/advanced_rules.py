# src/consistency/rules/advanced_rules.py
"""
Erweiterte regelbasierte Validierung für Stage 1.

Implementiert:
1. Inverse Relations - Konsistenz zwischen inversen Relationen
2. Transitive Closure - Transitiver Abschluss für hierarchische Relationen
3. Disjunkte Typen - Entities können nicht mehrere disjunkte Typen haben

Wissenschaftliche Grundlage:
- OWL 2 Constraints (inverseOf, TransitiveProperty, disjointWith)
- SHACL Shapes für Knowledge Graph Validation
"""

import logging
from typing import Optional, Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from src.models.entities import Triple, ConflictSet, ConflictType, Entity

logger = logging.getLogger(__name__)


# =============================================================================
# KONFIGURATION
# =============================================================================

@dataclass
class AdvancedRulesConfig:
    """Konfiguration für erweiterte Regeln."""

    # Inverse Relationen: Wenn A rel B, dann B inverse(rel) A
    inverse_relations: Dict[str, str] = field(default_factory=lambda: {
        # Familie
        "VATER_VON": "KIND_VON",
        "MUTTER_VON": "KIND_VON",
        "KIND_VON": "ELTERNTEIL_VON",
        "PARENT_OF": "CHILD_OF",
        "CHILD_OF": "PARENT_OF",
        "FATHER_OF": "CHILD_OF",
        "MOTHER_OF": "CHILD_OF",

        # Symmetrische Relationen (self-inverse)
        "VERHEIRATET_MIT": "VERHEIRATET_MIT",
        "MARRIED_TO": "MARRIED_TO",
        "GESCHWISTER_VON": "GESCHWISTER_VON",
        "SIBLING_OF": "SIBLING_OF",
        "KOLLEGE_VON": "KOLLEGE_VON",
        "COLLEAGUE_OF": "COLLEAGUE_OF",
        "KENNT": "KENNT",
        "KNOWS": "KNOWS",

        # Organisation
        "ARBEITET_BEI": "HAT_MITARBEITER",
        "WORKS_AT": "EMPLOYS",
        "LEITET": "GELEITET_VON",
        "LEADS": "LED_BY",
        "MITGLIED_VON": "HAT_MITGLIED",
        "MEMBER_OF": "HAS_MEMBER",

        # Ort
        "GEBOREN_IN": "GEBURTSORT_VON",
        "BORN_IN": "BIRTHPLACE_OF",
        "GESTORBEN_IN": "STERBEORT_VON",
        "DIED_IN": "DEATHPLACE_OF",
        "WOHNT_IN": "HAT_EINWOHNER",
        "LIVES_IN": "HAS_RESIDENT",

        # Geographie
        "LIEGT_IN": "ENTHÄLT",
        "LOCATED_IN": "CONTAINS",
        "TEIL_VON": "HAT_TEIL",
        "PART_OF": "HAS_PART",
        "HAUPTSTADT_VON": "HAT_HAUPTSTADT",
        "CAPITAL_OF": "HAS_CAPITAL",
    })

    # Transitive Relationen: Wenn A rel B und B rel C, dann A rel C
    transitive_relations: Set[str] = field(default_factory=lambda: {
        "LIEGT_IN", "LOCATED_IN",
        "TEIL_VON", "PART_OF",
        "ENTHÄLT", "CONTAINS",
        "VORGESETZTER_VON", "SUPERVISOR_OF",
        "VORFAHRE_VON", "ANCESTOR_OF",
        "NACHFAHRE_VON", "DESCENDANT_OF",
        "UNTERKLASSE_VON", "SUBCLASS_OF",
    })

    # Disjunkte Typen: Entities können nicht beide Typen gleichzeitig haben
    disjoint_types: List[Set[str]] = field(default_factory=lambda: [
        {"Person", "Organisation", "Ort", "Ereignis"},  # Grundtypen
        {"Stadt", "Land", "Kontinent"},  # Geographische Hierarchie
        {"Firma", "Universität", "Regierung"},  # Organisationstypen
        {"Lebend", "Tot"},  # Zustände
        {"Männlich", "Weiblich"},  # Geschlecht
    ])

    # Maximale Tiefe für transitive Suche
    max_transitive_depth: int = 5

    # Automatische Inferenz aktivieren
    enable_inference: bool = True


# =============================================================================
# INVERSE RELATIONS
# =============================================================================

class InverseRelationChecker:
    """
    Prüft Konsistenz zwischen inversen Relationen.

    Beispiel:
    - Wenn "Einstein VATER_VON Hans" existiert
    - Und neues Triple "Hans KIND_VON Marie" kommt
    - → Konflikt: Hans kann nicht Kind von Einstein UND Marie sein
      (außer Marie = Einsteins Frau)
    """

    def __init__(self, config: AdvancedRulesConfig):
        self.config = config
        # Bidirektionales Mapping aufbauen
        self.inverse_map: Dict[str, str] = {}
        for rel, inv in config.inverse_relations.items():
            self.inverse_map[rel.upper()] = inv.upper()
            if rel != inv:  # Nicht für symmetrische
                self.inverse_map[inv.upper()] = rel.upper()

    def check(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> Optional[ConflictSet]:
        """
        Prüft ob das Triple mit inversen Relationen konsistent ist.

        Returns:
            ConflictSet bei Inkonsistenz, sonst None
        """
        if not graph_repo:
            return None

        predicate = triple.predicate.upper().replace(" ", "_")
        inverse_pred = self.inverse_map.get(predicate)

        if not inverse_pred:
            return None

        # Prüfe ob die inverse Relation bereits existiert (in umgekehrter Richtung)
        # Das wäre eigentlich OK - es bestätigt die Relation
        # Aber prüfe auf KONFLIKTE

        # Fall 1: Symmetrische Relation - prüfe ob sie schon existiert
        if predicate == inverse_pred:
            existing = graph_repo.find_relations(source_id=triple.object.id)
            for rel in existing:
                rel_type = rel.get("rel_type", "").upper().replace(" ", "_")
                target = rel.get("target", {})
                target_id = target.get("id")

                if rel_type == predicate and target_id == triple.subject.id:
                    # Relation existiert schon in beide Richtungen - Duplikat
                    return ConflictSet(
                        conflict_type=ConflictType.DUPLICATE_RELATION,
                        description=f"Symmetrische Relation existiert bereits: "
                                   f"'{triple.object.name}' --[{predicate}]--> '{triple.subject.name}'",
                        severity=0.6
                    )

        # Fall 2: Prüfe auf widersprüchliche inverse Relationen
        # z.B. A VATER_VON B, aber B hat schon einen anderen VATER
        conflict = self._check_inverse_conflict(triple, inverse_pred, graph_repo)
        if conflict:
            return conflict

        return None

    def _check_inverse_conflict(
        self,
        triple: Triple,
        inverse_pred: str,
        graph_repo: Any
    ) -> Optional[ConflictSet]:
        """
        Prüft ob die inverse Relation einen Konflikt erzeugt.
        """
        # Hole Relationen VOM Object (da inverse Relation umgekehrt ist)
        existing = graph_repo.find_relations(source_id=triple.object.id)

        # Spezialfall: Funktionale inverse Relationen (max 1)
        # z.B. KIND_VON kann nur einen biologischen Vater/Mutter haben
        functional_inverses = {
            "KIND_VON": {"VATER_VON", "MUTTER_VON"},
            "CHILD_OF": {"FATHER_OF", "MOTHER_OF"},
        }

        predicate = triple.predicate.upper().replace(" ", "_")

        if predicate in functional_inverses:
            # Prüfe ob Object schon einen Vater/Mutter hat
            expected_types = functional_inverses[predicate]
            for rel in existing:
                rel_type = rel.get("rel_type", "").upper().replace(" ", "_")
                if rel_type in expected_types:
                    target = rel.get("target", {})
                    existing_parent = target.get("name", "?")

                    # Konflikt nur wenn es ein ANDERER Elternteil ist
                    if target.get("id") != triple.subject.id:
                        return ConflictSet(
                            conflict_type=ConflictType.CONTRADICTORY_RELATION,
                            description=f"Inverse Relation Konflikt: '{triple.object.name}' "
                                       f"hat bereits '{rel_type}' → '{existing_parent}', "
                                       f"neues Triple würde zweiten implizieren",
                            severity=0.7
                        )

        return None

    def infer_inverse(
        self,
        triple: Triple
    ) -> Optional[Triple]:
        """
        Inferiert die inverse Relation für ein Triple.

        Returns:
            Neues Triple mit inverser Relation oder None
        """
        if not self.config.enable_inference:
            return None

        predicate = triple.predicate.upper().replace(" ", "_")
        inverse_pred = self.inverse_map.get(predicate)

        if not inverse_pred or predicate == inverse_pred:
            return None

        # Erstelle inverses Triple
        return Triple(
            subject=triple.object,
            predicate=inverse_pred,
            object=triple.subject,
            source_text=f"Inferiert aus: {triple.source_text}",
            source_document_id=triple.source_document_id,
            extraction_confidence=triple.extraction_confidence * 0.95,  # Leicht reduziert
        )


# =============================================================================
# TRANSITIVE CLOSURE
# =============================================================================

class TransitiveClosureChecker:
    """
    Prüft und inferiert transitive Relationen.

    Beispiel:
    - Berlin LIEGT_IN Deutschland
    - Deutschland LIEGT_IN Europa
    → Berlin LIEGT_IN Europa (inferiert)

    Konflikt-Erkennung:
    - Berlin LIEGT_IN Deutschland
    - Berlin LIEGT_IN Asien
    → Konflikt (Deutschland nicht in Asien)
    """

    def __init__(self, config: AdvancedRulesConfig):
        self.config = config
        self.transitive_relations = {r.upper() for r in config.transitive_relations}

    def check(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> Optional[ConflictSet]:
        """
        Prüft ob das Triple transitiv konsistent ist.
        """
        if not graph_repo:
            return None

        predicate = triple.predicate.upper().replace(" ", "_")

        if predicate not in self.transitive_relations:
            return None

        # Prüfe auf Zyklen (A LIEGT_IN B, B LIEGT_IN A)
        cycle = self._check_transitive_cycle(triple, predicate, graph_repo)
        if cycle:
            return cycle

        # Prüfe auf Inkonsistenz in der Hierarchie
        inconsistency = self._check_hierarchy_inconsistency(triple, predicate, graph_repo)
        if inconsistency:
            return inconsistency

        return None

    def _check_transitive_cycle(
        self,
        triple: Triple,
        predicate: str,
        graph_repo: Any
    ) -> Optional[ConflictSet]:
        """
        Prüft ob das Triple einen transitiven Zyklus erzeugen würde.
        """
        # BFS: Starte beim Object, prüfe ob wir zum Subject zurückkommen
        visited = set()
        queue = [(triple.object.id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if depth > self.config.max_transitive_depth:
                continue

            if current_id in visited:
                continue
            visited.add(current_id)

            # Hole ausgehende Relationen
            existing = graph_repo.find_relations(source_id=current_id)
            for rel in existing:
                rel_type = rel.get("rel_type", "").upper().replace(" ", "_")
                if rel_type != predicate:
                    continue

                target = rel.get("target", {})
                target_id = target.get("id")

                if target_id == triple.subject.id:
                    return ConflictSet(
                        conflict_type=ConflictType.CONTRADICTORY_RELATION,
                        description=f"Transitiver Zyklus: '{triple.subject.name}' --[{predicate}]--> "
                                   f"'{triple.object.name}' würde Zyklus erzeugen (Tiefe: {depth + 1})",
                        severity=0.9
                    )

                if target_id not in visited:
                    queue.append((target_id, depth + 1))

        return None

    def _check_hierarchy_inconsistency(
        self,
        triple: Triple,
        predicate: str,
        graph_repo: Any
    ) -> Optional[ConflictSet]:
        """
        Prüft auf hierarchische Inkonsistenz.

        z.B. Berlin LIEGT_IN Deutschland UND Berlin LIEGT_IN Frankreich
        → Konflikt (außer Deutschland = Frankreich oder eins enthält das andere)
        """
        # Hole existierende Relationen des Subjects mit gleichem Prädikat
        existing = graph_repo.find_relations(source_id=triple.subject.id)

        existing_targets = []
        for rel in existing:
            rel_type = rel.get("rel_type", "").upper().replace(" ", "_")
            if rel_type == predicate:
                target = rel.get("target", {})
                existing_targets.append((target.get("id"), target.get("name", "?")))

        if not existing_targets:
            return None

        # Für LIEGT_IN, TEIL_VON: Mehrere sind OK wenn sie hierarchisch sind
        # z.B. Berlin LIEGT_IN Deutschland UND Berlin LIEGT_IN Europa
        # Aber: Berlin LIEGT_IN Deutschland UND Berlin LIEGT_IN Frankreich → Konflikt

        # Prüfe ob das neue Object eines der existierenden enthält (oder umgekehrt)
        for existing_id, existing_name in existing_targets:
            if existing_id == triple.object.id:
                continue  # Gleiche Entity - Duplikat, kein Hierarchie-Konflikt

            # Prüfe ob Object das existierende Target enthält
            if self._is_ancestor(triple.object.id, existing_id, predicate, graph_repo):
                continue  # OK: Neues Object ist Vorfahre des existierenden

            # Prüfe ob existierendes Target das Object enthält
            if self._is_ancestor(existing_id, triple.object.id, predicate, graph_repo):
                continue  # OK: Existierendes ist Vorfahre des neuen

            # Keine hierarchische Beziehung → potentieller Konflikt
            # Aber nur warnen, nicht hart ablehnen (könnte auch korrekt sein)
            return ConflictSet(
                conflict_type=ConflictType.CONTRADICTORY_RELATION,
                description=f"Potentieller Hierarchie-Konflikt: '{triple.subject.name}' hat bereits "
                           f"'{predicate}' → '{existing_name}', "
                           f"neues Triple fügt '{triple.object.name}' hinzu (nicht hierarchisch verbunden)",
                severity=0.5  # Niedrig - nur Warnung
            )

        return None

    def _is_ancestor(
        self,
        ancestor_id: str,
        descendant_id: str,
        predicate: str,
        graph_repo: Any
    ) -> bool:
        """
        Prüft ob ancestor_id ein Vorfahre von descendant_id ist.
        """
        visited = set()
        queue = [descendant_id]
        depth = 0

        while queue and depth < self.config.max_transitive_depth:
            next_queue = []
            for current_id in queue:
                if current_id in visited:
                    continue
                visited.add(current_id)

                existing = graph_repo.find_relations(source_id=current_id)
                for rel in existing:
                    rel_type = rel.get("rel_type", "").upper().replace(" ", "_")
                    if rel_type != predicate:
                        continue
                    target = rel.get("target", {})
                    target_id = target.get("id")

                    if target_id == ancestor_id:
                        return True

                    if target_id not in visited:
                        next_queue.append(target_id)

            queue = next_queue
            depth += 1

        return False

    def infer_transitive(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> List[Triple]:
        """
        Inferiert alle transitiven Relationen für ein Triple.

        Returns:
            Liste von inferierten Triples
        """
        if not self.config.enable_inference or not graph_repo:
            return []

        predicate = triple.predicate.upper().replace(" ", "_")

        if predicate not in self.transitive_relations:
            return []

        inferred = []

        # Hole alle Vorfahren des Objects
        ancestors = self._get_all_ancestors(triple.object.id, predicate, graph_repo)

        for ancestor_id, ancestor_name, depth in ancestors:
            inferred_triple = Triple(
                subject=triple.subject,
                predicate=predicate,
                object=Entity(id=ancestor_id, name=ancestor_name, entity_type=triple.object.entity_type),
                source_text=f"Transitiv inferiert (Tiefe {depth}): {triple.source_text}",
                source_document_id=triple.source_document_id,
                extraction_confidence=triple.extraction_confidence * (0.95 ** depth),
            )
            inferred.append(inferred_triple)

        return inferred

    def _get_all_ancestors(
        self,
        entity_id: str,
        predicate: str,
        graph_repo: Any
    ) -> List[Tuple[str, str, int]]:
        """
        Holt alle Vorfahren einer Entity für eine transitive Relation.

        Returns:
            Liste von (entity_id, entity_name, depth) Tupeln
        """
        ancestors = []
        visited = set()
        queue = [(entity_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if depth > self.config.max_transitive_depth:
                continue
            if current_id in visited:
                continue
            visited.add(current_id)

            existing = graph_repo.find_relations(source_id=current_id)
            for rel in existing:
                rel_type = rel.get("rel_type", "").upper().replace(" ", "_")
                if rel_type != predicate:
                    continue
                target = rel.get("target", {})
                target_id = target.get("id")
                target_name = target.get("name", "?")

                if target_id and target_id not in visited:
                    ancestors.append((target_id, target_name, depth + 1))
                    queue.append((target_id, depth + 1))

        return ancestors


# =============================================================================
# DISJUNKTE TYPEN
# =============================================================================

class DisjointTypeChecker:
    """
    Prüft dass Entities keine disjunkten Typen haben.

    Beispiel:
    - Entity "Berlin" hat Typ "Stadt"
    - Neues Triple impliziert Typ "Person" für Berlin
    → Konflikt: Stadt und Person sind disjunkt
    """

    def __init__(self, config: AdvancedRulesConfig):
        self.config = config
        # Build lookup: type -> set of disjoint types
        self.disjoint_map: Dict[str, Set[str]] = defaultdict(set)
        for type_set in config.disjoint_types:
            for t in type_set:
                self.disjoint_map[t.lower()].update(
                    other.lower() for other in type_set if other != t
                )

    def check(
        self,
        triple: Triple,
        graph_repo: Any = None
    ) -> Optional[ConflictSet]:
        """
        Prüft ob das Triple disjunkte Typ-Constraints verletzt.
        """
        # Prüfe Subject
        conflict = self._check_entity_types(triple.subject)
        if conflict:
            return conflict

        # Prüfe Object
        conflict = self._check_entity_types(triple.object)
        if conflict:
            return conflict

        # Prüfe implizite Typen durch Relation
        conflict = self._check_implied_types(triple, graph_repo)
        if conflict:
            return conflict

        return None

    def _check_entity_types(self, entity: Entity) -> Optional[ConflictSet]:
        """
        Prüft ob eine Entity widersprüchliche Typen hat.
        """
        entity_type = entity.entity_type.value.lower()

        # Prüfe gegen zusätzliche Typen (falls Entity mehrere hat)
        if hasattr(entity, 'additional_types'):
            for additional_type in entity.additional_types:
                additional_lower = additional_type.lower()
                if additional_lower in self.disjoint_map.get(entity_type, set()):
                    return ConflictSet(
                        conflict_type=ConflictType.SCHEMA_VIOLATION,
                        description=f"Disjunkte Typen: '{entity.name}' hat Typ '{entity_type}' "
                                   f"und '{additional_type}', aber diese sind disjunkt",
                        severity=0.9
                    )

        return None

    def _check_implied_types(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> Optional[ConflictSet]:
        """
        Prüft ob die Relation Typen impliziert die disjunkt zum aktuellen Typ sind.
        """
        # Domain/Range Constraints definieren implizite Typen
        # z.B. GEBOREN_IN impliziert Subject = Person, Object = Ort

        RELATION_TYPE_IMPLICATIONS = {
            "GEBOREN_IN": {"subject": "person", "object": "ort"},
            "BORN_IN": {"subject": "person", "object": "location"},
            "GESTORBEN_IN": {"subject": "person", "object": "ort"},
            "DIED_IN": {"subject": "person", "object": "location"},
            "HAUPTSTADT_VON": {"subject": "stadt", "object": "land"},
            "CAPITAL_OF": {"subject": "city", "object": "country"},
            "CEO_VON": {"subject": "person", "object": "organisation"},
            "CEO_OF": {"subject": "person", "object": "organization"},
            "GRUENDETE": {"subject": "person", "object": "organisation"},
            "FOUNDED": {"subject": "person", "object": "organization"},
        }

        predicate = triple.predicate.upper().replace(" ", "_")
        implications = RELATION_TYPE_IMPLICATIONS.get(predicate)

        if not implications:
            return None

        # Prüfe Subject
        if "subject" in implications:
            implied_type = implications["subject"]
            actual_type = triple.subject.entity_type.value.lower()

            if implied_type in self.disjoint_map.get(actual_type, set()):
                return ConflictSet(
                    conflict_type=ConflictType.SCHEMA_VIOLATION,
                    description=f"Typ-Konflikt: Relation '{predicate}' impliziert Subject-Typ "
                               f"'{implied_type}', aber '{triple.subject.name}' hat Typ '{actual_type}' "
                               f"(disjunkt)",
                    severity=0.8
                )

        # Prüfe Object
        if "object" in implications:
            implied_type = implications["object"]
            actual_type = triple.object.entity_type.value.lower()

            if implied_type in self.disjoint_map.get(actual_type, set()):
                return ConflictSet(
                    conflict_type=ConflictType.SCHEMA_VIOLATION,
                    description=f"Typ-Konflikt: Relation '{predicate}' impliziert Object-Typ "
                               f"'{implied_type}', aber '{triple.object.name}' hat Typ '{actual_type}' "
                               f"(disjunkt)",
                    severity=0.8
                )

        return None


# =============================================================================
# KOMBINIERTER CHECKER
# =============================================================================

class AdvancedRulesValidator:
    """
    Kombiniert alle erweiterten Regel-Checker.
    """

    def __init__(self, config: AdvancedRulesConfig = None):
        self.config = config or AdvancedRulesConfig()
        self.inverse_checker = InverseRelationChecker(self.config)
        self.transitive_checker = TransitiveClosureChecker(self.config)
        self.disjoint_checker = DisjointTypeChecker(self.config)

        logger.info(
            f"AdvancedRulesValidator initialisiert: "
            f"{len(self.config.inverse_relations)} inverse, "
            f"{len(self.config.transitive_relations)} transitive, "
            f"{len(self.config.disjoint_types)} disjoint type sets"
        )

    def validate(
        self,
        triple: Triple,
        graph_repo: Any = None
    ) -> List[ConflictSet]:
        """
        Führt alle erweiterten Regel-Checks durch.

        Returns:
            Liste von gefundenen Konflikten
        """
        conflicts = []

        # 1. Inverse Relations
        conflict = self.inverse_checker.check(triple, graph_repo)
        if conflict:
            conflicts.append(conflict)

        # 2. Transitive Closure
        conflict = self.transitive_checker.check(triple, graph_repo)
        if conflict:
            conflicts.append(conflict)

        # 3. Disjunkte Typen
        conflict = self.disjoint_checker.check(triple, graph_repo)
        if conflict:
            conflicts.append(conflict)

        return conflicts

    def infer(
        self,
        triple: Triple,
        graph_repo: Any = None
    ) -> List[Triple]:
        """
        Inferiert neue Triples basierend auf Regeln.

        Returns:
            Liste von inferierten Triples
        """
        inferred = []

        # Inverse Relation
        inverse = self.inverse_checker.infer_inverse(triple)
        if inverse:
            inferred.append(inverse)

        # Transitive Relationen
        if graph_repo:
            transitive = self.transitive_checker.infer_transitive(triple, graph_repo)
            inferred.extend(transitive)

        return inferred
