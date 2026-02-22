# src/consistency/rules/rule_validator.py
"""
Stufe 1: Regelbasierte Validierung

Schnelle, deterministische Prüfungen:
- Schema-Validierung
- Kardinalitätsprüfungen  
- Temporale Konsistenz

Inspiriert von SHACL (Shapes Constraint Language).
"""

import time
import logging
from typing import List, Any, Dict, Optional, Tuple

import numpy as np

from src.models.entities import Triple, ConflictSet, ConflictType
from src.consistency.base import ValidationStage, StageResult, ValidationOutcome, ConsistencyConfig
from src.consistency.rules.temporal_rules import TemporalValidator, get_temporal_validator
from src.consistency.rules.advanced_rules import AdvancedRulesValidator, AdvancedRulesConfig

logger = logging.getLogger(__name__)


class RuleBasedValidator(ValidationStage):
    """Stufe 1: Regelbasierte Validierung."""

    name = "rule_based"

    def __init__(self, config: ConsistencyConfig, embedding_model: Any = None):
        self.config = config
        self.embedding_model = embedding_model
        self.valid_entity_types = [t.lower() for t in config.valid_entity_types]
        self.valid_relation_types = [t.lower() for t in config.valid_relation_types]

        # Mapping normalisieren (alles lowercase)
        self.relation_type_mapping: Dict[str, str] = {
            k.lower().replace(" ", "_"): v.lower().replace(" ", "_")
            for k, v in config.relation_type_mapping.items()
        }

        # Pre-compute Embeddings für kanonische Relationstypen (Tier 3)
        self._canonical_embeddings: Dict[str, np.ndarray] = {}
        if embedding_model is not None:
            self._precompute_relation_embeddings()

        # Temporaler Validator für robuste temporale Prüfungen
        self.temporal_validator = TemporalValidator()

        # Erweiterte Regeln: Inverse Relations, Transitive Closure, Disjunkte Typen
        self.advanced_rules = AdvancedRulesValidator()

        logger.info(
            f"RuleBasedValidator initialisiert mit {len(self.valid_entity_types)} Entity-Types, "
            f"{len(self.valid_relation_types)} Relation-Types, "
            f"{len(self.relation_type_mapping)} Mappings, "
            f"Embedding-Tier: {'aktiv' if embedding_model else 'deaktiviert'}, "
            f"Temporale Regeln: aktiv, "
            f"Erweiterte Regeln: aktiv (Inverse, Transitive, Disjunkte Typen)"
        )

    def _precompute_relation_embeddings(self):
        """Pre-compute Embeddings für alle kanonischen Relationstypen."""
        try:
            # Menschenlesbare Labels für bessere Embeddings
            canonical_types = list(set(self.valid_relation_types))
            labels = [t.replace("_", " ").lower() for t in canonical_types]

            # Unterstütze sowohl LangChain-Wrapper als auch SentenceTransformer
            if hasattr(self.embedding_model, 'embed_documents'):
                embeddings = self.embedding_model.embed_documents(labels)
            elif hasattr(self.embedding_model, 'encode'):
                # Raw SentenceTransformer
                embeddings = self.embedding_model.encode(labels, convert_to_numpy=True).tolist()
            else:
                raise AttributeError("Embedding-Modell unterstützt weder embed_documents noch encode")

            for rel_type, emb in zip(canonical_types, embeddings):
                self._canonical_embeddings[rel_type] = np.array(emb)
            logger.info(f"Pre-computed Embeddings für {len(self._canonical_embeddings)} kanonische Relationstypen")
        except Exception as e:
            logger.warning(f"Embedding-Vorberechnung fehlgeschlagen: {e}")
            self._canonical_embeddings = {}
    
    def validate(self, triple: Triple, graph_repo: Any = None) -> StageResult:
        """
        Führt regelbasierte Validierung durch.

        Prüfungen:
        1. Schema-Validität (Typen) — mit 3-Tier Relationstyp-Matching
        2. Domain Constraints (#1)
        3. Self-Loop (#2)
        4. Kardinalität (#3 erweitert)
        5. Zykluserkennung (#4) — nur mit Graph
        6. Symmetrie/Asymmetrie (#5) — nur mit Graph
        7. Cross-Relation Inference (#9) — nur mit Graph
        8. Temporale Plausibilität
        """
        start_time = time.time()
        conflicts: List[ConflictSet] = []
        confidence = 1.0

        checks_performed = []

        # 1. Schema-Validierung (mit 3-Tier Relationstyp-Matching)
        schema_conflict, relation_details = self._check_schema(triple)
        checks_performed.append("schema")
        if schema_conflict:
            conflicts.append(schema_conflict)
            if schema_conflict.severity < 0.8:
                confidence *= 0.6
            else:
                confidence *= 0.3

        # 2. Domain Constraints (#1)
        domain_conflict = self._check_domain_constraints(triple)
        checks_performed.append("domain_constraints")
        if domain_conflict:
            conflicts.append(domain_conflict)
            confidence *= 0.3

        # 3. Self-Loop (#2)
        self_loop_conflict = self._check_self_loop(triple)
        checks_performed.append("self_loop")
        if self_loop_conflict:
            conflicts.append(self_loop_conflict)
            confidence *= 0.3

        # 4. Kardinalitätsprüfung (nur wenn Graph verfügbar)
        if graph_repo:
            cardinality_conflict = self._check_cardinality(triple, graph_repo)
            checks_performed.append("cardinality")
            if cardinality_conflict:
                conflicts.append(cardinality_conflict)
                confidence *= 0.6

        # 5. Graph-basierte Checks (nur wenn Graph verfügbar)
        if graph_repo:
            # 5a. Zykluserkennung (#4)
            cycle_conflict = self._check_cycles(triple, graph_repo)
            checks_performed.append("cycle_detection")
            if cycle_conflict:
                conflicts.append(cycle_conflict)
                confidence *= 0.3

            # 5b. Symmetrie/Asymmetrie (#5)
            symmetry_conflict = self._check_symmetry(triple, graph_repo)
            checks_performed.append("symmetry")
            if symmetry_conflict:
                conflicts.append(symmetry_conflict)
                confidence *= 0.5

            # 5c. Cross-Relation Inference (#9)
            cross_conflict = self._check_cross_relation(triple, graph_repo)
            checks_performed.append("cross_relation")
            if cross_conflict:
                conflicts.append(cross_conflict)
                confidence *= 0.6

        # 6. Temporale Plausibilität (mit Graph-Kontext für robuste Prüfung)
        temporal_conflict = self._check_temporal(triple, graph_repo)
        checks_performed.append("temporal")
        if temporal_conflict:
            conflicts.append(temporal_conflict)
            confidence *= 0.3  # Temporale Widersprüche sind schwerwiegend

        # 7. Erweiterte Regeln (Inverse, Transitive, Disjunkte Typen)
        advanced_conflicts = self.advanced_rules.validate(triple, graph_repo)
        checks_performed.append("advanced_rules")
        for adv_conflict in advanced_conflicts:
            conflicts.append(adv_conflict)
            confidence *= 0.7  # Moderate Gewichtung für erweiterte Regeln

        # Outcome bestimmen
        if conflicts:
            hard_conflicts = [
                c for c in conflicts
                if c.conflict_type == ConflictType.TEMPORAL_INCONSISTENCY
                or (c.conflict_type == ConflictType.SCHEMA_VIOLATION and c.severity >= 0.8)
                or (c.conflict_type == ConflictType.CONTRADICTORY_RELATION and c.severity >= 0.9)
            ]
            if hard_conflicts:
                outcome = ValidationOutcome.FAIL
            else:
                outcome = ValidationOutcome.UNCERTAIN
        else:
            outcome = ValidationOutcome.PASS

        processing_time = (time.time() - start_time) * 1000

        logger.debug(f"Stufe 1 [{triple.subject.name}]: {outcome.value} (conf: {confidence:.2f})")

        return StageResult(
            outcome=outcome,
            confidence=confidence,
            conflicts=conflicts,
            processing_time_ms=processing_time,
            details={
                "checks_performed": checks_performed,
                "entity_types_valid": self._is_entity_type_valid(triple.subject) and
                                      self._is_entity_type_valid(triple.object),
                "relation_type_valid": relation_details.get("tier") in ("exact", "mapped"),
                "relation_type_details": relation_details,
            }
        )
    
    def _check_schema(self, triple: Triple) -> Tuple[Optional[ConflictSet], Dict]:
        """
        Prüft Schema-Konformität.

        Returns:
            Tuple aus (optional ConflictSet, Relation-Details-Dict)
        """
        errors = []
        relation_details: Dict[str, Any] = {}

        # Entitätstypen prüfen
        if not self._is_entity_type_valid(triple.subject):
            errors.append(f"Ungültiger Subject-Typ: '{triple.subject.entity_type.value}'")

        if not self._is_entity_type_valid(triple.object):
            errors.append(f"Ungültiger Object-Typ: '{triple.object.entity_type.value}'")

        # Relationstyp prüfen (3-Tier)
        relation_details = self._check_relation_type(triple.predicate)

        if relation_details["tier"] == "fail":
            errors.append(f"Ungültiger Relationstyp: '{triple.predicate}'")

        # Wenn Relationstyp gemappt wurde, Triple-Prädikat aktualisieren
        if relation_details.get("mapped_to"):
            original = triple.predicate
            triple.predicate = relation_details["mapped_to"].upper()
            logger.debug(f"  Relationstyp gemappt: {original} → {triple.predicate}")

        if errors:
            # Severity hängt vom Relationstyp-Tier ab
            # Entity-Typ-Fehler sind immer severity 0.9
            has_entity_error = any("Subject-Typ" in e or "Object-Typ" in e for e in errors)
            has_relation_error = any("Relationstyp" in e for e in errors)

            if has_entity_error:
                severity = 0.9  # Harter Fehler
            elif has_relation_error and relation_details.get("confidence", 0) < 0.5:
                severity = 0.9  # Relationstyp komplett unbekannt → harter Fehler
            elif has_relation_error:
                severity = 0.6  # Relationstyp unsicher → Soft-Fehler → UNCERTAIN
            else:
                severity = 0.9

            return ConflictSet(
                conflict_type=ConflictType.SCHEMA_VIOLATION,
                description=" | ".join(errors),
                severity=severity
            ), relation_details

        return None, relation_details

    def _check_relation_type(self, predicate: str) -> Dict[str, Any]:
        """
        3-Tier Relationstyp-Matching.

        Tier 1: Exakter Match → PASS (conf=1.0)
        Tier 2: Mapping-Lookup → PASS + Mapping auf kanonischen Typ (conf=0.9)
        Tier 3: Embedding-Similarity → PASS/UNCERTAIN/FAIL je nach Similarity
        """
        normalized = predicate.lower().replace(" ", "_")

        # Tier 1: Exakter Match
        if normalized in self.valid_relation_types:
            return {"tier": "exact", "confidence": 1.0, "original": predicate, "mapped_to": None}

        # Tier 2: Mapping-Lookup
        if normalized in self.relation_type_mapping:
            canonical = self.relation_type_mapping[normalized]
            return {"tier": "mapped", "confidence": 0.9, "original": predicate, "mapped_to": canonical}

        # Tier 3: Embedding-Similarity (wenn verfügbar)
        if self._canonical_embeddings:
            try:
                label = normalized.replace("_", " ")
                # Unterstütze sowohl LangChain-Wrapper als auch SentenceTransformer
                if hasattr(self.embedding_model, 'embed_query'):
                    query_emb = np.array(self.embedding_model.embed_query(label))
                elif hasattr(self.embedding_model, 'encode'):
                    query_emb = np.array(self.embedding_model.encode(label, convert_to_numpy=True))
                else:
                    raise AttributeError("Embedding-Modell unterstützt weder embed_query noch encode")
                best_sim = 0.0
                best_type = None
                for canon_type, canon_emb in self._canonical_embeddings.items():
                    sim = self._cosine_similarity(query_emb, canon_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_type = canon_type

                threshold = self.config.relation_type_similarity_threshold
                if best_sim >= threshold:
                    logger.debug(f"  Tier 3 Match: {predicate} → {best_type} (sim={best_sim:.2f})")
                    return {
                        "tier": "embedding", "confidence": float(best_sim),
                        "original": predicate, "mapped_to": best_type,
                        "similarity": float(best_sim),
                    }
                elif best_sim >= 0.5:
                    # Unsicher — eskaliere
                    return {
                        "tier": "uncertain", "confidence": float(best_sim),
                        "original": predicate, "mapped_to": best_type,
                        "similarity": float(best_sim),
                    }
            except Exception as e:
                logger.warning(f"Tier 3 Embedding-Matching fehlgeschlagen: {e}")

        # Kein Match
        return {"tier": "fail", "confidence": 0.0, "original": predicate, "mapped_to": None}

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Berechnet Cosine Similarity zwischen zwei Vektoren."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _is_entity_type_valid(self, entity) -> bool:
        """Prüft ob Entitätstyp gültig ist."""
        return entity.entity_type.value.lower() in self.valid_entity_types

    def _is_relation_type_valid(self, predicate: str) -> bool:
        """Prüft ob Relationstyp gültig ist (Tier 1 oder Tier 2)."""
        normalized = predicate.lower().replace(" ", "_")
        return normalized in self.valid_relation_types or normalized in self.relation_type_mapping
    
    def _check_cardinality(self, triple: Triple, graph_repo) -> ConflictSet | None:
        """Prüft Kardinalitätsbeschränkungen."""
        predicate_normalized = triple.predicate.upper().replace(" ", "_")
        rule = self.config.cardinality_rules.get(predicate_normalized)
        
        if not rule:
            return None
        
        max_allowed = rule.get("max", float("inf"))
        
        # Zähle existierende Relationen dieses Typs vom Subject
        existing = graph_repo.find_relations(source_id=triple.subject.id)
        same_type_count = sum(
            1 for r in existing 
            if r.get("rel_type", "").upper() == predicate_normalized
        )
        
        if same_type_count >= max_allowed:
            return ConflictSet(
                conflict_type=ConflictType.SCHEMA_VIOLATION,  # Schema-Verletzung für hartes FAIL
                description=f"Kardinalität überschritten: {triple.subject.name} hat bereits "
                           f"{same_type_count} '{predicate_normalized}'-Relationen (max: {max_allowed})",
                severity=0.9  # Hoch genug für hard_conflicts
            )
        return None
    
    def _check_domain_constraints(self, triple: Triple) -> Optional[ConflictSet]:
        """
        Prüft Subject/Object-Typ gegen Domain-Constraints (#1).

        Beispiel: GEBOREN_IN erwartet Person als Subject, Ort als Object.
        """
        predicate_normalized = triple.predicate.upper().replace(" ", "_")
        constraint = self.config.domain_constraints.get(predicate_normalized)

        if not constraint:
            return None

        subject_type = triple.subject.entity_type.value
        object_type = triple.object.entity_type.value
        errors = []

        allowed_subject = constraint.get("subject_types", [])
        if allowed_subject and subject_type not in allowed_subject:
            errors.append(
                f"Subject-Typ '{subject_type}' nicht erlaubt für '{predicate_normalized}' "
                f"(erwartet: {allowed_subject})"
            )

        allowed_object = constraint.get("object_types", [])
        if allowed_object and object_type not in allowed_object:
            errors.append(
                f"Object-Typ '{object_type}' nicht erlaubt für '{predicate_normalized}' "
                f"(erwartet: {allowed_object})"
            )

        if errors:
            return ConflictSet(
                conflict_type=ConflictType.SCHEMA_VIOLATION,
                description=" | ".join(errors),
                severity=0.9
            )

        return None

    def _check_self_loop(self, triple: Triple) -> Optional[ConflictSet]:
        """
        Prüft auf Self-Loops (#2).

        Erkennt wenn Subject und Object die gleiche Entität sind.
        Erlaubt für bestimmte Relationstypen (config.allow_reflexive).
        """
        predicate_normalized = triple.predicate.upper().replace(" ", "_")

        # Prüfe ob Reflexivität für diesen Typ erlaubt ist
        if predicate_normalized in [r.upper() for r in self.config.allow_reflexive]:
            return None

        # Prüfe auf Self-Loop (ID-Match oder Name-Match)
        is_self_loop = (
            triple.subject.id == triple.object.id
            or triple.subject.name.lower().strip() == triple.object.name.lower().strip()
        )

        if is_self_loop:
            return ConflictSet(
                conflict_type=ConflictType.SCHEMA_VIOLATION,
                description=f"Self-Loop erkannt: '{triple.subject.name}' --[{predicate_normalized}]--> "
                           f"'{triple.object.name}' (nicht erlaubt für '{predicate_normalized}')",
                severity=0.9
            )

        return None

    def _check_cycles(self, triple: Triple, graph_repo) -> Optional[ConflictSet]:
        """
        Zykluserkennung via BFS (#4).

        Prüft ob das Hinzufügen des Triples einen Zyklus erzeugen würde
        bei Relationstypen die azyklisch sein müssen.
        """
        predicate_normalized = triple.predicate.upper().replace(" ", "_")

        if predicate_normalized not in [r.upper() for r in self.config.acyclic_relations]:
            return None

        # BFS: Starte beim Object, suche ob wir zum Subject zurückkommen
        visited = set()
        queue = [triple.object.id]
        depth = 0

        while queue and depth < self.config.max_cycle_depth:
            next_queue = []
            for node_id in queue:
                if node_id in visited:
                    continue
                visited.add(node_id)

                # Finde alle ausgehenden Relationen gleichen Typs
                outgoing = graph_repo.find_relations(source_id=node_id)
                for rel in outgoing:
                    rel_type = rel.get("rel_type", "").upper()
                    if rel_type != predicate_normalized:
                        continue
                    target = rel.get("target", {})
                    target_id = target.get("id") if target else None
                    if not target_id:
                        continue

                    # Zyklus gefunden: Wir erreichen das Subject
                    if target_id == triple.subject.id:
                        return ConflictSet(
                            conflict_type=ConflictType.CONTRADICTORY_RELATION,
                            description=f"Zyklus erkannt: '{triple.subject.name}' --[{predicate_normalized}]--> "
                                       f"'{triple.object.name}' würde einen Zyklus erzeugen (Tiefe: {depth + 1})",
                            severity=0.9
                        )

                    if target_id not in visited:
                        next_queue.append(target_id)

            queue = next_queue
            depth += 1

        return None

    def _check_symmetry(self, triple: Triple, graph_repo) -> Optional[ConflictSet]:
        """
        Prüft Symmetrie/Asymmetrie-Constraints (#5).

        Für asymmetrische Relationen: Prüfe ob die inverse Relation
        (object → subject) bereits existiert.
        """
        predicate_normalized = triple.predicate.upper().replace(" ", "_")

        if predicate_normalized not in [r.upper() for r in self.config.asymmetric_relations]:
            return None

        # Suche inverse Relation: object → subject mit gleichem Typ
        existing = graph_repo.find_relations(source_id=triple.object.id)
        for rel in existing:
            rel_type = rel.get("rel_type", "").upper()
            if rel_type != predicate_normalized:
                continue
            target = rel.get("target", {})
            target_id = target.get("id") if target else None
            if target_id == triple.subject.id:
                return ConflictSet(
                    conflict_type=ConflictType.CONTRADICTORY_RELATION,
                    description=f"Asymmetrie-Verletzung: '{triple.object.name}' --[{predicate_normalized}]--> "
                               f"'{triple.subject.name}' existiert bereits, "
                               f"inverse Relation nicht erlaubt für asymmetrische Relation '{predicate_normalized}'",
                    severity=0.8
                )

        return None

    def _check_cross_relation(self, triple: Triple, graph_repo) -> Optional[ConflictSet]:
        """
        Cross-Relation Inference (#9).

        Prüft ob existierende Relationen des Objects einen Entity-Typ implizieren
        der im Widerspruch zum erwarteten Object-Typ der neuen Relation steht.
        """
        predicate_normalized = triple.predicate.upper().replace(" ", "_")
        new_constraint = self.config.domain_constraints.get(predicate_normalized)

        if not new_constraint:
            return None

        new_expected_object_types = new_constraint.get("object_types", [])
        if not new_expected_object_types:
            return None

        # Hole existierende Relationen des Objects (als Source)
        existing = graph_repo.find_relations(source_id=triple.object.id)
        for rel in existing:
            rel_type = rel.get("rel_type", "").upper()
            existing_constraint = self.config.domain_constraints.get(rel_type)
            if not existing_constraint:
                continue

            # Die existierende Relation impliziert einen Subject-Typ für das Object
            implied_types = existing_constraint.get("subject_types", [])
            if not implied_types:
                continue

            # Prüfe ob der implizierte Typ mit dem erwarteten Object-Typ kompatibel ist
            overlap = set(implied_types) & set(new_expected_object_types)
            if not overlap:
                return ConflictSet(
                    conflict_type=ConflictType.CONTRADICTORY_RELATION,
                    description=f"Cross-Relation Widerspruch: '{triple.object.name}' hat existierende "
                               f"Relation '{rel_type}' (impliziert Typ {implied_types}), "
                               f"aber '{predicate_normalized}' erwartet Object-Typ {new_expected_object_types}",
                    severity=0.7
                )

        return None

    def _check_temporal(self, triple: Triple, graph_repo: Any = None) -> ConflictSet | None:
        """
        Prüft temporale Plausibilität mit robuster Jahresextraktion.

        Prüfungen:
        1. valid_from/valid_until Konsistenz auf Entities
        2. Jahresextraktion aus Entity-Namen
        3. Temporaler Kontext aus Graph (Geburt/Tod)
        4. Lifetime-Events (Ereignis während Lebensspanne)

        Args:
            triple: Das zu validierende Triple
            graph_repo: Repository für Graph-Zugriff (optional)

        Returns:
            ConflictSet bei temporalem Widerspruch, sonst None
        """
        # 1. Legacy-Prüfung: valid_from/valid_until auf Entities
        if triple.subject.valid_from and triple.subject.valid_until:
            if triple.subject.valid_from > triple.subject.valid_until:
                return ConflictSet(
                    conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                    description=f"Temporaler Fehler: valid_from > valid_until für '{triple.subject.name}'",
                    severity=0.8
                )

        if triple.object.valid_from and triple.object.valid_until:
            if triple.object.valid_from > triple.object.valid_until:
                return ConflictSet(
                    conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                    description=f"Temporaler Fehler: valid_from > valid_until für '{triple.object.name}'",
                    severity=0.8
                )

        # 2. Robuste temporale Validierung mit Jahresextraktion
        return self.temporal_validator.validate(triple, graph_repo)
