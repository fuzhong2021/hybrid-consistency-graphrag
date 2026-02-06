# src/consistency/embedding_validator.py
"""
Stufe 2: Embedding-basierte Validierung

Semantische Analyse mittels Vektor-Embeddings:
- Duplikaterkennung (Cosine Similarity)
- Entity Resolution (Merge-Logik)
- Semantische Konfliktprüfung

Basiert auf iText2KG (Lairgi et al., 2024): α = 0.6 für Name-Gewichtung.
"""

import time
import logging
from typing import List, Any, Tuple, Optional, Dict, Set
from difflib import SequenceMatcher
import numpy as np

from src.models.entities import (
    Entity, Triple, ConflictSet, ConflictType,
    EntityResolutionResult, MergeStrategy, merge_entities
)
from src.consistency.base import ValidationStage, StageResult, ValidationOutcome, ConsistencyConfig

logger = logging.getLogger(__name__)


# Konstanten für Entity Resolution
ALPHA_NAME_WEIGHT = 0.6  # iText2KG: 60% Name, 40% Embedding
MIN_NAME_LENGTH_FOR_BLOCKING = 3  # Mindestlänge für Präfix-Blocking


def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Berechnet Jaro-Winkler-Ähnlichkeit zwischen zwei Strings.

    Jaro-Winkler bevorzugt Strings mit gemeinsamem Präfix
    und ist robust gegenüber Tippfehlern/Schreibvarianten.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Jaro-Distanz
    max_dist = max(len(s1), len(s2)) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)

    matches = 0
    transpositions = 0

    for i in range(len(s1)):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len(s2))

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len(s1)):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len(s1) + matches / len(s2) +
            (matches - transpositions / 2) / matches) / 3

    # Winkler-Bonus: Gemeinsamer Präfix (max 4 Zeichen)
    prefix_len = 0
    for i in range(min(4, min(len(s1), len(s2)))):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * 0.1 * (1 - jaro)


def _koelner_phonetik(word: str) -> str:
    """
    Berechnet den Kölner-Phonetik-Code für ein deutsches Wort.

    Die Kölner Phonetik ist ein phonetischer Algorithmus für die
    deutsche Sprache, ähnlich zu Soundex aber optimiert für Deutsch.
    """
    if not word:
        return ""

    word = word.upper().strip()

    # Buchstaben-zu-Code-Mapping (kontextabhängig)
    code = []
    prev_code = ""

    for i, ch in enumerate(word):
        prev_ch = word[i - 1] if i > 0 else ""
        next_ch = word[i + 1] if i < len(word) - 1 else ""

        c = ""
        if ch in "AEIOUJY":
            c = "0"
        elif ch == "H":
            c = ""
        elif ch == "B":
            c = "1"
        elif ch == "P":
            c = "1" if next_ch not in "H" else "3"
        elif ch in "DT":
            c = "8" if next_ch in "CSZ" else "2"
        elif ch in "FVW":
            c = "3"
        elif ch in "GKQ":
            c = "4"
        elif ch == "C":
            if i == 0:
                c = "4" if next_ch in "AHKLOQRUX" else "8"
            else:
                c = "4" if prev_ch in "SZ" or next_ch in "AHKOQUX" else "8"
        elif ch == "X":
            c = "48" if prev_ch not in "CKQ" else "8"
        elif ch == "L":
            c = "5"
        elif ch in "MN":
            c = "6"
        elif ch == "R":
            c = "7"
        elif ch in "SZ":
            c = "8"

        # Keine doppelten Codes hintereinander
        if c and c != prev_code:
            code.append(c)
            prev_code = c[-1] if c else prev_code
        elif c:
            prev_code = c[-1] if c else prev_code

    # Führende Nullen entfernen (außer am Anfang)
    result = "".join(code)
    if len(result) > 1:
        result = result[0] + result[1:].replace("0", "")

    return result


class LightweightTransE:
    """
    Leichtgewichtiges TransE-Modell für Knowledge Graph Embedding.

    Implementiert das TransE-Prinzip: h + r ≈ t
    mit Margin-based Ranking Loss und SGD-Optimierung.

    Nur numpy, keine externen Abhängigkeiten.
    """

    def __init__(self, embedding_dim: int = 50, learning_rate: float = 0.01,
                 margin: float = 1.0):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.relation_embeddings: Dict[str, np.ndarray] = {}
        self._trained = False

    def _init_embedding(self) -> np.ndarray:
        """Initialisiert ein zufälliges Embedding (normalisiert)."""
        emb = np.random.randn(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def _get_or_create(self, store: Dict[str, np.ndarray], key: str) -> np.ndarray:
        if key not in store:
            store[key] = self._init_embedding()
        return store[key]

    def train(self, triples: list, epochs: int = 100):
        """
        Trainiert TransE auf gegebenen Triples.

        Args:
            triples: Liste von (head_id, relation, tail_id) Tupeln
            epochs: Anzahl Trainingsepochen
        """
        if not triples:
            return

        entity_ids = set()
        for h, r, t in triples:
            entity_ids.add(h)
            entity_ids.add(t)
        entity_list = list(entity_ids)

        # Initialisiere Embeddings
        for h, r, t in triples:
            self._get_or_create(self.entity_embeddings, h)
            self._get_or_create(self.relation_embeddings, r)
            self._get_or_create(self.entity_embeddings, t)

        for epoch in range(epochs):
            total_loss = 0.0
            np.random.shuffle(triples)

            for h_id, rel, t_id in triples:
                h = self.entity_embeddings[h_id]
                r = self.relation_embeddings[rel]
                t = self.entity_embeddings[t_id]

                # Positive Distanz: ||h + r - t||
                pos_dist = np.linalg.norm(h + r - t)

                # Negative Sample: Ersetze tail durch zufällige Entity
                neg_id = entity_list[np.random.randint(len(entity_list))]
                while neg_id == t_id:
                    neg_id = entity_list[np.random.randint(len(entity_list))]
                t_neg = self._get_or_create(self.entity_embeddings, neg_id)
                neg_dist = np.linalg.norm(h + r - t_neg)

                # Margin-based ranking loss
                loss = max(0, self.margin + pos_dist - neg_dist)
                total_loss += loss

                if loss > 0:
                    # Gradient-Update
                    grad = (h + r - t)
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 0:
                        grad /= grad_norm

                    self.entity_embeddings[h_id] = h - self.learning_rate * grad
                    self.relation_embeddings[rel] = r - self.learning_rate * grad
                    self.entity_embeddings[t_id] = t + self.learning_rate * grad
                    self.entity_embeddings[neg_id] = t_neg - self.learning_rate * (-grad)

                    # Normalisierung
                    for eid in [h_id, t_id, neg_id]:
                        norm = np.linalg.norm(self.entity_embeddings[eid])
                        if norm > 0:
                            self.entity_embeddings[eid] /= norm

        self._trained = True

    def score(self, h_id: str, relation: str, t_id: str) -> float:
        """
        Berechnet TransE-Score: ||h + r - t||₂

        Niedriger Score = plausibler Triple.

        Returns:
            L2-Distanz (0 = perfekt, höher = unplausibler)
        """
        if not self._trained:
            return 0.0

        h = self.entity_embeddings.get(h_id)
        r = self.relation_embeddings.get(relation)
        t = self.entity_embeddings.get(t_id)

        if h is None or r is None or t is None:
            return 0.0

        return float(np.linalg.norm(h + r - t))


# =============================================================================
# Embedding Model Wrapper
# =============================================================================

class LocalEmbeddingModel:
    """
    Wrapper für lokale Embedding-Modelle (SentenceTransformers).

    Ermöglicht Duplikaterkennung ohne externe API-Aufrufe.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Name des SentenceTransformers-Modells
                       - "all-MiniLM-L6-v2": Schnell, gut für Entity Resolution
                       - "paraphrase-multilingual-MiniLM-L12-v2": Mehrsprachig
        """
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy Loading des Modells."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Lade SentenceTransformer: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info(f"SentenceTransformer geladen: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers nicht installiert. "
                          "Installiere mit: pip install sentence-transformers")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Erstellt Embedding für einen Text."""
        self._ensure_initialized()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Erstellt Embeddings für mehrere Texte (Batch)."""
        self._ensure_initialized()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Dimension der Embeddings."""
        self._ensure_initialized()
        return self._model.get_sentence_embedding_dimension()


class EmbeddingValidator(ValidationStage):
    """Stufe 2: Embedding-basierte Duplikaterkennung."""
    
    name = "embedding_based"
    
    def __init__(self, config: ConsistencyConfig, embedding_model: Any = None):
        """
        Args:
            config: Konsistenz-Konfiguration
            embedding_model: Embedding-Modell (z.B. OpenAI, SentenceTransformers)
        """
        self.config = config
        self.embedding_model = embedding_model
        self.similarity_threshold = config.similarity_threshold
        
        # Cache für Embeddings
        self._embedding_cache = {}
        
        logger.info(f"EmbeddingValidator initialisiert (Threshold: {self.similarity_threshold})")
    
    def validate(self, triple: Triple, graph_repo: Any = None) -> StageResult:
        """
        Prüft auf semantische Duplikate und Konflikte.

        Erweitert um:
        - Provenance-Boost (#7)
        - Anomalie-Erkennung (#8)
        - TransE-Scoring (#10)
        """
        start_time = time.time()
        conflicts: List[ConflictSet] = []
        confidence = 1.0
        details: Dict[str, Any] = {}

        if graph_repo is None or not self.embedding_model:
            # Auch ohne Embedding-Modell können wir Provenance/Anomalie/TransE prüfen
            if graph_repo is not None:
                # 4. Provenance Boost (#7)
                if self.config.enable_provenance_boost:
                    prov_multiplier = self._check_provenance(triple, graph_repo)
                    if prov_multiplier > 1.0:
                        confidence = min(1.0, confidence * prov_multiplier)
                        details["provenance_boost"] = prov_multiplier

                # 5. Anomalie-Erkennung (#8)
                if self.config.enable_anomaly_detection:
                    anomaly_conflict = self._check_anomalies(triple, graph_repo)
                    if anomaly_conflict:
                        conflicts.append(anomaly_conflict)
                        confidence *= self.config.anomaly_confidence_penalty

                # 6. TransE Scoring (#10)
                if self.config.enable_transe:
                    transe_conflict = self._check_transe(triple, graph_repo)
                    if transe_conflict:
                        conflicts.append(transe_conflict)

            if not self.embedding_model:
                logger.warning("Embedding-Modell nicht verfügbar - Stufe 2 eingeschränkt")
                outcome = ValidationOutcome.CONFLICT if conflicts else ValidationOutcome.PASS
                return StageResult(
                    outcome=outcome,
                    confidence=min(0.8, confidence),
                    conflicts=conflicts,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    details={**details, "skipped_embedding": True, "reason": "missing_embedding_model"}
                )

        # 1. Duplikaterkennung für Subject
        subject_duplicates = self._find_semantic_duplicates(triple.subject, graph_repo)
        if subject_duplicates:
            for dup_entity, similarity in subject_duplicates:
                conflicts.append(ConflictSet(
                    conflict_type=ConflictType.DUPLICATE_ENTITY,
                    conflicting_items=[triple.subject, dup_entity],
                    description=f"Potentielles Duplikat: '{triple.subject.name}' ≈ "
                               f"'{dup_entity.name}' (Similarity: {similarity:.2%})",
                    severity=0.6
                ))
            confidence *= 0.7

        # 2. Duplikaterkennung für Object
        object_duplicates = self._find_semantic_duplicates(triple.object, graph_repo)
        if object_duplicates:
            for dup_entity, similarity in object_duplicates:
                conflicts.append(ConflictSet(
                    conflict_type=ConflictType.DUPLICATE_ENTITY,
                    conflicting_items=[triple.object, dup_entity],
                    description=f"Potentielles Duplikat: '{triple.object.name}' ≈ "
                               f"'{dup_entity.name}' (Similarity: {similarity:.2%})",
                    severity=0.6
                ))
            confidence *= 0.7

        # 2.5 Relationstyp-Validierung (für Stufe 1 UNCERTAIN-Eskalationen)
        relation_type_result = self._validate_relation_type(triple)
        if relation_type_result:
            rel_outcome, rel_confidence = relation_type_result
            if rel_outcome == "mapped":
                confidence *= rel_confidence
            elif rel_outcome == "uncertain":
                confidence *= rel_confidence
            elif rel_outcome == "fail":
                conflicts.append(ConflictSet(
                    conflict_type=ConflictType.SCHEMA_VIOLATION,
                    description=f"Relationstyp '{triple.predicate}' konnte auch per "
                               f"Embedding nicht validiert werden",
                    severity=0.9
                ))
                confidence *= 0.3

        # 3. Prüfe auf widersprüchliche Relationen
        if graph_repo:
            relation_conflict = self._check_contradictory_relations(triple, graph_repo)
            if relation_conflict:
                conflicts.append(relation_conflict)
                confidence *= 0.5

        # 4. Provenance Boost (#7)
        if graph_repo and self.config.enable_provenance_boost:
            prov_multiplier = self._check_provenance(triple, graph_repo)
            if prov_multiplier > 1.0:
                confidence = min(1.0, confidence * prov_multiplier)
                details["provenance_boost"] = prov_multiplier

        # 5. Anomalie-Erkennung (#8)
        if graph_repo and self.config.enable_anomaly_detection:
            anomaly_conflict = self._check_anomalies(triple, graph_repo)
            if anomaly_conflict:
                conflicts.append(anomaly_conflict)
                confidence *= self.config.anomaly_confidence_penalty

        # 6. TransE Scoring (#10)
        if graph_repo and self.config.enable_transe:
            transe_conflict = self._check_transe(triple, graph_repo)
            if transe_conflict:
                conflicts.append(transe_conflict)

        # Outcome bestimmen
        if conflicts:
            outcome = ValidationOutcome.CONFLICT
        else:
            outcome = ValidationOutcome.PASS

        processing_time = (time.time() - start_time) * 1000

        logger.debug(f"Stufe 2 [{triple.subject.name}]: {outcome.value}, "
                    f"{len(conflicts)} Konflikte, {processing_time:.1f}ms")

        return StageResult(
            outcome=outcome,
            confidence=confidence,
            conflicts=conflicts,
            processing_time_ms=processing_time,
            details={
                **details,
                "subject_duplicates_found": len(subject_duplicates) if self.embedding_model else 0,
                "object_duplicates_found": len(object_duplicates) if self.embedding_model else 0,
                "similarity_threshold": self.similarity_threshold,
                "relation_type_validated": relation_type_result is not None,
            }
        )
    
    def _validate_relation_type(
        self,
        triple: Triple
    ) -> Optional[Tuple[str, float]]:
        """
        Prüft unsichere Relationstypen per Embedding-Similarity (Stufe 2).

        Liest Stufe-1-Details aus triple.validation_history.
        Wird nur aktiv wenn Stufe 1 den Relationstyp als 'uncertain' oder 'fail' markiert hat.

        Returns:
            None wenn keine Aktion nötig,
            ("mapped", confidence) wenn erfolgreich gemappt,
            ("uncertain", confidence) wenn unsicher aber akzeptabel,
            ("fail", confidence) wenn nicht validierbar
        """
        if not triple.validation_history:
            return None

        # Suche Stufe-1-Details
        stage1_entry = None
        for entry in triple.validation_history:
            if entry.get("stage") == "rule_based":
                stage1_entry = entry
                break

        if not stage1_entry:
            return None

        details = stage1_entry.get("details", {})
        rel_details = details.get("relation_type_details", {})
        tier = rel_details.get("tier")

        # Nur aktiv bei unsicheren oder fehlgeschlagenen Relationstypen
        if tier not in ("uncertain", "fail"):
            return None

        if not self.embedding_model:
            # Ohne Embedding-Modell können wir nicht weiter validieren
            if tier == "uncertain":
                return ("uncertain", 0.6)
            return None

        # Embedding-basierte Validierung
        predicate = triple.predicate
        normalized = predicate.lower().replace(" ", "_")
        label = normalized.replace("_", " ")

        try:
            query_emb = np.array(self.embedding_model.embed_query(label))

            # Vergleiche mit allen kanonischen Typen
            canonical_types = list(set(
                t.lower().replace(" ", "_")
                for t in self.config.valid_relation_types
            ))
            canonical_labels = [t.replace("_", " ") for t in canonical_types]
            canonical_embeddings = self.embedding_model.embed_documents(canonical_labels)

            best_sim = 0.0
            best_type = None
            for canon_type, canon_emb in zip(canonical_types, canonical_embeddings):
                canon_emb = np.array(canon_emb)
                sim = self._cosine_similarity(query_emb, canon_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_type = canon_type

            threshold = self.config.relation_type_similarity_threshold

            if best_sim >= threshold and best_type:
                # Erfolgreich gemappt
                original = triple.predicate
                triple.predicate = best_type.upper()
                logger.info(f"  Stufe 2 Relationstyp-Mapping: {original} → {triple.predicate} "
                           f"(sim={best_sim:.2f})")
                return ("mapped", max(0.7, float(best_sim)))
            elif best_sim >= 0.5:
                return ("uncertain", float(best_sim))
            else:
                return ("fail", float(best_sim))

        except Exception as e:
            logger.warning(f"Stufe 2 Relationstyp-Validierung fehlgeschlagen: {e}")
            if tier == "uncertain":
                return ("uncertain", 0.5)
            return None

    def _find_semantic_duplicates(
        self,
        entity: Entity,
        graph_repo: Any
    ) -> List[Tuple[Entity, float]]:
        """
        Findet semantisch ähnliche Entitäten im Graphen.

        Verwendet gewichtete Kombination aus Name und Embedding (iText2KG-Ansatz):
        - weighted_sim = α * name_sim + (1-α) * embedding_sim
        - α = 0.6 (60% Name, 40% Embedding)

        Fallback: Wenn kein Embedding-Modell vorhanden, rein name-basierte
        Duplikaterkennung mit Jaro-Winkler + Kölner Phonetik.

        Suchstrategie:
        1. Alle Entitäten des gleichen Typs
        2. Zusätzlich: Namensbasierte Suche mit verschiedenen Tokens
        """
        duplicates = []

        if graph_repo is None:
            return []

        # Embedding für neue Entität (kann None sein wenn kein Modell)
        new_embedding = self._get_embedding(entity)
        has_embeddings = new_embedding is not None

        # Sammle Kandidaten aus mehreren Quellen
        candidates_set = set()

        # 1. Alle Entitäten des gleichen Typs (für kleine Graphen effizient)
        if hasattr(graph_repo, 'find_all_entities'):
            type_candidates = graph_repo.find_all_entities(entity_type=entity.entity_type)
            logger.debug(f"  find_all_entities({entity.entity_type.value}): {[c.name for c in type_candidates]}")
            for c in type_candidates:
                candidates_set.add(c.id)

        # 2. Namensbasierte Suche mit verschiedenen Tokens
        name_tokens = entity.name.replace(".", " ").replace("-", " ").split()
        for token in name_tokens:
            if len(token) >= 2:  # Mindestens 2 Zeichen
                try:
                    token_candidates = graph_repo.find_by_name(token, entity_type=None)
                    for c in token_candidates:
                        candidates_set.add(c.id)
                except Exception:
                    pass

        # Hole die tatsächlichen Entity-Objekte
        candidates = []
        for cid in candidates_set:
            if hasattr(graph_repo, 'get_entity'):
                c = graph_repo.get_entity(cid)
                if c:
                    candidates.append(c)
            elif hasattr(graph_repo, '_entities'):
                c = graph_repo._entities.get(cid)
                if c:
                    candidates.append(c)

        for candidate in candidates:
            if candidate.id == entity.id:
                continue

            # Berechne Name-Similarity (Jaro-Winkler + Kölner Phonetik)
            name_sim = self._compute_name_similarity(entity.name, candidate.name)

            if has_embeddings:
                # Gewichtete Similarity (60% Name + 40% Embedding)
                candidate_embedding = self._get_embedding(candidate)
                if candidate_embedding is None:
                    # Fallback: Nur Name-Similarity
                    if name_sim >= 0.8:
                        duplicates.append((candidate, name_sim))
                    continue

                # Cosine Similarity berechnen
                embedding_sim = self._cosine_similarity(new_embedding, candidate_embedding)

                # Gewichtete Similarity (iText2KG-Ansatz)
                weighted_sim = ALPHA_NAME_WEIGHT * name_sim + (1 - ALPHA_NAME_WEIGHT) * embedding_sim

                if weighted_sim >= self.similarity_threshold or embedding_sim >= 0.9:
                    duplicates.append((candidate, weighted_sim))
            else:
                # Kein Embedding-Modell: Rein name-basierte Duplikaterkennung
                if name_sim >= self.similarity_threshold:
                    duplicates.append((candidate, name_sim))

        # Sortiere nach Ähnlichkeit (höchste zuerst)
        duplicates.sort(key=lambda x: x[1], reverse=True)

        return duplicates
    
    def _get_embedding(self, entity: Entity) -> Optional[np.ndarray]:
        """Holt oder berechnet das Embedding einer Entität."""
        cache_key = f"{entity.name}:{entity.entity_type.value}"
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if entity.embedding:
            embedding = np.array(entity.embedding)
        else:
            # Text für Embedding zusammensetzen
            text = f"{entity.name} ({entity.entity_type.value})"
            if entity.description:
                text += f": {entity.description}"
            
            try:
                embedding = np.array(self.embedding_model.embed_query(text))
            except Exception as e:
                logger.error(f"Embedding-Fehler für '{entity.name}': {e}")
                return None
        
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Berechnet Cosine Similarity zwischen zwei Vektoren."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _check_contradictory_relations(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> Optional[ConflictSet]:
        """
        Prüft auf widersprüchliche Relationen.

        Beispiel: Person WOHNT_IN Berlin, aber auch WOHNT_IN München
        (wenn Kardinalität = 1)
        """
        # Hole existierende Relationen des Subjects
        existing_relations = graph_repo.find_relations(source_id=triple.subject.id)

        for rel in existing_relations:
            rel_type = rel.get("rel_type", "")
            target_name = rel.get("target", {}).get("name", "")

            # Gleicher Relationstyp, aber anderes Ziel?
            if rel_type.upper() == triple.predicate.upper():
                if target_name.lower() != triple.object.name.lower():
                    # Prüfe ob diese Relation nur einmal vorkommen darf
                    if triple.predicate.upper() in self.config.cardinality_rules:
                        return ConflictSet(
                            conflict_type=ConflictType.CONTRADICTORY_RELATION,
                            description=f"Widerspruch: '{triple.subject.name}' hat bereits "
                                       f"'{rel_type}' -> '{target_name}', "
                                       f"neues Ziel wäre '{triple.object.name}'",
                            severity=0.8
                        )

        return None

    # =========================================================================
    # Provenance, Anomalie, TransE (#7, #8, #10)
    # =========================================================================

    def _check_provenance(self, triple: Triple, graph_repo: Any) -> float:
        """
        Provenance-Boost (#7): Mehrfache Bestätigung durch verschiedene Quellen.

        Sucht existierende Relationen mit gleichem Subject+Predicate+Object
        und zählt verschiedene source_document_id-Werte.

        Returns:
            Multiplikator (1.0 = kein Boost, 1.1 = 2 Quellen, 1.2 = 3+ Quellen)
        """
        existing = graph_repo.find_relations(source_id=triple.subject.id)
        source_docs: Set[str] = set()

        # Sammle source_document_ids von übereinstimmenden Relationen
        predicate_upper = triple.predicate.upper()
        for rel in existing:
            rel_type = rel.get("rel_type", "").upper()
            if rel_type != predicate_upper:
                continue
            target = rel.get("target", {})
            target_name = target.get("name", "") if target else ""
            if target_name.lower() != triple.object.name.lower():
                continue

            # Gefunden: gleiche Relation
            relation_obj = rel.get("relation")
            if relation_obj and hasattr(relation_obj, "source_document_id"):
                if relation_obj.source_document_id:
                    source_docs.add(relation_obj.source_document_id)

        # Füge die Quelle des neuen Triples hinzu
        if triple.source_document_id:
            source_docs.add(triple.source_document_id)

        n_sources = len(source_docs)
        if n_sources >= 3:
            return self.config.provenance_boost_3_plus
        elif n_sources >= 2:
            return self.config.provenance_boost_2_sources
        return 1.0

    def _check_anomalies(self, triple: Triple, graph_repo: Any) -> Optional[ConflictSet]:
        """
        Anomalie-Erkennung via Knotengrad-Statistik (#8).

        Berechnet den Z-Score des Subject-Knotengrads.
        Entities mit ungewöhnlich vielen Relationen werden geflaggt.
        """
        if not hasattr(graph_repo, 'find_all_entities'):
            return None

        # Cache-Key für Grad-Statistik
        cache_key = "_anomaly_degree_cache"
        if not hasattr(self, cache_key):
            setattr(self, cache_key, {})
        cache = getattr(self, cache_key)

        # Statistik berechnen (mit Caching)
        stats = cache.get("stats")
        if stats is None:
            all_entities = graph_repo.find_all_entities()
            if len(all_entities) < 10:
                return None  # Zu wenig Daten für sinnvolle Statistik

            degrees = []
            degree_map = {}
            for entity in all_entities:
                rels = graph_repo.find_relations(source_id=entity.id)
                deg = len(rels)
                degrees.append(deg)
                degree_map[entity.id] = deg

            mean_degree = np.mean(degrees)
            std_degree = np.std(degrees)

            stats = {"mean": mean_degree, "std": std_degree, "degree_map": degree_map}
            cache["stats"] = stats

        mean_degree = stats["mean"]
        std_degree = stats["std"]
        degree_map = stats["degree_map"]

        if std_degree == 0:
            return None

        # Z-Score des Subject-Knotengrads
        subject_degree = degree_map.get(triple.subject.id, 0)
        # +1 für die neue Relation
        subject_degree += 1
        z_score = (subject_degree - mean_degree) / std_degree

        if abs(z_score) > self.config.anomaly_zscore_threshold:
            return ConflictSet(
                conflict_type=ConflictType.CONTRADICTORY_RELATION,
                description=f"Knotengrad-Anomalie: '{triple.subject.name}' hat Grad {subject_degree} "
                           f"(Durchschnitt: {mean_degree:.1f}, Z-Score: {z_score:.2f}, "
                           f"Threshold: {self.config.anomaly_zscore_threshold})",
                severity=0.5
            )

        return None

    def _check_transe(self, triple: Triple, graph_repo: Any) -> Optional[ConflictSet]:
        """
        TransE-Scoring (#10): Knowledge Graph Embedding-basierte Anomalie-Erkennung.

        Trainiert ein leichtgewichtiges TransE-Modell und prüft ob der
        neue Triple konsistent mit den gelernten Embeddings ist.
        """
        if not hasattr(graph_repo, 'find_all_entities'):
            return None

        # Initialisiere TransE-Modell wenn nötig
        if not hasattr(self, '_transe_model'):
            self._transe_model = LightweightTransE(
                embedding_dim=self.config.transe_embedding_dim,
                learning_rate=self.config.transe_learning_rate
            )
            self._transe_triple_count = 0

        # Sammle alle existierenden Triples
        all_entities = graph_repo.find_all_entities()
        all_triples = []
        for entity in all_entities:
            rels = graph_repo.find_relations(source_id=entity.id)
            for rel in rels:
                target = rel.get("target", {})
                target_id = target.get("id") if target else None
                if target_id:
                    all_triples.append((entity.id, rel.get("rel_type", ""), target_id))

        if len(all_triples) < self.config.transe_min_triples:
            return None

        # Retrain wenn nötig
        if (not self._transe_model._trained or
                len(all_triples) - self._transe_triple_count >= self.config.transe_retrain_interval):
            self._transe_model.train(all_triples, epochs=self.config.transe_epochs)
            self._transe_triple_count = len(all_triples)

        # Score berechnen
        score = self._transe_model.score(
            triple.subject.id,
            triple.predicate.upper(),
            triple.object.id
        )

        if score > self.config.transe_anomaly_threshold:
            return ConflictSet(
                conflict_type=ConflictType.CONTRADICTORY_RELATION,
                description=f"TransE-Anomalie: Triple ({triple.subject.name}, {triple.predicate}, "
                           f"{triple.object.name}) hat Score {score:.2f} "
                           f"(Threshold: {self.config.transe_anomaly_threshold})",
                severity=0.5
            )

        return None

    # =========================================================================
    # Entity Resolution Methoden
    # =========================================================================

    def resolve_entity(
        self,
        entity: Entity,
        graph_repo: Any,
        perform_merge: bool = False
    ) -> EntityResolutionResult:
        """
        Führt Entity Resolution für eine einzelne Entität durch.

        Args:
            entity: Die zu prüfende Entität
            graph_repo: Repository für Graph-Zugriff
            perform_merge: Wenn True, wird bei Duplikat automatisch gemergt

        Returns:
            EntityResolutionResult mit Duplikat-Info und ggf. Merge-Ergebnis
        """
        # Finde Duplikat-Kandidaten
        duplicates = self._find_semantic_duplicates(entity, graph_repo)

        if not duplicates:
            return EntityResolutionResult(
                is_duplicate=False,
                canonical_entity=entity,
                similarity_score=0.0,
                reasoning="Keine ähnlichen Entitäten gefunden"
            )

        # Bestes Match
        best_match, best_similarity = duplicates[0]

        # Berechne detaillierte Ähnlichkeiten
        name_sim = self._compute_name_similarity(entity.name, best_match.name)
        embedding_sim = best_similarity  # Aus _find_semantic_duplicates

        # Gewichtete Similarity (iText2KG-Ansatz)
        weighted_sim = ALPHA_NAME_WEIGHT * name_sim + (1 - ALPHA_NAME_WEIGHT) * embedding_sim

        # Typ-Match prüfen
        type_match = entity.entity_type == best_match.entity_type

        # Entscheide ob es ein echtes Duplikat ist
        is_duplicate = weighted_sim >= self.similarity_threshold and type_match

        # Bestimme Merge-Strategie
        if name_sim >= 0.95:
            strategy = MergeStrategy.NAME_MATCH
        elif embedding_sim >= 0.95:
            strategy = MergeStrategy.EMBEDDING_MATCH
        else:
            strategy = MergeStrategy.HYBRID

        result = EntityResolutionResult(
            is_duplicate=is_duplicate,
            similarity_score=weighted_sim,
            name_similarity=name_sim,
            embedding_similarity=embedding_sim,
            type_match=type_match,
            merge_strategy=strategy,
            reasoning=self._generate_resolution_reasoning(
                entity, best_match, name_sim, embedding_sim, type_match, is_duplicate
            )
        )

        # Optional: Merge durchführen
        if is_duplicate and perform_merge:
            merged = merge_entities([entity, best_match], strategy)
            result.canonical_entity = merged
            result.merged_from = [entity, best_match]
        elif is_duplicate:
            result.canonical_entity = best_match  # Existierende Entity als kanonisch
            result.merged_from = [entity]
        else:
            result.canonical_entity = entity

        return result

    def resolve_entities_batch(
        self,
        entities: List[Entity],
        graph_repo: Any
    ) -> Dict[str, EntityResolutionResult]:
        """
        Batch Entity Resolution mit Blocking-Strategie für Skalierbarkeit.

        Verwendet Präfix-basiertes Blocking um O(n²) Vergleiche zu vermeiden.

        Args:
            entities: Liste der zu prüfenden Entitäten
            graph_repo: Repository für Graph-Zugriff

        Returns:
            Dict von entity_id -> EntityResolutionResult
        """
        results: Dict[str, EntityResolutionResult] = {}

        # Blocking: Gruppiere nach Präfix
        blocks = self._create_blocks(entities)

        logger.info(f"Entity Resolution: {len(entities)} Entitäten in {len(blocks)} Blocks")

        # Verarbeite jeden Block
        for block_key, block_entities in blocks.items():
            # Innerhalb eines Blocks: Paarweise Vergleiche
            for entity in block_entities:
                if entity.id in results:
                    continue

                result = self.resolve_entity(entity, graph_repo, perform_merge=False)
                results[entity.id] = result

        return results

    def _create_blocks(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """
        Erstellt Blocks für effiziente Entity Resolution.

        Blocking-Strategie: Präfix des normalisierten Namens
        """
        blocks: Dict[str, List[Entity]] = {}

        for entity in entities:
            # Normalisiere Name
            normalized = entity.name.lower().strip()

            # Mehrere Block-Keys für bessere Recall:
            # 1. Präfix-Block
            if len(normalized) >= MIN_NAME_LENGTH_FOR_BLOCKING:
                prefix_key = f"prefix:{normalized[:MIN_NAME_LENGTH_FOR_BLOCKING]}"
                if prefix_key not in blocks:
                    blocks[prefix_key] = []
                blocks[prefix_key].append(entity)

            # 2. Typ-Block
            type_key = f"type:{entity.entity_type.value}"
            if type_key not in blocks:
                blocks[type_key] = []
            blocks[type_key].append(entity)

        return blocks

    def _compute_name_similarity(self, name1: str, name2: str) -> float:
        """
        Berechnet Namensähnlichkeit (#6: verbessert).

        Kombiniert:
        - Jaro-Winkler-Similarity (robust bei Tippfehlern)
        - SequenceMatcher (Longest Common Subsequence)
        - Kölner-Phonetik-Bonus (für deutsche Schreibvarianten)
        """
        # Normalisierung
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        # Exakter Match
        if n1 == n2:
            return 1.0

        # Basis-Similarity: Maximum aus Jaro-Winkler und SequenceMatcher
        base_sim = max(
            _jaro_winkler_similarity(n1, n2),
            SequenceMatcher(None, n1, n2).ratio()
        )

        # Phonetik-Bonus: Gleicher Kölner-Phonetik-Code → +0.15
        phonetic_bonus = 0.0
        if _koelner_phonetik(n1) == _koelner_phonetik(n2):
            phonetic_bonus = 0.15

        return min(1.0, base_sim + phonetic_bonus)

    def _generate_resolution_reasoning(
        self,
        entity: Entity,
        candidate: Entity,
        name_sim: float,
        embedding_sim: float,
        type_match: bool,
        is_duplicate: bool
    ) -> str:
        """Generiert menschenlesbares Reasoning für die Resolution-Entscheidung."""
        parts = []

        parts.append(f"Vergleich: '{entity.name}' vs. '{candidate.name}'")
        parts.append(f"Namensähnlichkeit: {name_sim:.2%}")
        parts.append(f"Embedding-Ähnlichkeit: {embedding_sim:.2%}")
        parts.append(f"Gewichtete Ähnlichkeit: {ALPHA_NAME_WEIGHT * name_sim + (1 - ALPHA_NAME_WEIGHT) * embedding_sim:.2%}")
        parts.append(f"Typ-Match: {'Ja' if type_match else 'Nein'} ({entity.entity_type.value} vs. {candidate.entity_type.value})")
        parts.append(f"Threshold: {self.similarity_threshold:.2%}")

        if is_duplicate:
            parts.append("Entscheidung: DUPLIKAT - Merge empfohlen")
        else:
            if not type_match:
                parts.append("Entscheidung: KEIN DUPLIKAT - Typen unterschiedlich")
            else:
                parts.append("Entscheidung: KEIN DUPLIKAT - Ähnlichkeit unter Threshold")

        return " | ".join(parts)

    def get_entity_resolution_candidates(
        self,
        entity: Entity,
        graph_repo: Any,
        top_k: int = 5
    ) -> List[Tuple[Entity, EntityResolutionResult]]:
        """
        Gibt die Top-K Merge-Kandidaten für eine Entität zurück.

        Nützlich für manuelle Überprüfung oder UI.

        Args:
            entity: Die zu prüfende Entität
            graph_repo: Repository für Graph-Zugriff
            top_k: Anzahl der zurückzugebenden Kandidaten

        Returns:
            Liste von (Entity, EntityResolutionResult) Tupeln
        """
        duplicates = self._find_semantic_duplicates(entity, graph_repo)

        candidates = []
        for candidate_entity, similarity in duplicates[:top_k]:
            name_sim = self._compute_name_similarity(entity.name, candidate_entity.name)

            result = EntityResolutionResult(
                is_duplicate=similarity >= self.similarity_threshold,
                canonical_entity=candidate_entity,
                similarity_score=similarity,
                name_similarity=name_sim,
                embedding_similarity=similarity,
                type_match=entity.entity_type == candidate_entity.entity_type,
                merge_strategy=MergeStrategy.HYBRID,
                reasoning=self._generate_resolution_reasoning(
                    entity, candidate_entity, name_sim, similarity,
                    entity.entity_type == candidate_entity.entity_type,
                    similarity >= self.similarity_threshold
                )
            )

            candidates.append((candidate_entity, result))

        return candidates
