#!/usr/bin/env python3
# src/evaluation/fever_loader.py
"""
FEVER Dataset Loader für Fact Verification.

FEVER (Fact Extraction and VERification) ist ideal für Konsistenzprüfung:
- Echte Fact Verification Aufgabe
- Labels: SUPPORTS, REFUTES, NOT ENOUGH INFO
- Evidence aus Wikipedia

Wissenschaftliche Referenz:
- Thorne et al. (2018): FEVER: a Large-scale Dataset for Fact Extraction and VERification
- https://fever.ai/

Dataset-Struktur:
- claim: Die zu verifizierende Aussage
- label: SUPPORTS, REFUTES, NOT ENOUGH INFO
- evidence: Liste von (doc_id, sentence_id) Paaren
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FEVERClaim:
    """Ein Claim aus dem FEVER Dataset."""
    id: str
    claim: str
    label: str  # SUPPORTS, REFUTES, NOT ENOUGH INFO
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    evidence_texts: List[str] = field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        return self.label == "SUPPORTS"

    @property
    def is_refuted(self) -> bool:
        return self.label == "REFUTES"

    @property
    def is_nei(self) -> bool:
        return self.label == "NOT ENOUGH INFO"


class FEVERLoader:
    """
    Lädt das FEVER Dataset für Fact Verification.

    FEVER ist besser für Konsistenzprüfung als HotpotQA:
    - Explizite SUPPORTS/REFUTES Labels
    - Echte Fact Verification statt QA
    - Evidence-basiert

    Verwendung:
        loader = FEVERLoader()
        claims = loader.load(split="dev", sample_size=100)
        for claim in claims:
            print(f"{claim.claim}: {claim.label}")
    """

    def __init__(self, cache_dir: str = "data/fever"):
        self.cache_dir = cache_dir
        self._dataset = None

    def load(
        self,
        split: str = "train",
        sample_size: Optional[int] = None,
        filter_nei: bool = False
    ) -> List[FEVERClaim]:
        """
        Lädt FEVER Claims.

        Args:
            split: "train", "dev", oder "test" (test hat keine Labels)
            sample_size: Anzahl der Claims (None = alle)
            filter_nei: Wenn True, werden NOT_ENOUGH_INFO Claims gefiltert

        Returns:
            Liste von FEVERClaim Objekten
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets nicht installiert: pip install datasets")
            return []

        # FEVER Dataset laden
        logger.info(f"Lade FEVER ({split})...")

        try:
            # FEVER von Hugging Face - neues Format
            # Versuche verschiedene Dataset-IDs und Split-Namen
            dataset = None
            split_mapping = {"dev": "validation", "train": "train", "test": "test"}
            actual_split = split_mapping.get(split, split)

            for dataset_id in ["copenlu/fever_gold_evidence"]:
                try:
                    dataset = load_dataset(dataset_id, split=actual_split, cache_dir=self.cache_dir)
                    logger.info(f"  FEVER geladen von: {dataset_id} (split={actual_split})")
                    break
                except Exception as e:
                    logger.debug(f"  {dataset_id} fehlgeschlagen: {e}")
                    continue

            if dataset is None:
                raise Exception("Kein FEVER Dataset gefunden")

        except Exception as e:
            logger.warning(f"Fehler beim Laden von FEVER via HuggingFace: {e}")
            # Fallback: JSONL-Dateien wenn vorhanden
            return self._load_from_jsonl(split, sample_size, filter_nei)

        claims = []

        for i, item in enumerate(dataset):
            if sample_size and i >= sample_size:
                break

            label = item.get("label", "NOT ENOUGH INFO")

            # Filter NOT ENOUGH INFO wenn gewünscht
            if filter_nei and label == "NOT ENOUGH INFO":
                continue

            claim = FEVERClaim(
                id=str(item.get("id", i)),
                claim=item.get("claim", ""),
                label=label,
                evidence=item.get("evidence", []),
            )
            claims.append(claim)

        logger.info(f"  ✓ {len(claims)} Claims geladen")

        # Label-Verteilung
        supports = sum(1 for c in claims if c.is_supported)
        refutes = sum(1 for c in claims if c.is_refuted)
        nei = sum(1 for c in claims if c.is_nei)
        logger.info(f"  → SUPPORTS: {supports}, REFUTES: {refutes}, NEI: {nei}")

        return claims

    def _load_from_jsonl(
        self,
        split: str,
        sample_size: Optional[int],
        filter_nei: bool
    ) -> List[FEVERClaim]:
        """Fallback: Lade aus lokalen JSONL-Dateien."""
        import json
        import os

        jsonl_path = os.path.join(self.cache_dir, f"{split}.jsonl")

        if not os.path.exists(jsonl_path):
            logger.warning(f"FEVER-Datei nicht gefunden: {jsonl_path}")
            logger.info("Download von: https://fever.ai/resources.html")
            return []

        claims = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break

                item = json.loads(line)
                label = item.get("label", "NOT ENOUGH INFO")

                if filter_nei and label == "NOT ENOUGH INFO":
                    continue

                claim = FEVERClaim(
                    id=str(item.get("id", i)),
                    claim=item.get("claim", ""),
                    label=label,
                    evidence=item.get("evidence", []),
                )
                claims.append(claim)

        logger.info(f"  ✓ {len(claims)} Claims aus JSONL geladen")
        return claims

    def convert_to_triples(
        self,
        claims: List[FEVERClaim]
    ) -> List[Dict[str, Any]]:
        """
        Konvertiert FEVER Claims zu Triple-Format für Konsistenzprüfung.

        FEVER Mapping zu Konsistenz:
        - SUPPORTS → Triple sollte ACCEPTED werden
        - REFUTES → Triple sollte REJECTED werden
        - NOT ENOUGH INFO → Unsicher

        Args:
            claims: Liste von FEVERClaim Objekten

        Returns:
            Liste von Triple-Dictionaries mit Ground Truth
        """
        from src.models.entities import Entity, EntityType, Triple

        triples_data = []

        for claim in claims:
            # Einfache Heuristik: Extrahiere Subject und Predicate aus Claim
            # In der Praxis würde man NLP verwenden
            words = claim.claim.split()

            if len(words) < 3:
                continue

            # Vereinfachte Extraktion
            subject_name = words[0]
            if words[0].lower() in ["the", "a", "an"]:
                subject_name = " ".join(words[:2])

            subject = Entity(
                name=subject_name,
                entity_type=EntityType.CONCEPT,
                source_document="fever"
            )

            # Object: Rest des Satzes
            object_text = " ".join(words[-3:]) if len(words) > 3 else words[-1]
            obj = Entity(
                name=object_text,
                entity_type=EntityType.CONCEPT,
                source_document="fever"
            )

            # Evidence-Text für Source Verification
            # Bei SUPPORTS: Evidence unterstützt den Claim
            # Bei REFUTES: Evidence widerspricht dem Claim
            evidence_text = ""
            if claim.evidence and len(claim.evidence) > 0:
                # Evidence Format: [[doc_id, sent_id, text], ...]
                for ev in claim.evidence:
                    if isinstance(ev, list) and len(ev) >= 3:
                        evidence_text = ev[2]  # Der eigentliche Evidence-Text
                        break

            # Fallback auf Claim wenn keine Evidence verfügbar
            if not evidence_text:
                evidence_text = claim.claim

            triple = Triple(
                subject=subject,
                predicate="CLAIMS",
                object=obj,
                source_text=evidence_text,  # Evidence statt Claim!
                source_document_id=claim.id,
                extraction_confidence=0.8
            )

            # Ground Truth basierend auf FEVER Label
            should_accept = claim.is_supported  # SUPPORTS → Accept
            is_contradiction = claim.is_refuted  # REFUTES → Contradiction

            triples_data.append({
                "triple": triple,
                "claim": claim,
                "ground_truth_accept": should_accept,
                "is_contradiction": is_contradiction,
                "is_nei": claim.is_nei,
            })

        return triples_data

    def get_statistics(self, claims: List[FEVERClaim]) -> Dict[str, Any]:
        """Gibt Statistiken über die geladenen Claims zurück."""
        if not claims:
            return {"total": 0}

        return {
            "total": len(claims),
            "supports": sum(1 for c in claims if c.is_supported),
            "refutes": sum(1 for c in claims if c.is_refuted),
            "nei": sum(1 for c in claims if c.is_nei),
            "avg_claim_length": sum(len(c.claim.split()) for c in claims) / len(claims),
            "with_evidence": sum(1 for c in claims if c.evidence),
        }

    def extract_enhanced_triples(
        self,
        claims: List[FEVERClaim],
        use_nli_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Erweiterte Triple-Extraktion für Streaming Evaluation.

        KRITISCH: Keine Kardinalitätsregeln für FEVER!
        → Relationen wie CLAIMS, STATES haben keine max=1 Beschränkung
        → Widersprüche müssen über NLI erkannt werden

        Args:
            claims: Liste von FEVERClaim Objekten
            use_nli_labels: Wenn True, werden NLI-Labels für Ground Truth verwendet

        Returns:
            Liste von Triple-Dictionaries mit erweiterten Metadaten
        """
        import re
        from src.models.entities import Entity, EntityType, Triple

        triples_data = []

        for claim in claims:
            # Parse Claim mit verbesserter Heuristik
            parsed = self._parse_claim_enhanced(claim.claim)

            if not parsed:
                continue

            subject_text, predicate, object_text = parsed

            # Erstelle Entities
            subject = Entity(
                name=subject_text,
                entity_type=self._infer_entity_type(subject_text),
                source_document=f"fever_{claim.id}"
            )

            obj = Entity(
                name=object_text,
                entity_type=self._infer_entity_type(object_text),
                source_document=f"fever_{claim.id}"
            )

            # Evidence-Text extrahieren
            evidence_text = self._extract_evidence_text(claim)

            # Triple erstellen mit generischer Relation (KEINE Kardinalität!)
            triple = Triple(
                subject=subject,
                predicate=predicate,  # CLAIMS, STATES, etc. - keine Kardinalität!
                object=obj,
                source_text=evidence_text if evidence_text else claim.claim,
                source_document_id=claim.id,
                extraction_confidence=0.85
            )

            # Ground Truth
            if use_nli_labels:
                should_accept = claim.is_supported
                is_contradiction = claim.is_refuted
                nli_label = "ENTAILMENT" if claim.is_supported else (
                    "CONTRADICTION" if claim.is_refuted else "NEUTRAL"
                )
            else:
                should_accept = True  # Default: akzeptieren
                is_contradiction = False
                nli_label = "UNKNOWN"

            triples_data.append({
                "triple": triple,
                "claim": claim,
                "ground_truth_accept": should_accept,
                "is_contradiction": is_contradiction,
                "is_nei": claim.is_nei,
                "nli_label": nli_label,
                "evidence_text": evidence_text,
                "category": "supports" if claim.is_supported else (
                    "refutes" if claim.is_refuted else "nei"
                ),
            })

        return triples_data

    def _parse_claim_enhanced(self, claim: str) -> Optional[tuple]:
        """
        Verbesserte Claim-Parsing-Heuristik.

        Strategien:
        1. Kapitalisierte Phrasen als Subject
        2. Kopulaverben als Prädikat-Hinweis
        3. Restliche Phrase als Object

        Returns:
            Tuple von (subject, predicate, object) oder None
        """
        import re

        if not claim or len(claim.split()) < 3:
            return None

        words = claim.split()

        # Strategie 1: Finde das erste Verb als Trenner
        verbs = ["is", "was", "are", "were", "has", "have", "had",
                 "became", "becomes", "plays", "played", "stars", "starred",
                 "directed", "wrote", "born", "died", "founded", "created"]

        verb_idx = -1
        verb_found = ""
        for i, word in enumerate(words):
            if word.lower() in verbs:
                verb_idx = i
                verb_found = word.upper()
                break

        if verb_idx > 0:
            subject_text = " ".join(words[:verb_idx]).strip()
            object_text = " ".join(words[verb_idx + 1:]).strip()

            # Bereinige
            subject_text = re.sub(r'^(The|A|An)\s+', '', subject_text, flags=re.IGNORECASE)
            object_text = re.sub(r'[.,!?;:]+$', '', object_text)

            if len(subject_text) >= 2 and len(object_text) >= 2:
                return (subject_text, verb_found, object_text)

        # Strategie 2: Kapitalisierte Wörter am Anfang als Subject
        subject_words = []
        i = 0
        while i < len(words):
            word = words[i]
            # Artikel überspringen
            if word.lower() in ["the", "a", "an"]:
                i += 1
                continue
            # Großgeschriebene Wörter sammeln
            if word[0].isupper() or i == 0:
                subject_words.append(word)
                i += 1
            else:
                break

        if len(subject_words) >= 1 and i < len(words):
            subject_text = " ".join(subject_words)
            object_text = " ".join(words[i:])
            object_text = re.sub(r'[.,!?;:]+$', '', object_text)

            if len(object_text) >= 2:
                return (subject_text, "STATES", object_text)

        # Fallback: Erste Hälfte als Subject, zweite als Object
        mid = len(words) // 2
        if mid >= 1:
            subject_text = " ".join(words[:mid])
            object_text = " ".join(words[mid:])
            object_text = re.sub(r'[.,!?;:]+$', '', object_text)
            return (subject_text, "CLAIMS", object_text)

        return None

    def _infer_entity_type(self, name: str) -> 'EntityType':
        """Inferiert den Entity-Typ aus dem Namen."""
        from src.models.entities import EntityType

        name_lower = name.lower()

        # Location Keywords
        if any(kw in name_lower for kw in [
            "city", "country", "state", "province", "river", "mountain",
            "lake", "ocean", "island", "continent", "valley", "street"
        ]):
            return EntityType.LOCATION

        # Organization Keywords
        if any(kw in name_lower for kw in [
            "university", "company", "inc", "ltd", "corp", "school",
            "band", "team", "organization", "association", "institute",
            "foundation", "agency", "department", "studio", "records"
        ]):
            return EntityType.ORGANIZATION

        # Event Keywords
        if any(kw in name_lower for kw in [
            "war", "battle", "championship", "election", "festival",
            "award", "ceremony", "tournament", "olympics", "concert", "tour"
        ]):
            return EntityType.EVENT

        # Concept Keywords
        if any(kw in name_lower for kw in [
            "film", "movie", "album", "song", "book", "novel", "series",
            "theory", "law", "principle", "concept", "genre", "style"
        ]):
            return EntityType.CONCEPT

        # Default: PERSON
        return EntityType.PERSON

    def _extract_evidence_text(self, claim: FEVERClaim) -> str:
        """Extrahiert Evidence-Text aus einem Claim."""
        if not claim.evidence:
            return ""

        for ev in claim.evidence:
            if isinstance(ev, list) and len(ev) >= 3:
                return str(ev[2])
            elif isinstance(ev, dict):
                text = ev.get("text", "")
                if text:
                    return text

        return ""

    def load_for_streaming(
        self,
        split: str = "dev",
        sample_size: Optional[int] = 200,
        balance_labels: bool = True
    ) -> List[FEVERClaim]:
        """
        Lädt FEVER Claims optimiert für Streaming Evaluation.

        Args:
            split: Dataset-Split
            sample_size: Anzahl Claims pro Label-Kategorie
            balance_labels: Wenn True, gleiche Anzahl SUPPORTS/REFUTES

        Returns:
            Ausbalancierte Liste von Claims
        """
        # Lade alle Claims (ohne NEI)
        all_claims = self.load(
            split=split,
            sample_size=sample_size * 3 if sample_size else None,  # Mehr laden für Balance
            filter_nei=True
        )

        if not balance_labels:
            return all_claims[:sample_size] if sample_size else all_claims

        # Balanciere SUPPORTS und REFUTES
        supports = [c for c in all_claims if c.is_supported]
        refutes = [c for c in all_claims if c.is_refuted]

        # Gleiche Anzahl von beiden
        n_each = min(len(supports), len(refutes))
        if sample_size:
            n_each = min(n_each, sample_size // 2)

        import random
        random.seed(42)  # Reproduzierbar
        balanced = random.sample(supports, n_each) + random.sample(refutes, n_each)
        random.shuffle(balanced)

        logger.info(f"Balanced: {n_each} SUPPORTS + {n_each} REFUTES = {len(balanced)} Claims")
        return balanced


# ===========================================================================
# CLI Interface
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FEVER Dataset Loader")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--filter-nei", action="store_true")

    args = parser.parse_args()

    loader = FEVERLoader()
    claims = loader.load(
        split=args.split,
        sample_size=args.sample_size,
        filter_nei=args.filter_nei
    )

    stats = loader.get_statistics(claims)
    print("\nStatistiken:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nBeispiele:")
    for claim in claims[:5]:
        print(f"  [{claim.label}] {claim.claim}")
