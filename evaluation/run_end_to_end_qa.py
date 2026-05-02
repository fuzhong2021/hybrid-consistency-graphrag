#!/usr/bin/env python3
# evaluation/run_end_to_end_qa.py
"""
Phase 3.1: End-to-End-QA mit/ohne Konsistenzmodul.

Vergleicht die Antwortqualität (Exact-Match + Token-F1) der GraphRAG-
Pipeline zwischen:
  Variante A: Konsistenzmodul aktiv (vollständige 4-Stufen-Kaskade)
  Variante B: alle extrahierten Triples ungeprüft akzeptiert

Ablauf pro Frage:
  1. Kontext-Paragraphen aus HotpotQA-Beispiel in die Pipeline ingesten
     (Triple-Extraktion → Konsistenzprüfung → Graph-Aufbau)
  2. Frage über Pipeline beantworten lassen
  3. Antwort gegen Gold-Antwort vergleichen (EM + F1)

Voraussetzungen:
  - Ollama läuft mit llama3.1:8b (für Extraktion + Antwort-Generierung)
  - sentence-transformers installiert (all-MiniLM-L6-v2)
  - GPU empfohlen (--gpu)

Nutzung:
  python evaluation/run_end_to_end_qa.py --num-questions 200 --gpu
"""

import argparse
import json
import logging
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, ".")

from src.evaluation.benchmark_loader import BenchmarkLoader, QAExample
from src.pipeline.graphrag_pipeline import GraphRAGPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Antwort-Metriken (SQuAD-Stil: Exact Match + Token-F1)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Pipeline-Factory & Backends
# ---------------------------------------------------------------------------

def build_pipeline(enable_consistency: bool, llm_client: Any, embedding_model: Any,
                   llm_model: str = "llama3.1:8b") -> GraphRAGPipeline:
    cfg = PipelineConfig(
        enable_consistency=enable_consistency,
        extraction_model=llm_model,
        answer_model=llm_model,
    )
    return GraphRAGPipeline(
        config=cfg,
        llm_client=llm_client,
        embedding_model=embedding_model,
    )


def _ingest_context(pipeline: GraphRAGPipeline, example: QAExample) -> Dict[str, int]:
    """Ingesten der HotpotQA-context_paragraphs ins Pipeline-Graphstore."""
    stats = {"chunks": 0, "triples_accepted": 0, "triples_rejected": 0}
    for idx, para in enumerate(example.context_paragraphs):
        # HotpotQA-Loader liefert oft Dicts mit "title" + "sentences" ODER roh-Strings
        if isinstance(para, dict):
            sentences = para.get("sentences") or []
            text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            title = para.get("title", f"para_{idx}")
        else:
            text = str(para)
            title = f"para_{idx}"
        if not text.strip():
            continue
        try:
            ingest_result = pipeline.ingest(text=text, document_id=f"{example.id}__{title}")
            stats["chunks"] += 1
            stats["triples_accepted"] += getattr(ingest_result, "triples_accepted", 0)
            stats["triples_rejected"] += getattr(ingest_result, "triples_rejected", 0)
        except Exception as exc:
            logger.debug(f"Ingest fehlgeschlagen für {example.id}/{title}: {exc}")
    return stats


def run_variant(label: str, enable_consistency: bool, questions: List[QAExample],
                llm_client: Any, embedding_model: Any, output_dir: Path,
                llm_model: str = "llama3.1:8b") -> Dict:
    logger.info(f"\n=== Variante: {label} (consistency={enable_consistency}) ===")
    per_question_path = output_dir / f"{label}_per_question.jsonl"
    per_question_path.parent.mkdir(parents=True, exist_ok=True)
    per_q_file = per_question_path.open("w")

    em_total, f1_total = 0.0, 0.0
    n_completed = 0
    start = time.time()

    for q_idx, q in enumerate(questions):
        pipeline = build_pipeline(enable_consistency, llm_client, embedding_model, llm_model)
        t0 = time.time()
        try:
            ingest_stats = _ingest_context(pipeline, q)
            pr = pipeline.query(q.question)
            predicted = pr.answer if pr and pr.answer else ""
        except Exception as exc:
            logger.warning(f"[{q_idx}] Frage '{q.id}' fehlgeschlagen: {exc}")
            predicted = ""
            ingest_stats = {"chunks": 0, "triples_accepted": 0, "triples_rejected": 0}

        em = exact_match(predicted, q.answer)
        f1 = token_f1(predicted, q.answer)
        em_total += em
        f1_total += f1
        n_completed += 1

        per_q_file.write(json.dumps({
            "question_id": q.id,
            "question": q.question,
            "gold": q.answer,
            "predicted": predicted,
            "exact_match": em,
            "f1": f1,
            "latency_s": round(time.time() - t0, 2),
            "ingest_stats": ingest_stats,
        }) + "\n")
        per_q_file.flush()

        if (q_idx + 1) % 10 == 0:
            logger.info(f"  {q_idx + 1}/{len(questions)}  running EM={em_total / n_completed:.3f}  F1={f1_total / n_completed:.3f}")

    per_q_file.close()
    elapsed = time.time() - start
    n = max(n_completed, 1)

    return {
        "variant": label,
        "enable_consistency": enable_consistency,
        "n": n_completed,
        "exact_match": em_total / n,
        "f1": f1_total / n,
        "wallclock_s": round(elapsed, 2),
        "per_question_file": str(per_question_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--output-dir", default="results/end_to_end_qa")
    ap.add_argument("--llm-model", default="llama3.1:8b")
    args = ap.parse_args()

    # Embedding-Modell
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info(f"Embedding-Modell auf {device}")
    except Exception as exc:
        logger.warning(f"Kein Embedding-Modell: {exc}")

    # LLM
    llm_client = None
    try:
        from src.llm.ollama_client import OllamaClient
        llm_client = OllamaClient(model=args.llm_model)
        logger.info(f"LLM-Client: {args.llm_model}")
    except Exception as exc:
        logger.error(f"Kein LLM-Client — End-to-End-QA kann nicht laufen: {exc}")
        sys.exit(1)

    # Daten laden (korrekter Kwarg: sample_size)
    loader = BenchmarkLoader()
    questions = loader.load_hotpotqa(split="validation", sample_size=args.num_questions)
    logger.info(f"Geladen: {len(questions)} HotpotQA-Fragen")
    if not questions:
        logger.error("Keine Fragen geladen — Abbruch.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_questions": len(questions),
        "llm_model": args.llm_model,
        "with_consistency": run_variant("with_consistency", True, questions,
                                        llm_client, embedding_model, output_dir,
                                        args.llm_model),
        "without_consistency": run_variant("without_consistency", False, questions,
                                           llm_client, embedding_model, output_dir,
                                           args.llm_model),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== End-to-End-QA Zusammenfassung ===")
    for key in ("with_consistency", "without_consistency"):
        v = summary[key]
        print(f"  {key:25s}  EM={v['exact_match']:.3f}  F1={v['f1']:.3f}  ({v['n']} Fragen)")
    print(f"\nErgebnisdatei: {summary_path}")


if __name__ == "__main__":
    main()
