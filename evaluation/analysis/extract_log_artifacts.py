#!/usr/bin/env python3
# evaluation/analysis/extract_log_artifacts.py
"""
Extrahiert Stage-Verteilung, Latenzen und LLM-Kosten aus Evaluations-Logs.

Verarbeitet die persistierten Logs (results/*.log) und liefert:
- Stage-Distribution pro Datensatz (Phase 2.4)
- Mittelwert-/Median-Latenz pro Stage (Phase 4.2)
- LLM-Aufruf-Häufigkeit + Token-/Kostenschätzung (Phase 4.3)

Logformat (aus orchestrator.py):
    YYYY-MM-DD HH:MM:SS,ms - INFO - ✅ ACCEPTED [Stufe N PASS/fail] in X.Yms
    YYYY-MM-DD HH:MM:SS,ms - INFO - ❌ REJECTED [Stufe N FAIL/fail] in X.Yms
    YYYY-MM-DD HH:MM:SS,ms - INFO - ❌ REJECTED [NLI CONTRADICTION] in X.Yms
    === Evaluiere DATASET ===
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis"

# Tarife für Kostenschätzung (Stand: GPT-4-Turbo Listenpreis Anfang 2025)
GPT4_INPUT_USD_PER_1K = 0.01
GPT4_OUTPUT_USD_PER_1K = 0.03
# Annahme über durchschnittliche Token-Aufteilung pro Stage-3-Aufruf
ASSUMED_INPUT_TOKENS_PER_LLM_CALL = 600
ASSUMED_OUTPUT_TOKENS_PER_LLM_CALL = 80

DATASET_HEADER = re.compile(r"=== Evaluiere ([A-Z]+) ===")
DECISION_LINE = re.compile(
    r"(?P<verdict>ACCEPTED|REJECTED) \[(?P<reason>[^\]]+)\] in (?P<latency>[\d.]+)ms"
)
TRIPLE_LINE = re.compile(r"━━━ Verarbeite: Triple\((?P<triple>.+)\) ━━━")


def parse_log(path: Path) -> Dict:
    """Parst eine Log-Datei in strukturierte Per-Decision-Records."""
    current_dataset = "_unspecified"
    decisions = []
    triples_window: List[str] = []  # letzte gesehene Triple-Strings (puffer)
    with path.open() as f:
        for line in f:
            m = DATASET_HEADER.search(line)
            if m:
                current_dataset = m.group(1).lower()
                continue
            m = TRIPLE_LINE.search(line)
            if m:
                triples_window.append(m.group("triple"))
                if len(triples_window) > 8:
                    triples_window.pop(0)
                continue
            m = DECISION_LINE.search(line)
            if m:
                triple_str = triples_window.pop(0) if triples_window else None
                decisions.append({
                    "dataset": current_dataset,
                    "verdict": m.group("verdict"),
                    "reason": m.group("reason").strip(),
                    "latency_ms": float(m.group("latency")),
                    "triple": triple_str,
                })
    return {"path": str(path), "decisions": decisions}


def stage_from_reason(reason: str) -> str:
    r = reason.lower()
    if "stufe 1" in r:
        return "stage1_rules"
    if "stufe 2" in r:
        return "stage2_embedding"
    if "stufe 3" in r:
        return "stage3_llm"
    if "nli" in r:
        return "nli"
    if "fast" in r or "high confidence" in r:
        return "fast_path"
    return "other"


def aggregate(decisions: List[Dict]) -> Dict:
    """Aggregiert Decisions nach Datensatz × Stage × Verdict."""
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    by_dataset: Dict[str, Counter] = defaultdict(Counter)
    llm_calls: Dict[str, int] = Counter()
    for d in decisions:
        stage = stage_from_reason(d["reason"])
        key = (d["dataset"], stage, d["verdict"])
        grouped[key].append(d["latency_ms"])
        by_dataset[d["dataset"]][stage] += 1
        if stage == "stage3_llm":
            llm_calls[d["dataset"]] += 1

    rows = []
    for (dataset, stage, verdict), latencies in sorted(grouped.items()):
        rows.append({
            "dataset": dataset,
            "stage": stage,
            "verdict": verdict,
            "count": len(latencies),
            "latency_ms": {
                "mean": round(mean(latencies), 2),
                "median": round(median(latencies), 2),
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
            },
        })

    distribution = {}
    for ds, counter in by_dataset.items():
        total = sum(counter.values())
        distribution[ds] = {
            "total_decisions": total,
            "by_stage": {stage: {"count": n, "share": round(n / total, 4)} for stage, n in counter.most_common()},
        }

    cost = {}
    for ds, n_calls in llm_calls.items():
        in_tokens = n_calls * ASSUMED_INPUT_TOKENS_PER_LLM_CALL
        out_tokens = n_calls * ASSUMED_OUTPUT_TOKENS_PER_LLM_CALL
        usd = (in_tokens / 1000) * GPT4_INPUT_USD_PER_1K + (out_tokens / 1000) * GPT4_OUTPUT_USD_PER_1K
        cost[ds] = {
            "llm_calls": n_calls,
            "assumed_input_tokens": in_tokens,
            "assumed_output_tokens": out_tokens,
            "estimated_gpt4_usd": round(usd, 2),
            "estimated_gpt4_usd_per_1k_triples": round(usd / max(distribution[ds]["total_decisions"], 1) * 1000, 2),
        }

    return {"rows": rows, "distribution": distribution, "cost": cost}


def render_stage_md(per_log: Dict[str, Dict]) -> str:
    lines = ["# Stage-Verteilung & Latenz (Phase 2.4 + 4.2)", ""]
    for log_name, agg in per_log.items():
        lines.append(f"## Logdatei: `{log_name}`")
        lines.append("")
        lines.append("### Stage-Anteile pro Datensatz")
        lines.append("")
        lines.append("| Datensatz | Total | Stage | Anteil | Count |")
        lines.append("|---|---:|---|---:|---:|")
        for ds, info in agg["distribution"].items():
            for stage, vals in info["by_stage"].items():
                lines.append(f"| {ds} | {info['total_decisions']:,} | {stage} | {vals['share']*100:.1f} % | {vals['count']:,} |")
        lines.append("")
        lines.append("### Latenz pro Stage (ms)")
        lines.append("")
        lines.append("| Datensatz | Stage | Verdict | n | Mean | Median | Max |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for r in agg["rows"]:
            lines.append(
                f"| {r['dataset']} | {r['stage']} | {r['verdict']} | {r['count']:,} "
                f"| {r['latency_ms']['mean']} | {r['latency_ms']['median']} | {r['latency_ms']['max']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def render_cost_md(per_log: Dict[str, Dict]) -> str:
    lines = [
        "# LLM-Kostenanalyse (Phase 4.3)",
        "",
        f"Annahmen: {ASSUMED_INPUT_TOKENS_PER_LLM_CALL} Input-Token + "
        f"{ASSUMED_OUTPUT_TOKENS_PER_LLM_CALL} Output-Token pro Stage-3-LLM-Aufruf "
        f"(GPT-4-Turbo Listenpreis: {GPT4_INPUT_USD_PER_1K} $ / 1K Input, {GPT4_OUTPUT_USD_PER_1K} $ / 1K Output).",
        "",
        "Reale Kosten der durchgeführten Evaluation = 0 USD, da Ollama lokal genutzt wurde. "
        "Die Spalten zeigen, was eine GPT-4-basierte Variante gekostet hätte.",
        "",
        "| Logdatei | Datensatz | LLM-Calls | Input-Tokens | Output-Tokens | GPT-4 USD | USD/1k Triples |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for log_name, agg in per_log.items():
        for ds, c in agg["cost"].items():
            lines.append(
                f"| `{log_name}` | {ds} | {c['llm_calls']:,} | "
                f"{c['assumed_input_tokens']:,} | {c['assumed_output_tokens']:,} | "
                f"{c['estimated_gpt4_usd']:.2f} | {c['estimated_gpt4_usd_per_1k_triples']:.2f} |"
            )
    lines.append("")
    lines.append(
        "> **Caveat:** Die Token-Annahmen sind grobe Schätzungen, da `LLMUsageStats` "
        "in den vorliegenden Logs nicht persistiert wurde. Für belastbare Token-Zahlen "
        "muss `llm_arbitrator.py` so erweitert werden, dass es `prompt_tokens` / "
        "`completion_tokens` ins Eval-Log schreibt."
    )
    return "\n".join(lines) + "\n"


def render_error_md(per_log: Dict[str, Dict[str, List[Dict]]]) -> str:
    """Sammelt Beispiele für qualitative Fehleranalyse (Phase 6)."""
    lines = [
        "# Qualitative Beispielanalyse (Phase 6)",
        "",
        "Stichproben aus den Logs ohne Ground-Truth-Verknüpfung. "
        "Daher hier *Beispielentscheidungen* nach interessanten Mustern, "
        "keine echten False-Positive/False-Negative-Klassifikationen — diese "
        "bräuchten einen instrumentierten Re-Lauf (siehe "
        "`evaluation/analysis/instrument_logging_patch.md`).",
        "",
    ]
    for log_name, examples in per_log.items():
        lines.append(f"## `{log_name}`")
        lines.append("")
        for category, items in examples.items():
            lines.append(f"### {category}")
            lines.append("")
            if not items:
                lines.append("_keine Beispiele in diesem Log_")
                lines.append("")
                continue
            for ex in items:
                triple = ex.get("triple", "<unknown triple>")
                lines.append(
                    f"- **{ex['verdict']}** [{ex['reason']}] in {ex['latency_ms']} ms "
                    f"(Datensatz: `{ex['dataset']}`)\n  - Triple: `{triple}`"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def collect_examples(decisions: List[Dict]) -> Dict[str, List[Dict]]:
    """Wählt repräsentative Decision-Beispiele für Phase 6."""
    rng_picks = {
        "Stage-3 LLM-REJECT (möglicher False Negative)": [],
        "Stage-3 LLM-ACCEPT nach Stage-2 Fast-Path-Bypass": [],
        "Stage-1 Fast-Reject (Schema/Cardinality)": [],
        "NLI Contradiction": [],
        "Hohe Latenz (>5000 ms)": [],
    }
    for d in decisions:
        stage = stage_from_reason(d["reason"])
        if d["verdict"] == "REJECTED" and stage == "stage3_llm" and len(rng_picks["Stage-3 LLM-REJECT (möglicher False Negative)"]) < 5:
            rng_picks["Stage-3 LLM-REJECT (möglicher False Negative)"].append(d)
        elif d["verdict"] == "ACCEPTED" and stage == "stage3_llm" and len(rng_picks["Stage-3 LLM-ACCEPT nach Stage-2 Fast-Path-Bypass"]) < 5:
            rng_picks["Stage-3 LLM-ACCEPT nach Stage-2 Fast-Path-Bypass"].append(d)
        elif d["verdict"] == "REJECTED" and stage == "stage1_rules" and len(rng_picks["Stage-1 Fast-Reject (Schema/Cardinality)"]) < 5:
            rng_picks["Stage-1 Fast-Reject (Schema/Cardinality)"].append(d)
        elif "nli" in d["reason"].lower() and len(rng_picks["NLI Contradiction"]) < 5:
            rng_picks["NLI Contradiction"].append(d)
        elif d["latency_ms"] > 5000 and len(rng_picks["Hohe Latenz (>5000 ms)"]) < 5:
            rng_picks["Hohe Latenz (>5000 ms)"].append(d)
    return rng_picks


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_files = [
        RESULTS_DIR / "full_evaluation_with_llm.log",
        RESULTS_DIR / "full_evaluation_no_llm.log",
        RESULTS_DIR / "musique_evaluation.log",
    ]

    per_log_agg: Dict[str, Dict] = {}
    per_log_examples: Dict[str, Dict[str, List[Dict]]] = {}

    for log_path in log_files:
        if not log_path.exists():
            continue
        parsed = parse_log(log_path)
        agg = aggregate(parsed["decisions"])
        per_log_agg[log_path.name] = agg
        per_log_examples[log_path.name] = collect_examples(parsed["decisions"])

    (OUTPUT_DIR / "stage_distribution.json").write_text(json.dumps(per_log_agg, indent=2, ensure_ascii=False))
    (OUTPUT_DIR / "stage_distribution.md").write_text(render_stage_md(per_log_agg))
    (OUTPUT_DIR / "cost_analysis.md").write_text(render_cost_md(per_log_agg))
    (OUTPUT_DIR / "error_examples.md").write_text(render_error_md(per_log_examples))

    print(f"Wrote analyses for {len(per_log_agg)} log files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
