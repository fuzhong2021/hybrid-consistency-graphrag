# LLM-Kostenanalyse (Phase 4.3)

Annahmen: 600 Input-Token + 80 Output-Token pro Stage-3-LLM-Aufruf (GPT-4-Turbo Listenpreis: 0.01 $ / 1K Input, 0.03 $ / 1K Output).

Reale Kosten der durchgeführten Evaluation = 0 USD, da Ollama lokal genutzt wurde. Die Spalten zeigen, was eine GPT-4-basierte Variante gekostet hätte.

| Logdatei | Datensatz | LLM-Calls | Input-Tokens | Output-Tokens | GPT-4 USD | USD/1k Triples |
|---|---|---:|---:|---:|---:|---:|
| `full_evaluation_with_llm.log` | hotpotqa | 2,296 | 1,377,600 | 183,680 | 19.29 | 0.69 |
| `full_evaluation_with_llm.log` | fever | 3,085 | 1,851,000 | 246,800 | 25.91 | 3.37 |
| `musique_evaluation.log` | musique | 1,092 | 655,200 | 87,360 | 9.17 | 0.77 |

> **Caveat:** Die Token-Annahmen sind grobe Schätzungen, da `LLMUsageStats` in den vorliegenden Logs nicht persistiert wurde. Für belastbare Token-Zahlen muss `llm_arbitrator.py` so erweitert werden, dass es `prompt_tokens` / `completion_tokens` ins Eval-Log schreibt.
