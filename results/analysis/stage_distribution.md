# Stage-Verteilung & Latenz (Phase 2.4 + 4.2)

## Logdatei: `full_evaluation_with_llm.log`

### Stage-Anteile pro Datensatz

| Datensatz | Total | Stage | Anteil | Count |
|---|---:|---|---:|---:|
| hotpotqa | 27,955 | stage2_embedding | 49.9 % | 13,949 |
| hotpotqa | 27,955 | stage1_rules | 35.4 % | 9,894 |
| hotpotqa | 27,955 | stage3_llm | 8.2 % | 2,296 |
| hotpotqa | 27,955 | nli | 6.5 % | 1,816 |
| fever | 7,692 | nli | 52.1 % | 4,009 |
| fever | 7,692 | stage3_llm | 40.1 % | 3,085 |
| fever | 7,692 | stage2_embedding | 7.8 % | 598 |

### Latenz pro Stage (ms)

| Datensatz | Stage | Verdict | n | Mean | Median | Max |
|---|---|---|---:|---:|---:|---:|
| fever | nli | REJECTED | 4,009 | 289.49 | 299.3 | 548.3 |
| fever | stage2_embedding | ACCEPTED | 598 | 193.27 | 161.95 | 4184.1 |
| fever | stage3_llm | ACCEPTED | 2,907 | 5634.51 | 6101.1 | 37282.5 |
| fever | stage3_llm | REJECTED | 178 | 6733.48 | 6986.05 | 14491.6 |
| hotpotqa | nli | REJECTED | 1,816 | 1272.55 | 1181.05 | 4140.8 |
| hotpotqa | stage1_rules | REJECTED | 9,894 | 0.05 | 0.0 | 1.9 |
| hotpotqa | stage2_embedding | ACCEPTED | 13,949 | 1273.5 | 1202.3 | 4062.7 |
| hotpotqa | stage3_llm | ACCEPTED | 321 | 5007.27 | 4437.3 | 16073.0 |
| hotpotqa | stage3_llm | REJECTED | 1,975 | 21589.89 | 4988.2 | 31630206.4 |

## Logdatei: `full_evaluation_no_llm.log`

### Stage-Anteile pro Datensatz

| Datensatz | Total | Stage | Anteil | Count |
|---|---:|---|---:|---:|
| hotpotqa | 22,262 | stage2_embedding | 53.1 % | 11,829 |
| hotpotqa | 22,262 | stage1_rules | 38.3 % | 8,527 |
| hotpotqa | 22,262 | nli | 8.6 % | 1,906 |
| fever | 4,707 | nli | 85.8 % | 4,039 |
| fever | 4,707 | stage2_embedding | 14.2 % | 668 |

### Latenz pro Stage (ms)

| Datensatz | Stage | Verdict | n | Mean | Median | Max |
|---|---|---|---:|---:|---:|---:|
| fever | nli | REJECTED | 4,039 | 181.02 | 184.4 | 329.3 |
| fever | stage2_embedding | ACCEPTED | 668 | 151.71 | 144.0 | 3172.9 |
| hotpotqa | nli | REJECTED | 1,906 | 1092.76 | 1006.3 | 3592.6 |
| hotpotqa | stage1_rules | REJECTED | 8,527 | 0.05 | 0.0 | 0.2 |
| hotpotqa | stage2_embedding | ACCEPTED | 11,829 | 1214.21 | 1219.0 | 3584.4 |

## Logdatei: `musique_evaluation.log`

### Stage-Anteile pro Datensatz

| Datensatz | Total | Stage | Anteil | Count |
|---|---:|---|---:|---:|
| musique | 11,893 | stage1_rules | 49.6 % | 5,905 |
| musique | 11,893 | stage2_embedding | 28.1 % | 3,338 |
| musique | 11,893 | nli | 13.1 % | 1,558 |
| musique | 11,893 | stage3_llm | 9.2 % | 1,092 |

### Latenz pro Stage (ms)

| Datensatz | Stage | Verdict | n | Mean | Median | Max |
|---|---|---|---:|---:|---:|---:|
| musique | nli | REJECTED | 1,558 | 205.83 | 197.0 | 598.0 |
| musique | stage1_rules | REJECTED | 5,905 | 0.04 | 0.0 | 0.2 |
| musique | stage2_embedding | ACCEPTED | 3,338 | 229.66 | 217.2 | 3328.8 |
| musique | stage3_llm | ACCEPTED | 85 | 3848.95 | 3512.5 | 10359.9 |
| musique | stage3_llm | REJECTED | 1,007 | 3842.76 | 3459.0 | 10114.5 |

