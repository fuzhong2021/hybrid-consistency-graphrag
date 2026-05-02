# McNemar + ECE: `per_example` vs `per_example`

## McNemar-Test (Phase 1.3)

Nullhypothese: beide Konfigurationen haben die gleiche Fehlerrate.

| Dataset | n (aligned) | b (A richtig, B falsch) | c (A falsch, B richtig) | Statistik | p-Wert | sig. α=0.05 |
|---|---:|---:|---:|---:|---:|:---:|
| fever | 1,293 | 22 | 313 | -291.000 | 0.0000 | ✓ |
| hotpotqa | 7,905 | 107 | 526 | -419.000 | 0.0000 | ✓ |
| musique | 10,484 | 227 | 675 | -448.000 | 0.0000 | ✓ |

## Expected Calibration Error (Phase 1.4)

Referenz: Guo et al. (2017). Niedriger = besser kalibriert.

| Dataset | Konfig | n | ECE | MCE | Brier | Qualität |
|---|---|---:|---:|---:|---:|---|
| fever | `per_example` | 1,293 | 0.3968 | 0.8390 | 0.4560 | poor |
| fever | `per_example` | 1,293 | 0.2678 | 0.3374 | 0.3168 | poor |
| hotpotqa | `per_example` | 7,905 | 0.2171 | 0.5904 | 0.1914 | poor |
| hotpotqa | `per_example` | 7,905 | 0.3206 | 0.6526 | 0.2438 | poor |
| musique | `per_example` | 10,484 | 0.1895 | 0.6238 | 0.1920 | poor |
| musique | `per_example` | 10,484 | 0.2851 | 0.6662 | 0.2361 | poor |
