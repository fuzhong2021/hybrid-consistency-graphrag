# McNemar + ECE: `per_example` vs `per_example`

## McNemar-Test (Phase 1.3)

Nullhypothese: beide Konfigurationen haben die gleiche Fehlerrate.

| Dataset | n (aligned) | b (A richtig, B falsch) | c (A falsch, B richtig) | Statistik | p-Wert | sig. α=0.05 |
|---|---:|---:|---:|---:|---:|:---:|
| fever | 1,293 | 22 | 314 | -292.000 | 0.0000 | ✓ |
| hotpotqa | 7,905 | 107 | 526 | -419.000 | 0.0000 | ✓ |
| musique | 10,484 | 249 | 748 | -499.000 | 0.0000 | ✓ |

## Expected Calibration Error (Phase 1.4)

Referenz: Guo et al. (2017). Niedriger = besser kalibriert.

| Dataset | Konfig | n | ECE | MCE | Brier | Qualität |
|---|---|---:|---:|---:|---:|---|
| fever | `per_example` | 1,293 | 0.3968 | 0.8390 | 0.4560 | poor |
| fever | `per_example` | 1,293 | 0.2684 | 0.3374 | 0.3171 | poor |
| hotpotqa | `per_example` | 7,905 | 0.0580 | 0.5904 | 0.1732 | good |
| hotpotqa | `per_example` | 7,905 | 0.2255 | 0.6537 | 0.2110 | poor |
| musique | `per_example` | 10,484 | 0.0706 | 0.4900 | 0.1771 | good |
| musique | `per_example` | 10,484 | 0.1640 | 0.6503 | 0.2118 | poor |
