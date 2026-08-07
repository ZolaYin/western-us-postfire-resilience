# Forest-Type Representation Comparison Across Resilience Dimensions (2026-05-17)

| Response | Variant | Split | Rows | Predictors | R2 | RMSE |
|---|---|---|---:|---:|---:|---:|
| IRI_good_pow2 | M2_evt_group_class | block | 56259 | 33 | 0.3787 | 0.1519 |
| IRI_good_pow2 | M3_evt_raw_code | block | 56259 | 217 | 0.3549 | 0.1548 |
| IRI_good_pow2 | M1_baseline_only | block | 56259 | 27 | 0.2644 | 0.1653 |
| IRI_good_pow2 | M3_evt_raw_code | random | 56259 | 217 | 0.6655 | 0.1112 |
| IRI_good_pow2 | M2_evt_group_class | random | 56259 | 33 | 0.6477 | 0.1141 |
| IRI_good_pow2 | M1_baseline_only | random | 56259 | 27 | 0.6339 | 0.1163 |
| Resistance | M2_evt_group_class | block | 133372 | 33 | 0.2960 | 0.1635 |
| Resistance | M3_evt_raw_code | block | 133372 | 217 | 0.2776 | 0.1656 |
| Resistance | M1_baseline_only | block | 133372 | 27 | 0.1977 | 0.1745 |
| Resistance | M3_evt_raw_code | random | 133372 | 217 | 0.6772 | 0.1148 |
| Resistance | M2_evt_group_class | random | 133372 | 33 | 0.6752 | 0.1152 |
| Resistance | M1_baseline_only | random | 133372 | 27 | 0.6513 | 0.1193 |
| STAB_good_pow2 | M2_evt_group_class | block | 56259 | 33 | 0.3252 | 0.0650 |
| STAB_good_pow2 | M3_evt_raw_code | block | 56259 | 217 | 0.2761 | 0.0673 |
| STAB_good_pow2 | M1_baseline_only | block | 56259 | 27 | 0.1695 | 0.0721 |
| STAB_good_pow2 | M2_evt_group_class | random | 56259 | 33 | 0.6440 | 0.0486 |
| STAB_good_pow2 | M3_evt_raw_code | random | 56259 | 217 | 0.6318 | 0.0494 |
| STAB_good_pow2 | M1_baseline_only | random | 56259 | 27 | 0.5834 | 0.0526 |
| T50 | M1_baseline_only | block | 56259 | 27 | 0.0562 | 0.4373 |
| T50 | M2_evt_group_class | block | 56259 | 33 | 0.0390 | 0.4413 |
| T50 | M3_evt_raw_code | block | 56259 | 217 | -0.1956 | 0.4922 |
| T50 | M3_evt_raw_code | random | 56259 | 217 | 0.4227 | 0.3004 |
| T50 | M2_evt_group_class | random | 56259 | 33 | 0.4101 | 0.3036 |
| T50 | M1_baseline_only | random | 56259 | 27 | 0.3648 | 0.3151 |
| T80 | M2_evt_group_class | block | 56259 | 33 | 0.2629 | 2.4171 |
| T80 | M3_evt_raw_code | block | 56259 | 217 | 0.2447 | 2.4467 |
| T80 | M1_baseline_only | block | 56259 | 27 | 0.1901 | 2.5336 |
| T80 | M3_evt_raw_code | random | 56259 | 217 | 0.5324 | 1.9179 |
| T80 | M2_evt_group_class | random | 56259 | 33 | 0.5226 | 1.9379 |
| T80 | M1_baseline_only | random | 56259 | 27 | 0.5043 | 1.9746 |

Variant definitions:
- M1_baseline_only: 27 baseline predictors, no forest-type term.
- M2_evt_group_class: baseline + 6 broad forest-type dummy columns.
- M3_evt_raw_code: baseline + 190 raw EVT code dummy columns.