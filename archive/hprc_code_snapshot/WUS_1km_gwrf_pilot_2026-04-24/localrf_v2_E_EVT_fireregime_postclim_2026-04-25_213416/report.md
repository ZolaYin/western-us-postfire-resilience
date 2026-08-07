# Local RF Bandwidth Sweep v2 (2026-04-25)

This is a more stable GW-RF-related local RF pilot, not a full multiscale GW-RF implementation.

| Model | Variant | XY mode | k_spatial | Split | Predictors | R2 | RMSE | Moran's I |
|---|---|---|---|---|---|---|---|---|
| global_rf | E_EVT_fireregime_postclim | with_xy | NA | block | 34 | 0.3574 | 0.1633 | 0.5396 |
| local_rf | E_EVT_fireregime_postclim | with_xy | 2000 | block | 34 | 0.2922 | 0.1714 | 0.5699 |
| local_rf | E_EVT_fireregime_postclim | with_xy | 5000 | block | 34 | 0.3120 | 0.1690 | 0.5627 |
| local_rf | E_EVT_fireregime_postclim | with_xy | 10000 | block | 34 | 0.3193 | 0.1681 | 0.5592 |
| global_rf | E_EVT_fireregime_postclim | with_xy | NA | random | 34 | 0.6329 | 0.1223 | 0.1001 |
| global_rf | E_EVT_fireregime_postclim | without_xy | NA | block | 29 | 0.3463 | 0.1647 | 0.5434 |
| local_rf | E_EVT_fireregime_postclim | without_xy | 2000 | block | 29 | 0.2975 | 0.1707 | 0.5607 |
| local_rf | E_EVT_fireregime_postclim | without_xy | 5000 | block | 29 | 0.3328 | 0.1664 | 0.5466 |
| local_rf | E_EVT_fireregime_postclim | without_xy | 10000 | block | 29 | 0.3405 | 0.1654 | 0.5416 |
| global_rf | E_EVT_fireregime_postclim | without_xy | NA | random | 29 | 0.5749 | 0.1316 | 0.1530 |
