# Western US Soft-MoE LISA Gate Round 1 (2026-04-28)

- Input: `/scratch/user/YOUR_NETID/westernus_current_candidate_table_plus_regions.parquet`
- Response: `Resistance`
- Rows used: `133,372`
- Block size: `100 km`
- Global RF trees: `300`
- Expert RF trees: `120`
- Expert min leaf: `10`
- Gate softness: `1.0`
- Gate fit sample: `60000`
- LISA k neighbors: `20`

## Variants

- `m2_baseline`: M2 baseline RF (reference): base predictors + EVT group dummies.
- `m2_mgwr`: M2+MGWR-smooth RF (parent of all soft experts): adds MGWR-radius smooth features.
- `m2_soft3_rawsmooth`: Soft-MoE (3 experts) over M2+MGWR; gate on raw + smooth stage5b gradients (round-1 baseline).
- `m2_soft3_lisa`: Soft-MoE (3 experts) over M2+MGWR; gate on LISA values (z_i * mean-neighbor-z_i) of stage5b — spatially-aware, transfer-safe.
- `m2_soft3_rawsmooth_lisa`: Soft-MoE (3 experts) over M2+MGWR; gate on raw + smooth + LISA stage5b features combined.

## Block Ranking

```text
                variant family model_kind split   rows  train_rows  test_rows  n_features  n_gate_features smooth_mode      gate_mode  n_experts     r2   rmse  moran_i  mean_gate_entropy_test  mean_gate_max_weight_test  delta_r2_vs_family_base  delta_rmse_vs_family_base  delta_moran_vs_family_base  delta_r2_vs_parent  delta_rmse_vs_parent  delta_moran_vs_parent  delta_r2_vs_m2_baseline  delta_rmse_vs_m2_baseline  delta_moran_vs_m2_baseline
            m2_baseline     m2         rf block 133372      106387      26985          33                0        none           none          0 0.2960 0.1635   0.5148                     NaN                        NaN                   0.0000                     0.0000                      0.0000              0.0000                0.0000                 0.0000                   0.0000                     0.0000                      0.0000
                m2_mgwr     m2         rf block 133372      106387      26985          44                0        mgwr           none          0 0.2692 0.1665   0.5279                     NaN                        NaN                  -0.0268                     0.0031                      0.0131             -0.0268                0.0031                 0.0131                  -0.0268                     0.0031                      0.0131
          m2_soft3_lisa     m2    softmoe block 133372      106387      26985          44               11        mgwr           lisa          3 0.2676 0.1667   0.5280                  1.0277                     0.4378                  -0.0284                     0.0033                      0.0132             -0.0016                0.0002                 0.0001                  -0.0284                     0.0033                      0.0132
     m2_soft3_rawsmooth     m2    softmoe block 133372      106387      26985          44               22        mgwr      rawsmooth          3 0.2674 0.1668   0.5278                  0.7871                     0.6518                  -0.0287                     0.0033                      0.0130             -0.0018                0.0002                -0.0001                  -0.0287                     0.0033                      0.0130
m2_soft3_rawsmooth_lisa     m2    softmoe block 133372      106387      26985          44               33        mgwr rawsmooth_lisa          3 0.2661 0.1669   0.5287                  0.8874                     0.5891                  -0.0300                     0.0034                      0.0138             -0.0031                0.0004                 0.0008                  -0.0300                     0.0034                      0.0138
```

## Full Summary

```text
                variant family model_kind  split   rows  train_rows  test_rows  n_features  n_gate_features smooth_mode      gate_mode  n_experts     r2   rmse  moran_i  mean_gate_entropy_test  mean_gate_max_weight_test  delta_r2_vs_family_base  delta_rmse_vs_family_base  delta_moran_vs_family_base  delta_r2_vs_parent  delta_rmse_vs_parent  delta_moran_vs_parent  delta_r2_vs_m2_baseline  delta_rmse_vs_m2_baseline  delta_moran_vs_m2_baseline
            m2_baseline     m2         rf  block 133372      106387      26985          33                0        none           none          0 0.2960 0.1635   0.5148                     NaN                        NaN                   0.0000                     0.0000                      0.0000              0.0000                0.0000                 0.0000                   0.0000                     0.0000                      0.0000
                m2_mgwr     m2         rf  block 133372      106387      26985          44                0        mgwr           none          0 0.2692 0.1665   0.5279                     NaN                        NaN                  -0.0268                     0.0031                      0.0131             -0.0268                0.0031                 0.0131                  -0.0268                     0.0031                      0.0131
          m2_soft3_lisa     m2    softmoe  block 133372      106387      26985          44               11        mgwr           lisa          3 0.2676 0.1667   0.5280                  1.0277                     0.4378                  -0.0284                     0.0033                      0.0132             -0.0016                0.0002                 0.0001                  -0.0284                     0.0033                      0.0132
     m2_soft3_rawsmooth     m2    softmoe  block 133372      106387      26985          44               22        mgwr      rawsmooth          3 0.2674 0.1668   0.5278                  0.7871                     0.6518                  -0.0287                     0.0033                      0.0130             -0.0018                0.0002                -0.0001                  -0.0287                     0.0033                      0.0130
m2_soft3_rawsmooth_lisa     m2    softmoe  block 133372      106387      26985          44               33        mgwr rawsmooth_lisa          3 0.2661 0.1669   0.5287                  0.8874                     0.5891                  -0.0300                     0.0034                      0.0138             -0.0031                0.0004                 0.0008                  -0.0300                     0.0034                      0.0138
            m2_baseline     m2         rf random 133372      106697      26675          33                0        none           none          0 0.6750 0.1152   0.0825                     NaN                        NaN                   0.0000                     0.0000                      0.0000              0.0000                0.0000                 0.0000                   0.0000                     0.0000                      0.0000
                m2_mgwr     m2         rf random 133372      106697      26675          44                0        mgwr           none          0 0.6965 0.1113   0.0668                     NaN                        NaN                   0.0216                    -0.0039                     -0.0157              0.0216               -0.0039                -0.0157                   0.0216                    -0.0039                     -0.0157
          m2_soft3_lisa     m2    softmoe random 133372      106697      26675          44               11        mgwr           lisa          3 0.7242 0.1062   0.0480                  0.9896                     0.4690                   0.0492                    -0.0091                     -0.0346              0.0276               -0.0052                -0.0189                   0.0492                    -0.0091                     -0.0346
     m2_soft3_rawsmooth     m2    softmoe random 133372      106697      26675          44               22        mgwr      rawsmooth          3 0.7241 0.1062   0.0474                  0.8156                     0.6502                   0.0491                    -0.0091                     -0.0351              0.0275               -0.0052                -0.0194                   0.0491                    -0.0091                     -0.0351
m2_soft3_rawsmooth_lisa     m2    softmoe random 133372      106697      26675          44               33        mgwr rawsmooth_lisa          3 0.7240 0.1062   0.0479                  0.9063                     0.5862                   0.0491                    -0.0091                     -0.0346              0.0275               -0.0052                -0.0190                   0.0491                    -0.0091                     -0.0346
```
