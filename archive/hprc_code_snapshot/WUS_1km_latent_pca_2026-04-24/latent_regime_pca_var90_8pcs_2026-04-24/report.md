# Latent Environmental Regime Discovery - PCA baseline (2026-04-24)

**Input:** `westernus_current_candidate_table_plus_regions.parquet`  
**Rows used:** 121,739  
**Features:** 20  
**PCA mode:** `var90`  
**PCs retained:** 14

## 1. PCA explained variance

| Component | Explained variance ratio | Cumulative explained variance |
|---|---|---|
| PC1 | 0.2333 | 0.2333 |
| PC2 | 0.0990 | 0.3322 |
| PC3 | 0.0917 | 0.4239 |
| PC4 | 0.0677 | 0.4916 |
| PC5 | 0.0600 | 0.5516 |
| PC6 | 0.0539 | 0.6055 |
| PC7 | 0.0526 | 0.6581 |
| PC8 | 0.0497 | 0.7078 |
| PC9 | 0.0478 | 0.7555 |
| PC10 | 0.0434 | 0.7989 |
| PC11 | 0.0367 | 0.8356 |
| PC12 | 0.0308 | 0.8664 |
| PC13 | 0.0263 | 0.8927 |
| PC14 | 0.0246 | 0.9173 |

## 2. Clustering metrics

| Method | k | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|---|
| gmm | 4 | 0.0874 | 9107.53 | 2.7314 |
| gmm | 5 | 0.0643 | 8135.36 | 3.1443 |
| gmm | 6 | 0.0613 | 7856.36 | 2.7346 |
| gmm | 7 | 0.0968 | 8034.66 | 2.4888 |
| gmm | 8 | 0.0976 | 8800.39 | 2.3162 |
| kmeans | 4 | 0.1259 | 15889.22 | 1.8598 |
| kmeans | 5 | 0.1464 | 15319.45 | 1.7071 |
| kmeans | 6 | 0.1511 | 15737.85 | 1.4483 |
| kmeans | 7 | 0.1503 | 15823.43 | 1.4158 |
| kmeans | 8 | 0.1533 | 15185.20 | 1.5419 |
