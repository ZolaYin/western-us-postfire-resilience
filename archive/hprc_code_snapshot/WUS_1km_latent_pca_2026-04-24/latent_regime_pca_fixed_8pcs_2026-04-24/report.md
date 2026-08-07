# Latent Environmental Regime Discovery - PCA baseline (2026-04-24)

**Input:** `westernus_current_candidate_table_plus_regions.parquet`  
**Rows used:** 121,739  
**Features:** 20  
**PCA mode:** `fixed`  
**PCs retained:** 8

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

## 2. Clustering metrics

| Method | k | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|---|
| gmm | 4 | 0.1303 | 8060.01 | 2.0282 |
| gmm | 5 | 0.0884 | 8399.64 | 2.4177 |
| gmm | 6 | 0.1176 | 10363.93 | 2.2337 |
| gmm | 7 | 0.0922 | 8708.51 | 2.5157 |
| gmm | 8 | 0.1084 | 9846.82 | 2.5696 |
| kmeans | 4 | 0.1614 | 22602.81 | 1.4487 |
| kmeans | 5 | 0.1820 | 22455.30 | 1.3154 |
| kmeans | 6 | 0.1948 | 22978.28 | 1.1639 |
| kmeans | 7 | 0.1932 | 23501.92 | 1.1716 |
| kmeans | 8 | 0.1963 | 22982.64 | 1.2295 |
