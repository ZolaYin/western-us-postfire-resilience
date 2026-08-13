# Retained results

This directory contains compact outputs needed to verify the manuscript workflow without rerunning every high-performance-computing job:

- `rf/`: final forest-type representation, random-versus-spatial validation metrics, the regularized-RF sensitivity check, and the 50/100/200 km block-size sensitivity check;
- `ols/`: global reference-model diagnostics and residual table;
- `mgwr/`: retained Resistance, integrated-recovery, and stability coefficient tables, bandwidths, and performance summary;
- `zoning/`: EPA Level III management-zone tables, GeoPackage, thresholds, and run metadata.

The larger derived zoning intermediate is available in the [public Google Drive data release](https://drive.google.com/drive/folders/1C1kPp0hS7RW5zTaVD0c7O88LxmNuJ3wk). Raw source rasters, temporary arrays, logs, environments, and superseded exploratory outputs are not repackaged. The complete historical code inventory is retained under `../archive/`.
