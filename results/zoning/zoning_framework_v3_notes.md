# MGWR constraint zoning framework v3

## Main choices

- Effect definition: realized local effect, `beta_j,r * X_j / sd(Y_r)`. This means the scores represent realized local contribution/constraint intensity, not coefficient strength alone.
- Main zoning is constraint-focused: dominance is based on negative realized effects for all mechanism groups.
- Forest structure handling: support and constraint are retained separately for diagnostics, but dominance uses `forest_structure_constraint`, consistent with climate and human-pressure constraint scores.
- Resilience priority: `R_comp_equal` is the equal-weight mean of ranked Resistance, IRI_good_pow2, and STAB_good_pow2.
- Main dominance threshold: q75 of `dominant_score`; q70 and q80 are reported as sensitivity tests.
- RF or MGWR-informed RF predictions are not used in this zoning run. The map is mechanism-based.

## Key diagnostics

- EPA L3 ecoregions retained: 26.
- Candidate pixels represented: 56129.
- q75 dominant-score threshold: 0.3447.
- q75 mixed-control share in mechanism map: 69.3% of candidate pixels.
- q75 mixed-control share after resilience-priority overlay: 34.1% of candidate pixels.
- PCA PC1 explained variance: 0.866.
- Equal composite vs PCA rank correlation: 0.982.
- Equal vs PCA priority-class agreement: 100.0% of L3 ecoregions.
- K-means four-cluster largest class: Moderate structure-constraint cluster (43.5% of candidate pixels).

## Interpretation guardrails

- MGWR-derived zones should be described as statistical dominant-factor or realized-effect zones, not as causal proof.
- If q75 leaves too much area in the mixed zone, use the sensitivity table to justify q70 or use the K-means cluster map as a data-driven constraint-space diagnostic.
- The equal-weight resilience composite is transparent and interpretable; the PCA check is a sensitivity diagnostic rather than the main map.
