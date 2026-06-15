# Step2c Graph-Level DFL Suitability: Phase 3 Matched Controls

## Scope

Phase 3 matches each README target graph to nearest heldout controls using only the six raw-topology variables specified in the protocol.
Each target uses up to 20 controls, excluding the README target set from the control pool.

## Match Variables

```text
num_vertices
num_arcs
density
num_2cycles
num_3cycles
largest_scc_fraction
```

## Target Vs Matched Controls

| Target | Group | Delta | Matched delta median | Delta pct | Ambiguity pct | Richness pct | Closest control |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| G-392.json | helpful_success | 24.18 | 0.00 | 1.00 | 0.25 | 0.30 | G-1442.json |
| G-1285.json | helpful_success | 22.36 | 0.00 | 1.00 | 0.90 | 0.25 | G-1040.json |
| G-1560.json | helpful_success | 35.37 | 0.00 | 1.00 | 0.30 | 0.05 | G-542.json |
| G-1169.json | helpful_success | 10.67 | 0.00 | 0.95 | 1.00 | 0.05 | G-1375.json |
| G-1449.json | helpful_success | 9.03 | 0.00 | 1.00 | 0.90 | 0.55 | G-546.json |
| G-142.json | both_poor_control | 0.00 | 0.00 | 0.60 | 0.15 | 0.45 | G-929.json |
| G-946.json | both_poor_control | 0.00 | 0.00 | 0.70 | 0.45 | 0.85 | G-313.json |
| G-14.json | harmful_reranking_control | -14.99 | 0.53 | 0.00 | 0.80 | 0.15 | G-1594.json |
| G-163.json | harmful_reranking_control | -14.92 | 0.00 | 0.00 | 0.85 | 0.45 | G-1052.json |

## Report-Safe Interpretation

This is a matched-control association audit. Matching is intentionally restricted to simple raw topology, then outcomes and higher-level feasible-set / prediction-boundary diagnostics are compared after matching.
A target that remains extreme relative to matched controls supports graph-instance specificity beyond the coarse topology variables used for matching. It still does not prove topology causality.
