# Step2c Graph-Level DFL Suitability: Phase 4 Top20 Boundary Readout

## Scope

Phase 4 extends Phase 2 from 2stage top5 to 2stage top20 candidate-boundary diagnostics. The raw top20 candidate artifact is generated outside this audit directory and is not committed to git.

## Population

- joined graphs: 400
- top20-covered graphs: 160

## Top20 Boundary Signal

| Feature | Helpful AUROC | Harmful AUROC | Spearman with median Delta |
| --- | ---: | ---: | ---: |
| mean_2stage_top20_within_1pct_count | 0.752 | 0.601 | 0.134 |

## Selected Case Overlay

| Graph | median Delta pp | top5 ambiguity pct | top20 ambiguity pct | top20 within-1pct pct | top20 diversity pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| G-392.json | 24.18 | 0.43 | 0.30 | 0.64 | 0.24 |
| G-1285.json | 22.36 | 0.76 | 0.40 | 0.44 | 0.71 |
| G-1560.json | 35.37 | 0.38 | 0.38 | 0.16 | 0.57 |
| G-1169.json | 10.67 | 0.96 | 0.88 | 0.80 | 0.73 |
| G-1449.json | 9.03 | 0.92 | 0.68 | 0.88 | 0.29 |
| G-142.json | 0.00 | 0.26 | 0.73 | 0.16 | 0.86 |
| G-946.json | 0.00 | 0.40 | 0.50 | 0.44 | 0.84 |
| G-14.json | -14.99 | 0.91 | 0.87 | 0.85 | 0.76 |
| G-163.json | -14.92 | 0.84 | 0.85 | 0.89 | 0.72 |

## Report-Safe Interpretation

This is a post-training boundary diagnostic, not a topology-only rule. Use it to test whether the broader 2stage candidate landscape explains cases that top5 diagnostics under-rank.
