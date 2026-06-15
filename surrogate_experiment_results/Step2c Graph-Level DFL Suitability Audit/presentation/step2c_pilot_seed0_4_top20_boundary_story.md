# Step2c Graph-Level DFL Suitability: Phase 4 Top20 Boundary Readout

## Scope

Phase 4 extends Phase 2 from 2stage top5 to 2stage top20 candidate-boundary diagnostics. The raw top20 candidate artifact is generated outside this audit directory and is not committed to git.

## Population

- graphs: 400

## Top20 Boundary Signal

| Feature | Helpful AUROC | Harmful AUROC | Spearman with median Delta |
| --- | ---: | ---: | ---: |
| ranking_ambiguity_top20_score | 0.739 | 0.732 | 0.063 |

## Selected Case Overlay

| Graph | median Delta pp | top5 ambiguity pct | top20 ambiguity pct | top20 within-1pct pct | top20 diversity pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| G-392.json | 24.18 | 0.43 | 0.31 | 0.71 | 0.21 |
| G-1285.json | 22.36 | 0.76 | 0.62 | 0.53 | 0.72 |
| G-1560.json | 35.37 | 0.38 | 0.44 | 0.26 | 0.57 |
| G-1169.json | 10.67 | 0.96 | 0.88 | 0.86 | 0.73 |
| G-1449.json | 9.03 | 0.92 | 0.73 | 0.91 | 0.27 |
| G-142.json | 0.00 | 0.26 | 0.79 | 0.26 | 0.83 |
| G-946.json | 0.00 | 0.40 | 0.67 | 0.53 | 0.80 |
| G-14.json | -14.99 | 0.91 | 0.88 | 0.91 | 0.76 |
| G-163.json | -14.92 | 0.84 | 0.88 | 0.92 | 0.73 |

## Report-Safe Interpretation

This is a post-training boundary diagnostic, not a topology-only rule. Use it to test whether the broader 2stage candidate landscape explains cases that top5 diagnostics under-rank.
