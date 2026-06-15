# Step2c Graph-Level DFL Suitability: Phase 2 Readout

## Scope

Phase 2 adds prediction-boundary features from existing all-400 2stage top5 candidate lists. No all-400 top20 rerun is used in this phase.

## Population

- graphs: 400

## Best Spearman Association By Feature Family

| Family | Feature | Spearman with median Delta | AUROC helpful | AUROC harmful |
| --- | --- | ---: | ---: | ---: |
| raw_topology | in_degree_gini | -0.111 | 0.355 | 0.274 |
| cycle_chain | cycle_to_chain_ratio | 0.038 | 0.559 | 0.679 |
| exchange_geometry | vertex_exchange_participation_gini | -0.055 | 0.455 | 0.437 |
| conflict_geometry | conflict_graph_density | -0.033 | 0.559 | 0.608 |
| prediction_boundary | median_2stage_top1_top5_pred_margin | -0.162 | 0.255 | 0.384 |

## Best Helpful / Harmful AUROC By Feature Family

| Family | Best helpful feature | Helpful AUROC | Best harmful feature | Harmful AUROC |
| --- | --- | ---: | --- | ---: |
| raw_topology | density | 0.630 | max_out_degree | 0.815 |
| cycle_chain | num_3cycles | 0.606 | cycle_to_chain_ratio | 0.679 |
| exchange_geometry | exchange_size_mean | 0.565 | exchange_size_mean | 0.622 |
| conflict_geometry | conflict_graph_density | 0.559 | conflict_graph_density | 0.608 |
| prediction_boundary | ranking_ambiguity_score | 0.715 | mean_2stage_top5_within_1pct_count | 0.587 |

## Selected Case Overlay

| Graph | median Delta pp | ambiguity pct | top1-top2 margin pct percentile | within-1pct count pct | diversity pct | modal rank1 rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G-392.json | 24.18 | 0.43 | 0.26 | 0.71 | 0.02 | 1.00 |
| G-1285.json | 22.36 | 0.76 | 0.48 | 0.52 | 0.96 | 1.00 |
| G-1560.json | 35.37 | 0.38 | 0.87 | 0.25 | 0.87 | 1.00 |
| G-1169.json | 10.67 | 0.96 | 0.28 | 0.85 | 0.95 | 1.00 |
| G-1449.json | 9.03 | 0.92 | 0.15 | 1.00 | 0.65 | 1.00 |
| G-142.json | 0.00 | 0.26 | 0.90 | 0.25 | 0.77 | 1.00 |
| G-946.json | 0.00 | 0.40 | 0.71 | 0.52 | 0.53 | 1.00 |
| G-14.json | -14.99 | 0.91 | 0.05 | 0.87 | 0.59 | 0.66 |
| G-163.json | -14.92 | 0.84 | 0.50 | 1.00 | 0.49 | 1.00 |

## Report-Safe Interpretation

This is still an association audit. Prediction-boundary features are available after training a standard 2stage model, not before any modeling. Use these results to test whether learned candidate margins add signal beyond raw graph and feasible-set geometry.
