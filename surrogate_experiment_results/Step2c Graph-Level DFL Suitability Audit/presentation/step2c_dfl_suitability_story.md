# Step2c Graph-Level DFL Suitability: Phase 1 Readout

## Scope

Phase 1 uses prospectively available graph and feasible-set descriptors only. It joins those descriptors to the existing all-400 Step2c model-seed outcome table.

## Population

- graphs: 400
- helpful_graph: 29
- harmful_graph: 7
- neutral_graph: 209

## Best Spearman Association By Feature Family

| Family | Feature | Spearman with median Delta | AUROC helpful | AUROC harmful |
| --- | --- | ---: | ---: | ---: |
| raw_topology | in_degree_gini | -0.111 | 0.355 | 0.274 |
| cycle_chain | cycle_to_chain_ratio | 0.038 | 0.559 | 0.679 |
| exchange_geometry | vertex_exchange_participation_gini | -0.055 | 0.455 | 0.437 |
| conflict_geometry | conflict_graph_density | -0.033 | 0.559 | 0.608 |

## Best Helpful / Harmful AUROC By Feature Family

| Family | Best helpful feature | Helpful AUROC | Best harmful feature | Harmful AUROC |
| --- | --- | ---: | --- | ---: |
| raw_topology | density | 0.630 | max_out_degree | 0.815 |
| cycle_chain | num_3cycles | 0.606 | cycle_to_chain_ratio | 0.679 |
| exchange_geometry | exchange_size_mean | 0.565 | exchange_size_mean | 0.622 |
| conflict_geometry | conflict_graph_density | 0.559 | conflict_graph_density | 0.608 |

## Selected Case Overlay

| Graph | median Delta pp | density pct | exchanges pct | conflict density pct | richness pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| G-392.json | 24.18 | 0.91 | 0.67 | 0.26 | 0.60 |
| G-1285.json | 22.36 | 0.14 | 0.28 | 0.95 | 0.29 |
| G-1560.json | 35.37 | 0.72 | 0.26 | 0.80 | 0.54 |
| G-1169.json | 10.67 | 0.97 | 0.60 | 0.36 | 0.61 |
| G-1449.json | 9.03 | 0.35 | 0.56 | 0.69 | 0.40 |
| G-142.json | 0.00 | 0.48 | 0.63 | 0.53 | 0.67 |
| G-946.json | 0.00 | 0.42 | 0.61 | 0.93 | 0.60 |
| G-14.json | -14.99 | 0.70 | 0.50 | 0.84 | 0.74 |
| G-163.json | -14.92 | 0.94 | 0.95 | 0.81 | 0.95 |

## Report-Safe Interpretation

This table is an association audit. A strong topology-only rule is not assumed. The result should be read as evidence about whether deployable graph and feasible-set descriptors contain signal about DFL suitability.
