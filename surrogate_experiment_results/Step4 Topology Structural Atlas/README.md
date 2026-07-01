# Step4 Topology Structural Atlas

This is the first Step4 layer. It is pure topology:

```text
compatibility graph
feasible cycle/chain candidate set
candidate conflict graph
structural metrics
```

No model weights, `X`, `y`, data seed, sample size, or test sample is needed for
this layer. The source of truth is the immutable Step3 topology template under:

```text
surrogate_experiment_results/Step3/pairs20_ndd2/data/topologies/<G-id>/template.json
```

Default topology set is the 8 sentinel topologies:

```text
G-269, G-398, G-784, G-970, G-364, G-836, G-79, G-670
```

Run from repo root:

```bash
python 'surrogate_experiment_results/Step4 Topology Structural Atlas/scripts/build_topology_structural_atlas.py'
```

To build all locked K18 topologies:

```bash
python 'surrogate_experiment_results/Step4 Topology Structural Atlas/scripts/build_topology_structural_atlas.py' \
  --use-k18-all
```

Outputs:

```text
results/topology_summary.csv
results/compatibility_arcs.csv
results/feasible_candidates.csv
results/candidate_conflicts.csv
visualizations/<topology_id>/compatibility_graph.svg
visualizations/<topology_id>/candidate_conflict_graph.svg
```

These artifacts define the structural substrate used by the later Step4
Decision Overlay and Rank-Reversal Detail layers.
