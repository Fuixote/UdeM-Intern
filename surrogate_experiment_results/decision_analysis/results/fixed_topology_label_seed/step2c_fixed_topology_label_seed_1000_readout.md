# Step2c Fixed-Topology Label-Seed Robustness

Scope: topology and trained Step2c model weights are fixed; only Step2c label_seed varies.

## G-1560.json
- label seeds: 1000
- unique topology hashes: 1
- unique label hashes: 1000
- Case C preserved rate: 0.545 +/- 0.031
- SPO+ better rate: 0.998
- correction persistence rate: 0.000 +/- 0.000
- rank-2 promotion persistence rate: 0.545 +/- 0.031
- mean rank-1 gap reduction: 0.2720
- oracle unique solutions: 8
- mean oracle Jaccard to first seed: 0.820

## G-392.json
- label seeds: 1000
- unique topology hashes: 1
- unique label hashes: 1000
- Case C preserved rate: 0.661 +/- 0.029
- SPO+ better rate: 1.000
- correction persistence rate: 0.661 +/- 0.029
- rank-2 promotion persistence rate: 0.000 +/- 0.000
- mean rank-1 gap reduction: 0.3026
- oracle unique solutions: 29
- mean oracle Jaccard to first seed: 0.582
