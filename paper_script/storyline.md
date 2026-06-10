Title:
When is decision-focused learning useful for graph combinatorial optimization?

Main question:
Why does SPO+ improve substantially over 2stage in shortest path, but only moderately in KEP?

Hypothesis:
DFL has limited value in decomposable packing problems with close substitute solutions.

Property X:
Close substitute solutions in a decomposable packing structure.

Mechanism:
Prediction errors may change solution identity, but if the selected alternative has similar true objective value, the decision regret remains small.

Empirical evidence:
1. Best/second-best KEP solutions often both close to oracle.
2. This persists under max cycle length 4 and 5.
3. Density perturbations reshape mechanisms but do not remove the general close-alternative behavior in selected cases.

Toy examples:
1. KEP / set-packing toy example.
2. Shortest-path contrast.
3. Stable-set or set-packing example as another combinatorial problem with the same behavior.