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
1. Packing-style positive family: KEP / set-packing, stable set, weighted matching, cardinality knapsack, and partition matroid.
2. Parametric epsilon construction: solution identity changes while normalized regret can be made arbitrarily small.
3. Path-like negative controls: shortest path and serial path, where a ranking error can redirect the whole connected decision and create large regret.
