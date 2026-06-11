# Toy examples for Property X

## Hypothesis

Decision-focused learning is most useful when prediction errors change the
objective value of the selected solution, not merely its identity.

For packing-style combinatorial problems with close substitute solutions,
2stage can select a different feasible solution from the oracle while still
having small objective regret. In such cases, SPO+/FY has limited room to
improve.

## Property X

Close substitute solutions in a decomposable packing structure.

A problem has this property when feasible solutions are composed of several
mostly independent components, and replacing one component with another feasible
component often changes the true objective only slightly.

## Positive packing-style examples

The oracle packing is C12 + C34 with true value 20.0. The 2stage prediction
selects C13 + C24 with true value 19.8. The solution identity changes, but the
normalized oracle gap is only 1%.

The same phenomenon appears in a small stable-set problem. The oracle stable set
is {A,B}, while 2stage selects {C,D}. The normalized oracle gap is again only 1%.

The expanded toy family adds weighted matching, cardinality-knapsack, and
partition-matroid examples. In each case, 2stage selects a different feasible
solution from the oracle, but the selected alternative is a close substitute and
the normalized regret remains small.

## Parametric epsilon family

The parametric construction has two complete feasible solutions: S_star with
true value 1 and S_epsilon with true value 1 - epsilon. The 2stage prediction
slightly overestimates S_epsilon, so it selects the wrong solution. The identity
changes, but the normalized regret is exactly epsilon and can be made
arbitrarily small.

This supports a structural, not merely numerical, version of Property X. The
claim should still be framed as a mechanism and not as a theorem covering every
packing instance.

## Path-like negative controls

The shortest-path example has two paths. The oracle path has true cost 5.0, but
2stage underestimates the wrong path and selects a path with true cost 8.0. The
normalized regret is 60%.

A second serial-path negative control has true costs 10.0 and 15.0, yielding 50%
regret when 2stage underestimates the high-cost chain.

These negative controls illustrate why shortest path can behave differently: a
prediction error can redirect the entire connected path, producing a much larger
regret.
