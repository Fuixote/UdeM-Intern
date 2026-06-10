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

## KEP / set-packing example

The oracle packing is C12 + C34 with true value 20.0. The 2stage prediction
selects C13 + C24 with true value 19.8. The solution identity changes, but the
normalized oracle gap is only 1%.

## Stable-set example

The same phenomenon appears in a small stable-set problem. The oracle stable set
is {A,B}, while 2stage selects {C,D}. The normalized oracle gap is again only 1%.

This suggests that the mechanism is not specific to kidney exchange. It can
occur in other packing-style combinatorial optimization problems.

## Shortest-path contrast

The shortest-path example has two paths. The oracle path has true cost 5.0, but
2stage underestimates the wrong path and selects a path with true cost 8.0. The
normalized regret is 60%.

This illustrates why shortest path can behave differently: a prediction error
can redirect the entire connected path, producing a much larger regret.
