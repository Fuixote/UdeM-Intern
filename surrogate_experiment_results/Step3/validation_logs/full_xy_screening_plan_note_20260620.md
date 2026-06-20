# Phase-B Prime Full-(X,y) Screening Plan Note

Date: 2026-06-20

The existing `pairs20_ndd2` topology bank does not need to be regenerated for
Phase-B prime screening. The fixed object remains:

```text
G_k = vertices, arcs, arc ordering, cycle/chain candidates, and feasible set
```

The Phase-A topology hashes, arc-order hashes, feasible-set hashes, structural
descriptors, and oracle-only landscape summaries remain valid as topology
pre-screen evidence.

The previous Phase-B fixed-X / relabel-y evidence remains historical screening
and optimization-debugging evidence. It is not final topology-selection evidence
under the advisor-facing full-(X,y) protocol, because it changes Step2c label
noise while holding model-visible X fixed.

Phase-B prime full-(X,y) screening should use the existing 160 selected Phase-B
topologies as its candidate pool. It must generate new samples on each fixed
topology with the screening namespaces:

```text
screen_train
screen_validation
screen_test
```

Formal confirmation remains separate and must use:

```text
confirm_train
confirm_validation
confirm_test
```

Screening configurations may remain `pilot_not_locked` while the protocol is
being validated. Formal confirmation requires a reviewed `locked` context
generator configuration before confirmation data or jobs are launched.
