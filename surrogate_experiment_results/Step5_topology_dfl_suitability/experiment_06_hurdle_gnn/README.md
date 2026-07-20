# Experiment 06 — two-stage hurdle topology GNN

Experiment 06 preserves Experiment 05 as the completed continuous-regression
baseline and changes the learning problem to match the zero-inflated formal
target. It uses the same locked 1,000 formal graphs, five folds, and three-seed
mean label. No sampled context, test statistic, label uncertainty, or target is
an input feature.

## Locked protocol

Stage one predicts whether `formal_label_mean_pp` is exactly zero or nonzero.
It uses the same relation-aware incidence GNN backbone and class-balanced binary
cross entropy. The training-fold zero/nonzero ratio determines `pos_weight`;
the fixed decision threshold is 0.5. Validation weighted BCE selects the early
stopping checkpoint. Test reporting includes balanced accuracy, F1, AUROC, and
average precision.

Stage two trains a separate regressor on either:

- all nonzero training-fold topologies; or
- material training-fold topologies with `abs(label) > 0.1 pp`.

Each subset tests five objectives while keeping the backbone, split, optimizer,
batch size, and early stopping policy fixed:

1. `huber`: the Experiment 05 pointwise objective on the subset;
2. `weighted_huber`: train-only magnitude weights, normalized to mean one;
3. `mse`: standardized-target mean squared error;
4. `signed_log_mse`: MSE on `sign(y) * log1p(abs(y))`, inverted to pp;
5. `huber_rank`: Huber plus 0.25 times within-batch pairwise logistic ranking.

The ranking variant retains a pointwise term because ranking loss alone does
not identify the pp scale. Every regressor reports raw predictions, hard hurdle
predictions (zero below classifier probability 0.5), soft hurdle predictions
(`p(nonzero) * regression`), and an oracle true-nonzero gate used only to
separate classifier error from regression error.

## Execution gate

The first run is a bounded `fold=0, seed=42` smoke: one classifier plus ten
subset/objective regressors, for 11 jobs total. It must pass input hashes,
dependency checks, prediction uniqueness, and review before any five-fold or
three-seed expansion. The full seed-42 screen would contain five classifiers
and 50 regressors. Only the strongest variants should be promoted to seeds 43
and 44.

Current status: the implementation, four unit tests, locked input audit, 11-row
dependency plan, and both launcher previews pass locally. No Experiment 06
training has been launched yet.
