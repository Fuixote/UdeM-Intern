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

## Fold-0 smoke result

The `fold=0, seed=42` smoke completed on Garnet on 2026-07-20. All 11 jobs
succeeded and the review audited 160 per-run metric rows without failures. The
classifier early-stopped after 62 epochs at epoch 32. On its 200-topology test
fold it achieved AUROC 0.7234, average precision 0.5924, balanced accuracy
0.6989, recall 0.8077, and F1 0.6597. Its confusion matrix was TP=63, TN=72,
FP=50, and FN=15. This is evidence that topology carries useful zero/nonzero
signal, but it is only one fold.

The best deployable fold-0 variants on the full test fold were:

| gate | stage-two training set | objective | MAE (pp) | RMSE (pp) | R2 | Spearman |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| hard | nonzero | signed-log MSE | 1.3957 | 4.5046 | 0.0164 | 0.1324 |
| soft | nonzero | signed-log MSE | 1.3611 | 4.5428 | -0.0003 | 0.1034 |
| hard | nonzero | Huber | 1.8006 | **4.4673** | **0.0326** | 0.1380 |
| soft | nonzero | Huber | 1.7099 | 4.4780 | 0.0280 | 0.1128 |

For reference on the identical fold, Experiment 05 seed 42 had MAE 1.3002,
RMSE 4.5448, and R2 -0.0012; its three-seed ensemble had MAE 1.3376, RMSE
4.5409, and R2 0.0005; the zero predictor had MAE 1.0629, RMSE 4.6522, and R2
-0.0491. Thus the smoke hurdle model improves tail-sensitive RMSE/R2 but does
not beat the zero or Experiment 05 baselines on MAE. The oracle nonzero gate
reached MAE 1.1755 with signed-log MSE and RMSE 4.3519/R2 0.0820 with Huber,
showing that classification error is a material bottleneck.

The smoke ordering does not support dropping small nonzero labels: every best
deployable full-fold row trained on the full nonzero subset, not the material-
only subset. Weighted Huber and plain MSE were worse; adding ranking loss was
close to, but did not improve on, plain Huber. Signed-log MSE is the leading
MAE candidate while Huber is the leading RMSE/R2 candidate.

Current status: implementation, five unit tests, locked input audit, dependency
planning, launcher previews, and the 11-job smoke all pass. The next gated run
is the seed-42 five-fold screen: five classifiers plus 50 regressors. The
reviewer pools the five disjoint test folds into 1,000-topology OOF metrics;
only variants that survive that screen may be promoted to seeds 43 and 44.
