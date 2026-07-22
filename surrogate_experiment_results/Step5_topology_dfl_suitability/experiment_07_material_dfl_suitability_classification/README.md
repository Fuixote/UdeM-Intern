# Experiment 07 — Material DFL suitability classification

## Research question

Can pure topology identify kidney-exchange instances where SPO+ produces a
practically material improvement over 2stage?

This experiment follows the negative magnitude-regression result from
Experiments 05 and 06. It does not try to predict the exact uplift. The locked
primary task is binary material-helpful detection:

    material_helpful versus non_helpful

where non_helpful pools neutral/uncertain and harmful topologies. Three-class
classification is a secondary descriptive analysis, and harmful cases are a
safety endpoint rather than a trainable primary class.

## Status

The local label, split, scalar-baseline, protocol, planner, launcher, and first
incidence-GNN stages are complete as of 2026-07-22. All 12 Experiment 07 tests
pass. The binary five-fold scalar review uses validation-only temperature
scaling and threshold selection. The five-job seed-42 incidence plan and safe
launcher preview pass. A target-free candidate-conflict graph artifact is
ready.

The formal seed-42 binary incidence screen completed successfully on Garnet on
2026-07-22. All five jobs succeeded with zero failures and produced 1,000 unique
OOF predictions; launcher wall time was 147 seconds with one worker and four
threads per job. The pooled reviewer and 5,000-resample paired bootstrap passed
their integrity audits. The preregistered promotion gate did not pass. Seeds 43
and 44 and the candidate-conflict GNN therefore remain blocked. Brevo
notification was not started because sending an external email requires
separate explicit approval.

## Locked primary labels

The target is the mean normalized improvement in percentage points over train
seeds 42, 43, and 44:

    delta = normalized_gap_2stage - normalized_gap_SPO+

The strict primary thresholds are:

- material_helpful: mean delta greater than 0.5 pp;
- material_harmful: mean delta less than -0.5 pp;
- neutral_or_uncertain: all remaining values, including exactly +/-0.5 pp.

The locked 1,000-topology distribution is:

| class | count |
| --- | ---: |
| material_harmful | 11 |
| neutral_or_uncertain | 827 |
| material_helpful | 162 |

The harmful class is extremely small: each primary-label-stratified fold has
only two or three harmful examples. It cannot support stable three-class
training. The operational endpoint is therefore helpful-vs-rest AUROC/AUPRC,
calibration, and frozen-policy utility. Every policy additionally reports
harmful selection count, total negative uplift, and worst selected harm.

## Confidence-aware labels

Confidence variants are exported as separate columns and never overwrite the
primary label:

1. At least two of the three seed deltas must match the primary material
   direction.
2. An exact 27-resample, three-seed percentile bootstrap interval is computed;
   the interval must exclude zero in the material direction.
3. Seed standard deviation above 0.5 pp is marked high variance.

The strict confidence label retains a material class only when all three gates
pass. Otherwise it maps to neutral_or_uncertain while preserving an explicit
uncertainty reason. Its distribution is 4 harmful, 877 neutral/uncertain, and
119 helpful. It is not used to train the primary classifier. It is reserved for
evaluation subsets, sample-weighting/abstention follow-ups, and uncertainty
analysis. With only three seeds, the bootstrap is a sensitivity screen rather
than strong inferential evidence.

The label audit passed 1,000/1,000 with no failures. There are 50
uncertain-material topologies: all 50 exceed the 0.5 pp seed-standard-deviation
gate, eight also have a bootstrap interval crossing zero, and four also fail
the two-of-three sign gate.

## Inputs and leakage boundary

The label source is the locked Experiment 05 three-seed target table. Scalar
features come from the Experiment 03 topology summary, and incidence graphs
come from the formal Experiment 05 graph artifact.

Topology-only models may use structural scalar fields or graph structure. They
must not use seed-level deltas, the mean target, target variance, sampled
contexts, training curves, or test statistics as inputs. Confidence fields are
targets/audit metadata only.

## Model sequence

Models are evaluated in this order:

1. training-fold class-prior and deterministic random-score baselines;
2. class-balanced binary logistic regression on 21 scalar topology features;
3. class-balanced binary ExtraTrees on the same scalar features;
4. the current binary relation-aware incidence GNN;
5. candidate-conflict scalar statistics, then a candidate-conflict GNN;
6. topology plus a preregistered cheap-context summary.

The first three stages run locally before any new Garnet job. Incidence and
candidate-conflict GNNs require local dry-run/input audits before a five-fold
seed-42 screen. Cheap context is disabled until its sampling budget and the
conversion from compute cost to pp utility are fixed explicitly.

Every model uses the same nested split: the outer test fold has 200
topologies, the following fold modulo five is the 200-topology validation
fold, and the remaining three folds provide 600 training topologies. Validation
selects the checkpoint, fits one temperature, and selects decision thresholds.
These objects are frozen before the untouched outer test fold is evaluated.

## Local binary scalar baseline checkpoint

The primary binary scalar pipeline produced exactly 1,000 unique OOF
predictions per model. Helpful prevalence is 0.162. Temperature scaling fits a
single positive temperature by validation NLL; it never changes ranking within
one fold, though fold-specific temperatures can change pooled cross-fold order.

| model | helpful AUROC | helpful AUPRC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| class prior | 0.5015 | 0.1621 | 0.1358 | 0.0000 |
| random score | 0.5345 | 0.1851 | 0.2501 | 0.3387 |
| logistic | **0.6411** | **0.2265** | 0.2300 | 0.3045 |
| ExtraTrees | 0.5846 | 0.1904 | **0.1783** | **0.1544** |

Thus aggregate scalar topology signal remains weak but nonzero. Temperature
scaling does not fix the large calibration error caused by the class-balanced
fits; this is retained as evidence rather than hidden by a test-fitted method.

At zero assumed compute cost, validation-regret thresholds select 908 logistic
and 880 ExtraTrees OOF cases. Their policy regrets are 0.0438 and 0.0520 pp,
respectively. Always using SPO+ selects all 1,000, captures all positive uplift,
and has only 0.0234 pp regret. This is a hard deployment baseline: at cost zero,
the selector is not useful unless it beats or safely approximates always-SPO+.

The validation precision constraints expose a different failure mode. Logistic
selects 15 cases under the 0.4 constraint and six under 0.5, with untouched-OOF
precision 0.333 and 0.500. ExtraTrees selects nine under either constraint but
achieves only 0.111 OOF precision. A validation precision constraint is not a
guarantee on the outer test distribution.

## Secondary three-class scalar checkpoint

The earlier three-class nested five-fold OOF review passed for all three models. Each outer
test fold contains 200 topologies; every fit uses 600 training topologies and
reserves 200 validation topologies.

| model | macro AUROC | macro AUPRC | macro-F1 | helpful AUROC | helpful AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| class prior | 0.4884 | 0.3324 | 0.3018 | 0.4963 | 0.1609 |
| logistic | **0.6610** | **0.4124** | 0.3315 | 0.5733 | **0.2105** |
| ExtraTrees | 0.5917 | 0.3682 | **0.3983** | **0.5835** | 0.1911 |

The scalar signal is real but weak. Logistic is best on macro ranking and
helpful AUPRC; ExtraTrees is best on macro-F1. At the argmax decision rule,
logistic selects 33.7% of topologies, has helpful precision 0.199, captures
42.2% of oracle positive improvement, and has policy regret 0.778 pp.
ExtraTrees selects 16.8%, has helpful precision 0.238, captures 33.7%, and has
regret 0.885 pp. These policy values are descriptive because the conversion
from extra compute to pp utility is not yet justified.

The confidence-label review also passed, but it contains only four harmful
examples. It is not a primary training benchmark.

## Incidence GNN fold-0 smoke

The relation-aware incidence GNN was implemented with the same 600/200/200
split, class-balanced cross entropy, three RGCN layers, seed 42, and CPU-only
execution. Input audit verified the locked graph SHA-256, 1,000 unique graph,
label, and fold records, matching topology/feasible-set hashes, and no target
or uncertainty input feature.

The fold-0 smoke early-stopped after 33 epochs and selected epoch 3. On the 200
untouched test topologies:

| metric | incidence GNN |
| --- | ---: |
| macro AUROC | 0.7235 |
| macro AUPRC | 0.4180 |
| macro-F1 | 0.4215 |
| helpful AUROC | 0.6682 |
| helpful AUPRC | 0.2856 |
| helpful precision / recall | 0.3333 / 0.5938 |

On this same fold, helpful AUPRC was 0.2000 for logistic and 0.1987 for
ExtraTrees. The structural smoke is therefore promising enough to justify a
five-fold seed-42 screen.

It is not yet deployable. Helpful ECE is 0.174, and the fixed helpful
probability threshold 0.5 selects no topology even though argmax predicts 57
helpful cases. Under the descriptive argmax policy, it captures 65.6% of the
fold's oracle positive improvement with 0.402 pp regret. This smoke motivated,
but is not part of, the now-locked binary five-fold protocol.

## Formal binary incidence seed-42 result

The five outer folds selected epochs 32, 14, 7, 24, and 19 and early-stopped
after 62, 44, 37, 54, and 49 epochs. Each topology appears in exactly one test
fold. Remote float32 delta copies differed from the locked double-precision
target by at most 1.81e-6 pp; the reviewer restored the authoritative target by
topology ID without changing any probability or selection.

| model | helpful AUROC | helpful AUPRC | Brier | ECE | selected | helpful precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Incidence GNN | 0.6342 | 0.2131 | 0.2300 | 0.2995 | 926 | 0.1728 / 0.9877 |
| Logistic | **0.6411** | **0.2265** | 0.2300 | 0.3045 | 908 | 0.1762 / 0.9877 |
| ExtraTrees | 0.5846 | 0.1904 | **0.1783** | **0.1544** | 880 | **0.1784** / 0.9691 |

The GNN beats the 0.162 prevalence and ExtraTrees point estimate but not
Logistic. It beats the best scalar model in only two of five folds. Paired
bootstrap GNN-minus-Logistic AUPRC is -0.0145 on average with 95% interval
[-0.0480, 0.0182] and only 0.184 probability above zero. Against ExtraTrees the
difference is +0.0227, interval [-0.0146, 0.0607], probability above zero 0.890.
Neither comparison satisfies the locked 0.95 evidence gate.

The zero-cost regret policy looks better than the learned scalar policies but
does not beat the non-learned operational baseline:

| policy | regret pp | positive uplift captured | harmful selected | total negative uplift pp |
| --- | ---: | ---: | ---: | ---: |
| always SPO+ | **0.0234** | **1.0000** | 11 | -23.3904 |
| Incidence GNN | 0.0269 | 0.9974 | 11 | -23.3904 |
| Logistic | 0.0438 | 0.9843 | 11 | -22.9576 |
| ExtraTrees | 0.0520 | 0.9773 | **10** | **-21.8994** |

The GNN selects every materially harmful topology and incurs exactly the same
negative uplift as always-SPO+, while missing about 0.26% of positive uplift.
It is therefore strictly worse than always-SPO+ at the preregistered zero-cost
operating point. Removing the five largest positive outliers leaves GNN regret
at 0.0270 pp, so the scalar-policy comparison is not driven only by those
outliers; it does not repair the always-SPO+ domination.

Validation precision constraints do not transfer. Both the 0.4 and 0.5 GNN
rules select 13 pooled OOF cases but achieve only 0.077 helpful precision,
0.0062 recall, and 0.30% positive-uplift capture. Top-k ranking is also weak:
top 100 contains 20 helpful and five harmful topologies and captures only 8.51%
of positive uplift.

The exact promotion failures are: pooled AUPRC below Logistic, only two of five
favorable fold directions, and insufficient paired-bootstrap AUPRC support.
Policy regret/capture, harmful-count, and top-five-outlier checks pass relative
to the two learned scalar policies, but those checks are not enough and the
always-SPO+ baseline is better.

## Locked nested decision protocol

For every outer fold, checkpoint selection uses validation weighted BCE. One
temperature is then fitted on validation logits only. Two decision families are
frozen and transferred unchanged to the outer test fold:

1. primary: maximize validation realized uplift minus a preregistered compute
   cost, currently 0 pp;
2. secondary: maximize validation helpful recall subject to helpful precision
   at least 0.4 or 0.5, with abstention when no nonempty policy is feasible.

The reviewer pools five untouched test folds, verifies each topology occurs
exactly once, and compares GNN, Logistic, and ExtraTrees with paired bootstrap.
It also reports always-2stage, always-SPO+, oracle, harmful selections, negative
uplift, worst harm, top-k capture for k=10/25/50/100, top-1/top-5 removal, and
1/99 winsorization.

## Candidate-conflict graph checkpoint

The target-free candidate-conflict graph builder completed and audited all
1,000 topologies. It produced 18,468 candidate nodes and 691,794 directed
conflict edges. Every undirected edge count matches the structural summary,
and the artifact contains no target or uncertainty metadata.

Locked compact artifact hashes:

| artifact | SHA-256 |
| --- | --- |
| material labels | 1ef744a7277b01355837e53fb11ac68754f769f65ba27a059c1fe37f74bbdbe8 |
| material folds | f26edb03ed0fd9edb7d5ca0f33ea9643e7328d3ebb59a9fe5df4becc473f0499 |
| candidate-conflict graphs | 0bb55f9f5f4df309ee72ad7a6f5b471ea39aa1558a341d9c13d53f0080d0dbfa |
| formal incidence OOF predictions | 6b5f2211e4b41647dc9f1be13dd3e812538e24ae7b3383cfef3a4564a5b6f4fb |
| formal incidence review audit | 824cdb684571ccf6958bb9c20935227d8c66ba8250938d47a71ab7d6f352f479 |
| formal model comparison | bfc6657ec7ca3e7e87ef3112eea78f11979f0628475d2904106b0902019e40b9 |

## Metrics and policy semantics

Classification reporting includes per-class and macro one-vs-rest AUROC/AUPRC,
macro-F1, balanced accuracy, log loss, multiclass Brier score, helpful-class
ECE, and top-label ECE.

For policy evaluation, the deployed threshold is selected on validation as
specified above, never fixed or retuned on test. With delta measured relative
to always using 2stage:

    policy regret = mean(max(delta, 0) - choose_SPO * delta)

    oracle improvement captured =
        sum(choose_SPO * max(delta, 0)) / sum(max(delta, 0))

Helpful precision is the fraction of selected topologies whose primary label
is material_helpful. Compute-adjusted net benefit is reported over explicit
extra-cost assumptions of 0, 0.05, 0.1, 0.25, and 0.5 pp per SPO+ selection;
it is not interpreted as a real deployment value until that conversion is
justified.

## Local workflow

From the repository root:

    python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_07_material_dfl_suitability_classification/scripts/build_material_labels.py

    python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_07_material_dfl_suitability_classification/scripts/plan_material_folds.py

    conda run -n KEPs python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_07_material_dfl_suitability_classification/scripts/run_helpful_scalar_baselines.py

    conda run -n KEPs python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_07_material_dfl_suitability_classification/scripts/run_scalar_classification_baselines.py

    python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_07_material_dfl_suitability_classification/scripts/build_candidate_conflict_graphs.py

    conda run -n KEPs python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_07_material_dfl_suitability_classification/scripts/train_helpful_incidence_classifier.py

    conda run -n KEPs python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_07_material_dfl_suitability_classification/scripts/plan_helpful_incidence_jobs.py

The trainer and launcher are preview-only without `--execute`. The formal plan
contains five fold jobs for seed 42 only and conservatively runs one job at a
time with four threads.

Tests:

    conda run -n KEPs python -m unittest discover -s surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_07_material_dfl_suitability_classification/tests -v

Canonical local outputs are under results/labels, results/splits, and
results/baselines/helpful_binary. Result files remain ignored by git; durable results
and interpretations must be recorded in this README.

## Start gates for structural models

- Label audit passes 1,000/1,000 with the locked target SHA-256.
- Five folds contain 200 topologies each and balance each primary class within
  one example.
- Scalar baseline OOF predictions contain exactly 1,000 unique topologies per
  model and pass the no-leakage audit.
- Incidence GNN formal screening is five folds at seed 42 only.
- Calibration and both threshold families are validation-only and locked.
- Promotion requires GNN helpful AUPRC above prevalence and both scalar models,
  at least four of five favorable fold directions, paired-bootstrap support,
  better policy regret/capture without worse harmful selection, and a top-5
  outlier-removed policy advantage.
- Seeds 43 and 44 are blocked until the seed-42 promotion gate passes.
- Candidate-conflict GNN must independently rebuild and hash its graph input.
- Topology plus cheap context requires an explicit context count, generation
  seed, runtime accounting rule, and compute-to-utility conversion.

## Current decision

Experiment 07 completed its seed-42 formal screen and did not pass promotion.
The binary reframing is methodologically better than magnitude regression and
topology does carry above-prevalence suitability signal, but the incidence GNN
does not outperform Logistic on pooled helpful AUPRC and does not yield a useful
zero-cost selector against always-SPO+. The correct present conclusion is:

> topology-only material suitability is weakly learnable, but the current
> incidence representation has not formed a reliable DFL method selector.

Do not run seeds 43/44 or candidate-conflict GNN under the current gate. A new
step requires either an explicitly justified nonzero SPO+ compute cost/safety
utility, or a separately preregistered representation hypothesis. Cheap context
remains out of scope until topology-only escalation is explicitly reopened.
