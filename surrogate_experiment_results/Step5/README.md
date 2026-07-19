
  # Step5 Topology DFL Suitability Weak-Label Experiment Plan

  ## Summary

  Use the existing pairs20_ndd2 1000-topology bank to generate 1000 topology-level weak labels for whether DFL/SPO+ is useful versus 2stage under
  one fixed cheap protocol:

  topology_count = 1000
  sample_size = 50
  training_size = 40
  validation_size = 10
  data_seed = 42
  theta_seed = 42
  gurobi_seed = 42
  master_label_seed = 20260719
  test_size = 1000
  protocol = screen
  max_epochs = 1500
  metric_stride = 1
  early_stop_patience = 20
  early_stop_min_delta = 0.0001

  Heavy data, model outputs, and job results should live on Garnet under /local1/fuweik/UdeM-Intern/.... Tracked repo changes should be limited to
  small configs, scripts, summaries, tests, and README documentation.

  ## Local Implementation Status

  As of 2026-07-19, the six experiment-local pipeline scripts have been implemented under:

  surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_01_weak_label_seed42_sample50/scripts/

  Implemented locally:

  - topology-bank locking and integrity audit;
  - deterministic 4:1 artifact construction;
  - artifact integrity audit;
  - dry-run-only paired-job planning;
  - bounded and resumable execution launching;
  - result review and weak-label export.

  Local validation covers the 40/10 split, artifact hashes, dry-run safety gate, launcher command conversion, label calculation, and real-bank
  compatibility. The real 1000-row bank passed the temporary validation with 1000 unique topology/hash rows and no missing template/source files.

  The implementation was recorded in commit `6208873d53dfe4e624a80bfbb2aea8b1cb40b386` and synchronized to Garnet. The pre-smoke protocol is locked in
  `configs/experiment.yaml`: `master_label_seed=20260719`, `max_epochs=1500`, `metric_stride=1`, `early_stop_patience=20`, and
  `early_stop_min_delta=0.0001`. The generator config and 1000-row topology manifest are locked, and `results/topology_bank_audit.json` reports
  `passed=true` with no failures. On 2026-07-19, the `G-0` through `G-19` smoke20 artifacts were built on Garnet: 20/20 topology bundles passed
  the artifact audit with zero failures. No model training jobs have been launched yet; the next gate is dry-run plan generation and launcher review.

  ## Key Changes

  Create a new experiment namespace:

  surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_01_weak_label_seed42_sample50/

  Add three small experiment-local script groups:

  - Artifact builder / planner
      - Lock the 1000 topology rows from surrogate_experiment_results/Step3/pairs20_ndd2/data/topologies/topology_bank.csv.
      - Build per-topology train_bank.npz, validation_sample_size050.npz, test.npz, and eval_manifest_sample_size050.json.
      - Use the K18 4:1 convention: sample_size=50 means 40 training and 10 validation samples.
      - Write plans/weak_label_jobs.csv with exactly 1000 rows and dry-run run_one_job.py commands.

  - Launcher / watcher
      - Run the paired 2stage and SPO+ jobs through the existing Step3 run_one_job.py.
      - Use Garnet runtime setup from configs/runtime/garnet.env.
      - Use conservative concurrency: start with smoke concurrency 4; full run default 16 normal jobs, with a small long-job queue if early smoke
        shows outliers.

      - Use scripts/experiment_notify.py for completion email, consistent with repo notification practice.

  - Reviewer / label exporter
      - Aggregate job outputs into:
          - results/weak_label_job_metrics.csv
          - results/weak_label_topology_summary.csv
          - results/weak_label_integrity_audit.json

      - Define continuous label:

        delta = test_gap_2stage - test_gap_spoplus

      - Define weak class label:

        helpful      if delta > 0.1
        harmful      if delta < -0.1
        near_neutral if abs(delta) <= 0.1

      - Mark all labels as weak_label=true and record the full label protocol in every summary.

  ## Execution Stages

  1. Bank confirmation
      - Verify topology_bank.csv has exactly 1000 topology rows plus header.
      - Verify all template_path files exist.
      - Verify topology_hash, arc_order_hash, and feasible_set_hash are unique enough for graph-level splitting.
      - Commit only the locked topology manifest and audit summary, not generated NPZ data.

  2. 20-50 topology smoke
      - Use the first 20 topology rows by current bank order unless a smoke config explicitly selects 50.
      - Build artifacts, run paired jobs, and review summary.
      - Acceptance: all smoke jobs success; every job has matching 2stage/SPO+/evaluation status; weak_label_topology_summary.csv has one row per
        topology.

  3. Full 1000 weak-label sweep
      - Build artifacts for all 1000 topologies under /local1/fuweik/UdeM-Intern/....
      - Run all 1000 paired jobs in tmux on Garnet.
      - Review results only after weak_label_integrity_audit.json reports:

        job_rows = 1000
        success = 1000
        failures = []
        passed = true

  4. Baseline before GNN
      - Train tabular baselines using only topology-available structural fields.
      - Exclude leakage fields such as prior selection bucket, screen outcome, K18 role, or any DFL/2stage result-derived feature.
      - Primary target: regression on delta.
      - Secondary target: classification into helpful / harmful / near_neutral.
      - Split strictly by topology: default 700 / 150 / 150.

  5. First GNN
      - Input graph: unweighted directed compatibility graph.
      - Node features: Pair vs NDD type only.
      - Edge features: none for v1.
      - Compare GNN against tabular baseline on the same split and same labels.
      - Keep model small; this experiment tests whether topology structure has signal, not whether a large model can overfit.

  6. Targeted multi-seed stability check
      20 most predicted helpful
      20 most predicted harmful
      20 most uncertain / closest to zero

      - Re-run those 60 topologies with seeds 101,102,103,104,105 at sample_size=50.
      - Acceptance: report seed stability, sign flip count, mean/median delta, and whether single-seed seed42 labels were reliable.

  ## Test Plan

  - Unit tests:
      - 1000-topology manifest locking preserves row count and hash fields.
      - 4:1 split maps sample_size=50 to training_size=40, validation_size=10.
      - Planner refuses to produce non-dry-run commands during planning.
      - Reviewer refuses to pass if any paired job lacks 2stage, SPO+, or evaluation outputs.
      - Baseline/GNN dataset builder rejects leakage columns.

  - Smoke acceptance:
      - 20-50 topology smoke completes with no failed jobs.
      - Review summary has one topology-level label per topology.
      - Re-running reviewer is deterministic.

  - Full-run acceptance:
      - 1000 job rows, 1000 successes, zero failures.
      - Exactly 1000 topology labels.
      - Every label row records data_seed=42, sample_size=50, training_size=40, validation_size=10, theta_seed=42, gurobi_seed=42.

  ## Assumptions

  - Do not expand to 2000 topologies before this first weak-label experiment.
  - Do not treat the 50 training/validation samples as 50 graph labels.
  - Do not treat the single-seed labels as final stable truth.
  - Do not use Slurm; use ssh cirrelt, Garnet, tmux, and /local1/fuweik.
  - Keep generated datasets, weights, full job folders, and raw large outputs ignored by Git.
