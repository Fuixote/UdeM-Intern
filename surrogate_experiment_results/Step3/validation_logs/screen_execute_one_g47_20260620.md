# Screen-Protocol Execute-One Smoke: G-47

Date: 2026-06-20

- commit: `e936d545111af7051ae91a86c2f7d48516040010`
- machine: Garnet
- Gurobi: `gurobipy 13.0.0`
- protocol: `screen`
- topology: `G-47`
- train_seed_count: 1
- train_size: 2
- max_train_size: 5
- prefix_sizes: `2,3,5`
- validation/test: `2/2`
- max_epochs: 2
- audit: passed
- `run_one_job --execute`:
  - 2stage success
  - SPO+ success
  - evaluation success
- runtime: 14.94s
- output size: 796K
- output directory: `/local1/fuweik/step3_screen_execute_one_20260620`

Observed files:

```text
pilot_summary.json
audit_results.json
pilot_jobs.csv
job_status.json
paired_job_manifest.json
2stage/model_weights/2stage_best_by_validation_mse_loss.npz
spoplus/model_weights/spoplus_best_by_validation_spoplus_loss.npz
evaluation/metrics/test_summary.csv
```

Safety:

- no full 160 screening
- no confirmation
- no topology regeneration
- no large tracked artifact
