#!/usr/bin/env bash
set -u

TOPOLOGIES=(G-269 G-398 G-784 G-970 G-364 G-836 G-79 G-670)
SAMPLE_SIZES=(50 100 500)
DATA_SEEDS=(101 102 103 104 105)

OVERLAY_DIR="surrogate_experiment_results/Step4 Decision Overlay"
SCRIPT="$OVERLAY_DIR/scripts/compute_decision_overlay.py"
RESULT_DIR="$OVERLAY_DIR/results/full_8topology_5seed_3size"
LOG_DIR="$OVERLAY_DIR/logs/full_8topology_5seed_3size_20260701"
MASTER_LOG="$LOG_DIR/master.log"

if [[ -f configs/runtime/garnet.env ]]; then
  source configs/runtime/garnet.env
fi

PYTHON_BIN="${KEP_PYTHON:-python}"

mkdir -p "$RESULT_DIR/per_batch" "$LOG_DIR"
rm -f "$RESULT_DIR/_SUCCESS" "$RESULT_DIR/_FAILED"
: > "$MASTER_LOG"

echo "START $(date -Is)" >> "$MASTER_LOG"
echo "topologies=${TOPOLOGIES[*]}" >> "$MASTER_LOG"
echo "sample_sizes=${SAMPLE_SIZES[*]}" >> "$MASTER_LOG"
echo "data_seeds=${DATA_SEEDS[*]}" >> "$MASTER_LOG"
echo "python=$PYTHON_BIN" >> "$MASTER_LOG"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pids=()
batches=()

for topology_id in "${TOPOLOGIES[@]}"; do
  for sample_size in "${SAMPLE_SIZES[@]}"; do
    batch="${topology_id}_sample${sample_size}"
    batches+=("$batch")
    (
      set -euo pipefail
      echo "START $batch $(date -Is)"
      "$PYTHON_BIN" "$SCRIPT" \
        --topology-ids "$topology_id" \
        --data-seeds "${DATA_SEEDS[@]}" \
        --sample-sizes "$sample_size" \
        --context-limit 1000 \
        --gurobi-threads 1 \
        --decision-output "$RESULT_DIR/per_batch/${batch}_decision_solution_rows.csv" \
        --summary-output "$RESULT_DIR/per_batch/${batch}_candidate_overlay_summary.csv"
      echo "DONE $batch $(date -Is)"
    ) > "$LOG_DIR/${batch}.log" 2>&1 &
    pid="$!"
    pids+=("$pid")
    echo "LAUNCHED $batch pid=$pid log=$LOG_DIR/${batch}.log" >> "$MASTER_LOG"
  done
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  batch="${batches[$idx]}"
  if wait "$pid"; then
    echo "BATCH_DONE $batch $(date -Is)" >> "$MASTER_LOG"
  else
    echo "BATCH_FAILED $batch $(date -Is)" >> "$MASTER_LOG"
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  python - "$RESULT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
per_batch = result_dir / "per_batch"

outputs = [
    ("*_decision_solution_rows.csv", result_dir / "decision_solution_rows.csv"),
    ("*_candidate_overlay_summary.csv", result_dir / "candidate_overlay_summary.csv"),
]

for pattern, output_path in outputs:
    files = sorted(per_batch.glob(pattern))
    wrote_header = False
    with output_path.open("w", newline="") as fout:
        writer = None
        for path in files:
            with path.open(newline="") as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                if not wrote_header:
                    writer.writeheader()
                    wrote_header = True
                for row in reader:
                    writer.writerow(row)
PY
  touch "$RESULT_DIR/_SUCCESS"
  echo "DONE_ALL $(date -Is)" >> "$MASTER_LOG"
else
  touch "$RESULT_DIR/_FAILED"
  echo "FAILED $(date -Is)" >> "$MASTER_LOG"
fi

exit "$status"
