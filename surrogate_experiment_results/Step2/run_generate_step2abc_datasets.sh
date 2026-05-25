#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
LABEL_SEED="${LABEL_SEED:-20260523}"
STEP2_RHO="${STEP2_RHO:-0.5}"
STEP2_KAPPA="${STEP2_KAPPA:-3}"
STEP2_DELTA="${STEP2_DELTA:-1e-12}"
STEP2_EPSILON_BAR="${STEP2_EPSILON_BAR:-0.5}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

STEP2A_SCRIPT="surrogate_experiment_results/Step2/Step2a_additive_linear_gaussian/data-processing.py"
STEP2B_SCRIPT="surrogate_experiment_results/Step2/Step2b_polynomial_degree_noiseless/data-processing.py"
STEP2C_SCRIPT="surrogate_experiment_results/Step2/Step2c_polynomial_degree_multiplicative_noise/data-processing.py"
VALIDATOR_SCRIPT="surrogate_experiment_results/Step2/validate_step2_processed_dataset.py"

format_decimal_tag() {
  "${PYTHON_BIN}" -c 'import sys; print(f"{int(round(float(sys.argv[1]) * 100)):03d}")' "$1"
}

RHO_TAG="$(format_decimal_tag "${STEP2_RHO}")"
EPS_TAG="$(format_decimal_tag "${STEP2_EPSILON_BAR}")"

declare -A RAW_DIRS=(
  [main2000]="dataset/raw/2026-04-17_135607"
  [val2000]="dataset/raw/2026-05-19_000000__step1_noisy_linear_sigma010_validation2000_seed20260519"
  [unseen10000]="dataset/raw/2026-05-20_000000__step1_noisy_linear_sigma010_unseen_test10000_seed20260520"
)

declare -A EXPECTED_GRAPH_COUNTS=(
  [main2000]="2000"
  [val2000]="2000"
  [unseen10000]="10000"
)

SPLITS=(main2000 val2000 unseen10000)
DEGREES=(1 2 4 8)

run_cmd() {
  printf '>'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

process_dataset() {
  local script_path="$1"
  local raw_dir="$2"
  local output_dir="$3"
  local expected_graph_count="$4"
  local expected_label_mode="$5"
  shift 5

  local cmd=("${PYTHON_BIN}" "${script_path}" "${raw_dir}" "${output_dir}" --all --output_as_batch_dir "$@")
  if [[ "${FORCE}" == "1" ]]; then
    cmd+=(--force)
  fi

  run_cmd "${cmd[@]}"
  run_cmd \
    "${PYTHON_BIN}" \
    "${VALIDATOR_SCRIPT}" \
    "${output_dir}" \
    --strict \
    --expected_graph_count "${expected_graph_count}" \
    --expected_label_mode "${expected_label_mode}"
}

echo "Step2 ABC dataset generation"
echo "DRY_RUN=${DRY_RUN} FORCE=${FORCE} LABEL_SEED=${LABEL_SEED} RHO_TAG=${RHO_TAG} EPS_TAG=${EPS_TAG}"
echo "Validator: ${VALIDATOR_SCRIPT}"

for split in "${SPLITS[@]}"; do
  raw_dir="${RAW_DIRS[${split}]}"
  output_dir="dataset/processed/step2a_additive_rho${RHO_TAG}_${split}_seed${LABEL_SEED}"
  process_dataset \
    "${STEP2A_SCRIPT}" \
    "${raw_dir}" \
    "${output_dir}" \
    "${EXPECTED_GRAPH_COUNTS[${split}]}" \
    "step2a_additive_linear_gaussian" \
    --label_mode step2a_additive_linear_gaussian \
    --step2a_noise_rho "${STEP2_RHO}" \
    --label_seed "${LABEL_SEED}"
done

for degree in "${DEGREES[@]}"; do
  for split in "${SPLITS[@]}"; do
    raw_dir="${RAW_DIRS[${split}]}"
    output_dir="dataset/processed/step2b_poly_d${degree}_${split}_seed${LABEL_SEED}"
    process_dataset \
      "${STEP2B_SCRIPT}" \
      "${raw_dir}" \
      "${output_dir}" \
      "${EXPECTED_GRAPH_COUNTS[${split}]}" \
      "step2b_polynomial_degree_noiseless" \
      --label_mode step2b_polynomial_degree_noiseless \
      --step2b_degree "${degree}" \
      --step2b_kappa "${STEP2_KAPPA}" \
      --step2b_delta "${STEP2_DELTA}" \
      --label_seed "${LABEL_SEED}"
  done
done

for degree in "${DEGREES[@]}"; do
  for split in "${SPLITS[@]}"; do
    raw_dir="${RAW_DIRS[${split}]}"
    output_dir="dataset/processed/step2c_poly_d${degree}_mult_eps${EPS_TAG}_${split}_seed${LABEL_SEED}"
    process_dataset \
      "${STEP2C_SCRIPT}" \
      "${raw_dir}" \
      "${output_dir}" \
      "${EXPECTED_GRAPH_COUNTS[${split}]}" \
      "step2c_polynomial_degree_multiplicative_noise" \
      --label_mode step2c_polynomial_degree_multiplicative_noise \
      --step2c_degree "${degree}" \
      --step2c_kappa "${STEP2_KAPPA}" \
      --step2c_delta "${STEP2_DELTA}" \
      --step2c_epsilon_bar "${STEP2_EPSILON_BAR}" \
      --label_seed "${LABEL_SEED}"
  done
done
