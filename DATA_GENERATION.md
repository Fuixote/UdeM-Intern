# Data Generation

This document describes how to generate synthetic raw kidney-exchange data in the current repository.

## Entry Point

Use:

```bash
./run_experiment.sh data-generate [generator args...]
```

This dispatches to [0-data-generation.py](/home/weikang/projects/UdeM-Intern/Exps/0-data-generation.py).

## Current Project Default

The current project default is:

- global donor blood-type distribution
- tuning enabled
- `SplitPRA` enabled by default
- `BandedXMatch` enabled by default
- recipient age sampled uniformly on `[18, 68]`
- donor age sampled uniformly on `[18, 68]`
- edge utility sampled uniformly on `[1, 90]`

In practice, a plain run now uses split cPRA bands and the banded compatibility model:

```bash
./run_experiment.sh data-generate
```

## Output

By default, batches are written to:

```bash
dataset/raw/<YYYY-MM-DD_HHMMSS>/
```

Artifacts in each batch:

- `genjson-*.json`
- `config.json`
- `effective_config.json`
- `run_info.json`
- `batch_summary.json`
- `batch_report.md`

## Main Parameters

### Scale

- `--instances`, default `1000`
- `--patients`, default `50`
- `--seed`, default `42`

### Blood-type and donor-count distributions

- `--prob_ndd`
- `--prob_o`, `--prob_a`, `--prob_b`
- `--donor_prob_o`, `--donor_prob_a`, `--donor_prob_b`
- `--donors1`, `--donors2`, `--donors3`

Notes:

- recipient `AB` probability is the residual `1 - (O + A + B)`
- donor `AB` probability is the residual `1 - (O + A + B)`
- 4-donor probability is the residual `1 - (donors1 + donors2 + donors3)`

### Family-related parameters

- `--prob_spousal`
- `--prob_female`
- `--prob_spousal_pra_compat`

### cPRA distribution

Single shared cPRA distribution:

- `--pra_bands_string`

This is now a fallback path. It is used when you explicitly provide `--pra_bands_string` and do not explicitly choose split cPRA bands.

Project default split cPRA parameters:

- `--compat_pra_bands_string`
- `--incompat_pra_bands_string`

### Compatibility model

- `--compat_bands_string`

Project default:

```text
0.0 0.50 0.4349 0.33012
0.50 0.95 0.342 0.64194
0.95 0.96 0.942
0.96 0.97 0.947
0.97 0.98 0.975
0.98 0.99 0.985
0.99 1 0.985
1 1.01 0.988
```

### Tuning

- `--tune_iters`, default `100`
- `--tune_size`, default `1000`
- `--tune_error`, default `0.05`
- `--no_tune` to disable tuning

### Age and utility ranges

- `--recipient_age_min`, `--recipient_age_max`
- `--donor_age_min`, `--donor_age_max`
- `--utility_min`, `--utility_max`

### Split donor blood distributions

- `--split_donor_blood`
- `--donor_probs_by_patient_o`
- `--donor_probs_by_patient_a`
- `--donor_probs_by_patient_b`
- `--donor_probs_by_patient_ab`
- `--donor_probs_by_patient_ndd`

Each donor-probability override is a comma-separated 4-tuple in the order:

```text
probO,probA,probB,probAB
```

## Presets

You can also load a named preset:

```bash
./run_experiment.sh data-generate --preset <name>
```

Available presets:

- `saidman`
- `paper-recip-blood`
- `paper-split-donor-blood`
- `split-pra`
- `calc-xmatch`
- `tweak-xmatch`
- `tweak-xmatch-pra0`
- `banded-xmatch`
- `banded-xmatch-pra0`

Preset application order:

1. load preset values
2. apply explicit CLI overrides

Example:

```bash
./run_experiment.sh data-generate --preset split-pra --compat_bands_string $'0 1 0.45 0.51'
```

## Common Commands

Use the current project default:

```bash
./run_experiment.sh data-generate --seed 42 --run_name default_v1
```

Use a different preset:

```bash
./run_experiment.sh data-generate --preset calc-xmatch --run_name calc_xmatch_v1
```

Force a single shared PRA distribution:

```bash
./run_experiment.sh data-generate \
  --pra_bands_string $'0.7 0.05\n0.2 0.10\n0.1 0.90' \
  --run_name single_pra_v1
```

Change age and utility ranges:

```bash
./run_experiment.sh data-generate \
  --recipient_age_min 25 --recipient_age_max 75 \
  --donor_age_min 21 --donor_age_max 70 \
  --utility_min 5 --utility_max 60
```

## Preset Matrix

For the full parameter-by-preset table, see:

- [PRESET_PARAMETER_MATRIX.md](/home/weikang/projects/UdeM-Intern/Exps/PRESET_PARAMETER_MATRIX.md)

This matrix includes:

- all configurable generator parameters
- the default effective value
- the effective value under each built-in preset
- full string bodies for long cPRA / compatibility-band settings

## Reproducibility Notes

- identical parameters with the same `--seed` produce deterministic output
- `config.json` stores the requested generator configuration
- `effective_config.json` stores the post-tuning configuration actually used
- `batch_summary.json` and `batch_report.md` should be checked before downstream experiments
