#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT_DIR/run_experiment.sh" 2stg-reg
