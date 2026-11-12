#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ENV=/data/anaconda3/envs/cloudrobo-habitat-gs
export PATH="$TASK_ENV/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=3
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=warning
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -u "$PROJECT_ROOT/scripts/validate_navigation.py" "$@"
