#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/data/projects/CloudRobo
TASK_ENV=/data/anaconda3/envs/cloudrobo-habitat-gs
export PATH="$TASK_ENV/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=3
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=warning
exec python -u "$TASK_ROOT/deployment/validate_navigation.py" "$@"
