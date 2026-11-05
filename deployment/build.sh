#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/data/projects/CloudRobo
TASK_ENV=/data/anaconda3/envs/cloudrobo-habitat-gs
export PATH="$TASK_ENV/bin:/usr/local/cuda-12.4/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.4
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc
export CPATH="$TASK_ROOT/deployment/system-deps/root/usr/include${CPATH:+:$CPATH}"
export CMAKE_BUILD_PARALLEL_LEVEL=3
export HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF HABITAT_BUILD_GUI_VIEWERS=OFF HABITAT_BUILD_ASSIMP=OFF
cd "$TASK_ROOT/habitat-gs"
python -m pip install . --no-build-isolation --no-deps --config-settings=cmake.define.USE_SYSTEM_OPENEXR=ON --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=86
