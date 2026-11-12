#!/usr/bin/env bash
# 运行全密度 50 条路线；配置和输出目录可通过参数覆盖。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$PROJECT_ROOT/configs/fullres-50.yaml}"
OUTPUT="${2:-$PROJECT_ROOT/outputs/fullres-50}"
SCENE="$PROJECT_ROOT/data/scene_fullres"
if [[ ! -d "$SCENE" && -d "$PROJECT_ROOT/deployment/scene_fullres" ]]; then
  SCENE="$PROJECT_ROOT/deployment/scene_fullres"
fi
bash "$PROJECT_ROOT/scripts/run_validation.sh" --config "$CONFIG" \
  --scene-dir "$SCENE" --skip-bake --count 50 --output "$OUTPUT"
/data/anaconda3/envs/cloudrobo-habitat-gs/bin/python "$PROJECT_ROOT/scripts/summarize_results.py" \
  --results "$OUTPUT" --output "$OUTPUT/summary.html"
