#!/usr/bin/env bash
# 中文说明：按指定完整配置生成 50 条路线，并在成功后生成 HTML 汇总。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$PROJECT_ROOT/configs/fullres-50.yaml}"
OUTPUT="${2:-$PROJECT_ROOT/outputs/batch-50}"
SCENE="${3:-$PROJECT_ROOT/data/scene_fullres}"
PYTHON="${PYTHON:-/data/anaconda3/envs/cloudrobo-habitat-gs/bin/python}"

if [[ ! -f "$CONFIG" ]]; then
  printf '配置文件不存在: %s\n' "$CONFIG" >&2
  exit 2
fi
if [[ ! -f "$SCENE/wanda.stage_config.json" || ! -f "$SCENE/wanda.navmesh" ]]; then
  printf '场景目录缺少 wanda.stage_config.json 或 wanda.navmesh: %s\n' "$SCENE" >&2
  printf '请先准备场景并生成/放置 NavMesh。\n' >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  printf 'Python 环境不可执行: %s（可通过 PYTHON=/path/to/python 覆盖）\n' "$PYTHON" >&2
  exit 2
fi

bash "$PROJECT_ROOT/scripts/run_validation.sh" --config "$CONFIG" \
  --scene-dir "$SCENE" --skip-bake --count 50 --output "$OUTPUT"
"$PYTHON" "$PROJECT_ROOT/scripts/summarize_results.py" \
  --results "$OUTPUT" --output "$OUTPUT/summary.html"
