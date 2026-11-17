#!/usr/bin/env bash
# 中文说明：兼容旧命令，转交通用 50 条批量入口。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$PROJECT_ROOT/configs/fullres-50.yaml}"
OUTPUT="${2:-$PROJECT_ROOT/outputs/fullres-50}"
SCENE="$PROJECT_ROOT/data/scene_fullres"
if [[ ! -d "$SCENE" && -d "$PROJECT_ROOT/deployment/scene_fullres" ]]; then
  SCENE="$PROJECT_ROOT/deployment/scene_fullres"
fi
bash "$PROJECT_ROOT/scripts/run_batch_50.sh" "$CONFIG" "$OUTPUT" "$SCENE"
