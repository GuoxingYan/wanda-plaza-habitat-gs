#!/usr/bin/env bash
# 将旧 deployment 本地资产目录原位移动到 data/；不会复制或删除文件。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${1:-$PROJECT_ROOT/deployment}"
TARGET_ROOT="$PROJECT_ROOT/data"
mkdir -p "$TARGET_ROOT"
for name in scene scene_crop scene_fullres scene_topdown_crop logs system-deps; do
  source_path="$SOURCE_ROOT/$name"
  target_path="$TARGET_ROOT/$name"
  if [[ -e "$source_path" && ! -e "$target_path" ]]; then
    mv -- "$source_path" "$target_path"
    printf 'moved %s -> %s\n' "$source_path" "$target_path"
  elif [[ -e "$source_path" && -e "$target_path" ]]; then
    printf 'skip (both exist): %s and %s\n' "$source_path" "$target_path"
  fi
done
for name in results_fullres results_full results_density_smoke pose_schema_smoke; do
  source_path="$SOURCE_ROOT/$name"
  target_path="$PROJECT_ROOT/outputs/$name"
  if [[ -e "$source_path" && ! -e "$target_path" ]]; then
    mv -- "$source_path" "$target_path"
    printf 'moved %s -> %s\n' "$source_path" "$target_path"
  elif [[ -e "$source_path" && -e "$target_path" ]]; then
    printf 'skip (both exist): %s and %s\n' "$source_path" "$target_path"
  fi
done
