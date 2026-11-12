#!/usr/bin/env bash
# 兼容旧命令；正式入口位于 scripts/run_validation.sh。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PROJECT_ROOT/scripts/run_validation.sh" "$@"
