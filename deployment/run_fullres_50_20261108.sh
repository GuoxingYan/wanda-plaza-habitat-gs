#!/usr/bin/env bash
# 2026-11-08 的独立高清批次；重复执行时复用已完成路线。
set -euo pipefail
cd /data/projects/CloudRobo
# cron 是一次性计划（crontab 本身无年份字段）；触发后立即移除此标记行。
if crontab -l 2>/dev/null | grep -q '# cloudrobo-fullres-20261108'; then
  crontab -l 2>/dev/null | grep -v '# cloudrobo-fullres-20261108' | crontab -
fi
result_dir=/data/projects/CloudRobo/deployment/results_fullres/batch_20261108
mkdir -p "$result_dir"
exec > >(tee -a "$result_dir/run.log") 2>&1
printf '开始时间: %s\n' "$(date --iso-8601=seconds)"
bash deployment/run_validation.sh --config deployment/config_fullres_50.yaml \
  --skip-bake --count 50 --output "$result_dir"
/data/anaconda3/envs/cloudrobo-habitat-gs/bin/python deployment/summarize_results.py \
  --results "$result_dir" --output "$result_dir/summary.html"
printf '完成时间: %s\n' "$(date --iso-8601=seconds)"
