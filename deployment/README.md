# 兼容目录

项目代码已整理到仓库顶层的 `src/`、`scripts/`、`configs/`、`tests/` 和 `docs/`。

历史生成的本地场景/结果若仍在此目录，可在确认目标目录后运行 `bash scripts/migrate_local_data.sh` 将其原位迁到 `data/`。旧运行命令 `bash deployment/run_validation.sh` 仍可用，它会转发到新入口。
