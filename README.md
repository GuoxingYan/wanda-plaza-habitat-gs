# Wanda Plaza Habitat-GS 导航

室外场景导航数据生成与验证工具：从带颜色的 mesh 生成 NavMesh，结合 `cameras.json` 采集轨迹先验规划路线，按运动学约束生成 D435 视角的 Gaussian RGB 视频、位姿 JSON 和诊断图。

## 快速开始

项目依赖 `data/Wanda Plaza/` 原始输入链接（当前指向 `/data/data/Wanda Plaza`）和已构建的 Habitat-GS 环境 `cloudrobo-habitat-gs`。本地生成场景与原始数据链接保存在被 Git 忽略的 `data/`，运行结果和报告保存在 `outputs/`。

```bash
cd /data/projects/CloudRobo
conda activate cloudrobo-habitat-gs

# 由原始 mesh 与 Gaussian PLY 准备场景
python scripts/prepare_scene.py --out data/scene_crop

# 使用默认配置生成一条路线并验证
bash scripts/run_validation.sh --config configs/default.yaml --single

# 全密度 50 条批量生成与 HTML 汇总（需先准备 data/scene_fullres）
bash scripts/run_fullres_50.sh

# 运行测试
PYTHONPATH=. python -m unittest discover -s tests -v
```

## 目录

```text
configs/       运行配置，所有参数带中文说明
src/           Python 实现模块（CLI、规划、运动、渲染、可视化、避障）
scripts/       场景准备、运行、构建和结果汇总入口
tests/         单元与回归测试
docs/          完整使用说明、可视化说明、位姿规范和变更记录
third_party/   Habitat-GS 构建补丁及资源配置
data/          本地生成的场景、缓存、日志和诊断批次（忽略）
outputs/       默认运行结果（忽略）
```

旧 `deployment/` 下的本地生成资产不会随代码自动移动；盘点后可执行 `bash scripts/migrate_local_data.sh` 进行原位迁移。原始数据链接位于 `data/Wanda Plaza`。

Python 源码按你的目录要求直接放在 `src/`，没有额外的 `src/wanda_nav/` 层级。各功能模块通过 `src` 包名导入。

## 输入与坐标

原始数据不进入 Git。`scripts/prepare_scene.py` 对 mesh 和 Gaussian PLY 使用相同裁剪范围，并将源 Z-up 坐标 `(x,y,z)` 转为 Habitat Y-up `(x,z,-y)`。更多参数、验证产物、运动模型及已知限制见 [完整使用说明](docs/deployment.md)。图像颜色、叠加图和坐标轴含义见 [可视化说明](docs/visualization.md)。

当前本地全密度 50 条结果汇总（如果已生成）：[HTML 汇总](outputs/results_fullres/batch_20261109_50/summary.html)。该路径属于本地输出，不会随仓库发布。

## 版本

NavMesh 首个完整打通版本：`v0.5.0-navmesh-complete`。功能变更见 [CHANGELOG](docs/changelog.md)。
