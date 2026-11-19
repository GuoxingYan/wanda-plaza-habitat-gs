# Wanda Plaza / SanyaPo Habitat-GS 导航

室外场景导航数据生成与验证工具：从带颜色的 mesh 生成 NavMesh，结合 `cameras.json` 采集轨迹先验规划路线，并按运动学约束生成 D435 视角的 Gaussian RGB 视频、位姿 JSON、MCAP 和诊断图。丰富运动模式支持弧线掉头、S 弯、切向弧线及盲端倒车；正向段连接会尝试切向连续运动，候选不满足 NavMesh 或轨迹缓冲约束时回退到常规连接。

## 快速开始

项目依赖 `data/Wanda Plaza/` 原始输入链接（当前指向 `/data/data/Wanda Plaza`）和已构建的 Habitat-GS 环境 `cloudrobo-habitat-gs`。本地生成场景与原始数据链接保存在被 Git 忽略的 `data/`，运行结果和报告保存在 `outputs/`。

```bash
cd /data/projects/CloudRobo
conda activate cloudrobo-habitat-gs

# 由原始 mesh 与 Gaussian PLY 准备高分辨率场景；首次运行时执行
python scripts/prepare_scene.py --out data/scene_fullres

# 准备 NavMesh（首次运行或参数改变后运行一次）
bash scripts/run_validation.sh --config configs/ackermann-50.yaml \
  --scene-dir data/scene_fullres --bake-only --output outputs/ackermann-50

# 阿克曼底盘生成 50 条 MCAP，并在成功后生成汇总 HTML
bash scripts/run_batch_50.sh configs/ackermann-50.yaml \
  outputs/ackermann-50 data/scene_fullres
# 输出：outputs/ackermann-50/route_0000..route_0049/*.mcap
#       outputs/ackermann-50/summary.html

# SanyaPo 原始数据：先按 Gaussian 数量自动递归切块并落盘缓存
/data/anaconda3/envs/cloudrobo-habitat-gs/bin/python \
  scripts/split_sanyapo_scene.py --input data/sanyapo \
  --output data/sanyapo/tiles --max-gaussians 15000000
# 每个 tile 保存 mesh.ply、point_cloud.ply、cameras.json 和 bounds.json，
# data/sanyapo/tiles/manifest.json 存储清单；输入不变时重复执行会直接复用缓存。

# 使用默认配置生成一条路线并验证
bash scripts/run_validation.sh --config configs/default.yaml --single

# 差速底盘全密度 50 条批量生成与 HTML 汇总
bash scripts/run_fullres_50.sh

# 运行测试
PYTHONPATH=. python -m unittest discover -s tests -v
```

### 配置示例

`configs/examples/` 提供可直接运行的完整示例配置，`configs/ackermann-50.yaml` 是 50 条阿克曼 MCAP 批次配置：

```bash
# 阿克曼底盘单条预验证：限制最小转弯半径，不能原地转向
bash scripts/run_validation.sh --config configs/examples/ackermann.yaml --single

# 差速底盘旧行为：legacy 运动生成流程，用于和 rich 曲线运动对照
bash scripts/run_validation.sh --config configs/examples/legacy-differential.yaml --single

# 动态避障：在路线中段放置圆柱并生成绕行轨迹
bash scripts/run_validation.sh --config configs/examples/avoidance-demo.yaml --single
```

示例文件均为完整配置，可直接复制后修改。重点对照 `motion.model`、`motion.generation_profile` 和 `avoidance.enabled`；运行前请根据本机数据位置调整 `scene.directory` 与 `navigation.camera_trajectory_file`。

### 批量生成与汇总

`scripts/run_batch_50.sh <配置> <输出目录> <场景目录>` 是通用 50 条入口。它复用已经存在的 NavMesh，逐条生成路线，并在整批成功后调用 `scripts/summarize_results.py` 写出 `<输出目录>/summary.html`。脚本不会自动烘焙 NavMesh；首次准备场景时先用 `--bake-only` 烘焙，再运行批次。批次可中断后用同一命令续跑，已完成且文件齐全的路线会跳过；不要在输出目录中混用不同配置。需要从头重建时使用底层入口的 `--force`，它会覆盖已有路线结果。

阿克曼 50 条推荐流程使用 `configs/ackermann-50.yaml`：它打开 MCAP，保留相机轨迹先验过滤，并关闭耗时较高且不影响 MCAP 内容的俯视图和密度对比图。每条 MCAP 包含原始 RGB、32 位米制深度、深度预览、位姿、相机信息和轨迹标签；深度数据会使 MCAP 显著大于 MP4。若不需要 Foxglove 深度预览，可在配置中将 `camera.depth_preview.enabled` 设为 `false` 以减小文件体积。阿克曼约束下部分丰富动作路线可能规划失败；入口会保留失败报告并返回错误，不会悄悄切换到差速或 legacy 模式。调整最小转弯半径、动作约束或失败路线后，可用同一命令续跑。

只重建汇总页而不重新生成数据：

```bash
/data/anaconda3/envs/cloudrobo-habitat-gs/bin/python scripts/summarize_results.py \
  --results outputs/ackermann-50 --output outputs/ackermann-50/summary.html
```

### Shell 脚本说明

`scripts/` 下的 Shell 脚本分别负责以下工作：

| 脚本 | 作用 | 典型用法 / 注意事项 |
| --- | --- | --- |
| `run_validation.sh` | 设置 Habitat-GS 所需的 Python、CUDA 动态库和线程环境，然后调用导航验证 CLI。支持单条、批量、NavMesh 烘焙、只刷新可视化等 CLI 参数。 | `bash scripts/run_validation.sh --config configs/default.yaml --single`。这是底层运行入口，不会自动生成批次 HTML。 |
| `run_batch_50.sh` | 通用 50 条批次入口：检查配置、场景和 NavMesh，调用 `run_validation.sh` 生成数据，成功后调用 Python 汇总器生成 `summary.html`。 | `bash scripts/run_batch_50.sh <配置.yaml> <输出目录> <场景目录>`。可中断后续跑；汇总只在整批成功退出后执行。 |
| `run_fullres_50.sh` | 兼容旧命令的全密度差速 50 条入口，默认选择 `configs/fullres-50.yaml` 和 `data/scene_fullres`，实际工作转交给 `run_batch_50.sh`。 | `bash scripts/run_fullres_50.sh [配置.yaml] [输出目录]`。新批次建议直接用通用入口，以便显式选择底盘和场景。 |
| `build_habitat_gs.sh` | 在项目指定 Conda/CUDA 环境中，从 `third_party/habitat-gs` 构建并安装 Habitat-GS。 | 仅在首次搭建或需要重建 Habitat-GS 依赖时运行；会执行本地软件包安装，日常生成数据不需要运行。 |
| `migrate_local_data.sh` | 将旧 `deployment/` 下列出的本地场景、日志和结果目录移动到 `data/`，并跳过目标位置已存在的项目。 | `bash scripts/migrate_local_data.sh [旧 deployment 目录]`。这是文件移动操作；运行前先检查源/目标目录及脚本列出的目录，不要用于生成数据。 |

场景准备和结果汇总本身目前由 Python 入口完成：`scripts/prepare_scene.py` 准备 Wanda 场景，`scripts/split_sanyapo_scene.py` 为 SanyaPo 做递归切分缓存，`scripts/summarize_results.py` 单独重建 HTML；这些 Python 入口没有额外的 `.sh` 包装脚本。

### SanyaPo 切分规则与限制

`split_sanyapo_scene.py` 以 Gaussian 的 XY 分布为依据递归二分：块内数量不超过 `--max-gaussians` 就停止，否则沿较长轴按中位数继续切分，直到所有块满足上限或触发 `--min-size`。每个 mesh 三角面按质心唯一归属 tile，跨边界三角形会复制它所需的顶点；相机按同一 XY 规则写入对应 `cameras.json`。这一步只生成原始坐标的缓存块，不会自动生成 Habitat `stage_config` 或 NavMesh。

SanyaPo 当前 `point_cloud.ply` 是 17 个 float32 属性的 Gaussian 格式，而仓库现有 `prepare_scene.py` 面向 Wanda 的 62 列 SH 格式。因此不能直接把 SanyaPo tile 当作 `data/scene_fullres` 运行；需要使用支持该 17 列格式的 Habitat-GS 资产转换器，完成坐标转换、stage 配置和 NavMesh 后，再调用 `run_batch_50.sh` 生成 MCAP 与 `summary.html`。MCAP 中 RGB 已使用 ROS 2 `sensor_msgs/CompressedImage` JPEG 质量 90，深度仍为米制 `32FC1`。

## 目录

```text
configs/       运行配置，所有参数带中文说明；examples/ 提供可直接运行的学习示例
src/           Python 实现模块（CLI、规划、运动、渲染、可视化、避障）
scripts/       场景准备、运行、构建和结果汇总入口
tests/         单元与回归测试
docs/          完整使用说明、可视化说明、位姿规范和变更记录
third_party/   Habitat-GS 构建补丁及资源配置
data/          本地生成的场景、缓存、日志和诊断批次（忽略）
outputs/       默认运行结果（忽略）
```

Python 源码按你的目录要求直接放在 `src/`，没有额外的 `src/wanda_nav/` 层级。各功能模块通过 `src` 包名导入。

## 输入与坐标

原始数据不进入 Git。`scripts/prepare_scene.py` 对 mesh 和 Gaussian PLY 使用相同裁剪范围，并将源 Z-up 坐标 `(x,y,z)` 转为 Habitat Y-up `(x,z,-y)`。更多参数、验证产物、运动模型及已知限制见 [完整使用说明](docs/deployment.md)。图像颜色、叠加图和坐标轴含义见 [可视化说明](docs/visualization.md)。

## 运动生成

`motion.generation_profile: rich` 使用可复现的批次动作排程生成弧线掉头、S 型转向、弧线切向运动和盲端倒车，并在 NavMesh 与 `cameras.json` 缓冲区内验证路线。差速底盘和阿克曼底盘都可以使用这些连续曲线动作；差异在于差速底盘可以在需要时低速原地调向，而阿克曼底盘必须满足 `motion.minimum_turn_radius_m`，也不能原地转向。正向运动段之间默认以 `motion.actions.tangent_connector_probability: 0.35` 的概率尝试匹配端点切线；连接候选通过 NavMesh、轨迹先验和底盘曲率约束检查后，会合并做统一速度规划。候选不可行时保留常规平滑连接及差速底盘的朝向调整行为，因此不保证每个拐角均连续转向。设为 `legacy` 可切换到旧路线运动模式。相关参数见 `configs/default.yaml`、`configs/fullres-5.yaml` 和 `configs/fullres-50.yaml`。

### 轨迹标签

标签按整条轨迹汇总写入 `validation.json`、`navigation_poses.json` 和 MCAP 的 `/trajectory/tags`，一条轨迹可以同时包含多个标签。标签依据实际逐帧位姿及运动事件判定，不是生成配置的默认值：

| 标签 | 含义与判定 |
| --- | --- |
| `straight_forward` | 朝向变化小于 5°、连续前进至少 2 米的直线段。 |
| `straight_backward` | 朝向变化小于 5°、连续倒车至少 2 米的直线段。 |
| `curved_trajectory_minor` | 有运动事件记录的弧线段，首尾净朝向变化为 15° 至小于 45°。 |
| `curved_trajectory_moderate` | 弧线段首尾净朝向变化为 45° 至小于 90°。 |
| `curved_trajectory_major` | 弧线段首尾净朝向变化不小于 90°，例如弧线掉头。 |
| `in_place_rotation_minor` | 原地转向 15° 至小于 45°。 |
| `in_place_rotation_moderate` | 原地转向 45° 至小于 90°。 |
| `in_place_rotation_major` | 原地转向不小于 90°。 |
| `elevation_change` | 轨迹位姿的最高与最低高度差不小于 0.15 米。 |

曲线标签只对生成器记录的弧线掉头、切向弧线和 S 弯事件判定；原地转向单独标为 `in_place_rotation_*`。没有达到判定阈值的类别不会写入，因而标签列表可能为空。

批量结果的 HTML 汇总保存在相应输出目录（例如 `outputs/ackermann-50/summary.html`），属于本地输出，不会随仓库发布。

## 版本

当前版本标签：`v0.7.0-complex-turning`。此前版本包括 `v0.6.0-refactored`（项目目录重构）和 `v0.5.0-navmesh-complete`（NavMesh 导航流程打通）。功能变更见 [CHANGELOG](docs/changelog.md)。
