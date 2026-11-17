`# 使用与运行说明

项目在 Wanda Plaza 室外场景上构建 NavMesh、筛选并规划路线，使用 Habitat-GS 输出 D435 RGB 视频、位姿 JSON 和诊断可视化。原始数据与大体积产物均保留在本地忽略目录。

## 环境和运行

已验证环境：Conda `cloudrobo-habitat-gs`、Python 3.12、CUDA 12.4、RTX 3090。运行入口会设置 CUDA、动态库和 Python 导入路径：

```bash
conda activate cloudrobo-habitat-gs
bash scripts/run_validation.sh --config configs/default.yaml
bash scripts/run_validation.sh --config configs/default.yaml --skip-bake --single
bash scripts/run_validation.sh --config configs/default.yaml --bake-only
```

默认输出到 `outputs/default/`。批处理可用 `--count N` 覆盖 YAML 中的轨迹数，用 `--force` 重建已有路线。续跑时会校验配置快照和必需产物，跳过已完成路线。全密度批次入口是 `scripts/run_fullres_50.sh`，也可指定配置和输出目录作为前两个参数。

测试命令：

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## 场景准备

默认原始输入路径是 `data/Wanda Plaza/`，它是指向 `/data/data/Wanda Plaza` 的链接。也可显式传入：

```bash
python scripts/prepare_scene.py --input "data/Wanda Plaza" \
  --crop -50 10 -40 20 --out data/scene_crop
```

mesh 和 Gaussian 使用同一源坐标包围盒；mesh 三角形仅在三个顶点均落入范围时保留。坐标从源 Z-up `(x,y,z)` 转成 Habitat Y-up `(x,z,-y)`，同时变换法线、高斯旋转和球谐颜色系数。默认 Gaussian stride 为 1；只有显式设置 `--gaussian-stride` 才会降低点数。

场景准备报告 `preparation.json` 会记录输入、裁剪范围、保留数量和坐标变换。生成场景目录包含 Gaussian PLY、碰撞 mesh、stage 配置和可选 NavMesh。

## 导航行为

- NavMesh 由 Recast/Habitat 从碰撞 mesh 烘焙；配置包含 cell size/height、代理宽度和高度、最大爬升及最大坡度。
- 起点和终点默认从 `cameras.json` 轨迹位置吸附到可行走面后选取；规划路径可要求保持在采集轨迹缓冲区内。
- 可按 NavMesh 连通岛上的采集轨迹密度过滤稀疏区域。该过滤仅影响候选区域，不会修改 NavMesh 文件。
- `motion.generation_profile: rich` 会在 Habitat PathFinder/NavMesh 上组合弧线掉头、S 型转向、弧线切向运动和盲端倒车等动作，并按批次种子确定动作组合。
- 正向运动段连接默认以 `motion.actions.tangent_connector_probability: 0.35` 的概率尝试切向连接。只有候选曲线同时通过 NavMesh 连通性、采集轨迹缓冲和最小转弯半径检查时才采用，并与相邻正向段合并做一次速度规划；失败时回退到普通平滑连接及必要的差速原地转向。此行为用于增加连续曲线运动样本，不保证每个拐角都切向连续。
- 速度、角速度、加减速度及横向加速度受 YAML 配置限制。`motion.generation_profile: legacy` 可切换回旧运动生成方式；差速模型不代表真实控制器，也不包含 SLAM 或视觉策略。

丰富运动参数在 `configs/*.yaml`，并带中文注释。常用项包括 `motion.minimum_turn_radius_m`、`motion.max_lateral_acceleration_mps2`、`motion.actions.curve_radius_m`、`motion.actions.tangent_connector_probability`、动作组合模式及重采样次数，以及 NavMesh 和采集轨迹缓冲设置。

## 输出文件

每条路线通常包含：

- `validation.json`：生效配置摘要、NavMesh/路径统计、逐帧验证结果。
- `navigation_poses.json`：起终点及每帧相机 `position` 和 `xyzw` 四元数，不记录 agent 字段。
- `trajectory_<route_id>.mcap`：ROS 2 CDR 编码的 RGB、米制 `32FC1` Depth、CameraInfo、PoseStamped，以及根据实际轨迹运动分类写入的轨迹标签；标签定义见 [README](../README.md#轨迹标签)。
- `depth_0000.png`：可选的首帧深度灰度预览；映射范围由 `camera.depth_preview` 控制，仅供查看，原始米制深度仍在 MCAP 中。
- `navigation_rgb.mp4`、`rgb_*.png`：全密度 Gaussian 的相机视角渲染。
- `trajectory_topdown.png`：NavMesh、轨迹、起终点与障碍标记。
- `mesh_navmesh_topdown.png`：彩色 mesh 顶视聚合与 NavMesh 叠图。
- `camera_trajectory_density_comparison.png`：采集轨迹密度过滤前后对照及独立轨迹视图。
- `path.npy`、`trajectory.npy`：内部规划折线和代理轨迹，用于复核及汇总图生成。

批量结果可用 `scripts/summarize_results.py` 汇总成 HTML：

```bash
python scripts/summarize_results.py --results outputs/default \
  --output outputs/default/summary.html
```

汇总页会将路线叠加在同一张 mesh 底图，并链接每条路线的视频。可视化图层和坐标方向见 [visualization.md](visualization.md)。

## 构建 Habitat-GS

源码位于本地忽略目录 `habitat-gs/`，构建脚本为 `scripts/build_habitat_gs.sh`，所需 GLM 头文件位于 `data/system-deps/`。构建依赖仅保留静态场景加载、导航和 CUDA Gaussian 渲染所需组件；设置见 `third_party/` 中补丁与资源配置。

## 本地目录迁移

- `data/scene/`：常规场景；`data/scene_fullres/`：全密度场景。
- `outputs/results_fullres/`：本地全分辨率批次、验证和避障演示。
- `data/logs/`：依赖下载、构建和运行日志。
- `outputs/`：默认运行结果、批次报告和汇总 HTML。

以上目录均被 Git 忽略，不上传原始 Gaussian、mesh、视频或大型 HTML。旧 `deployment/` 下的场景资产、日志和临时验证目录不会在代码迁移时自动移动；盘点后可运行 `bash scripts/migrate_local_data.sh` 将其原位归入 `data/` 或 `outputs/`。如果新旧同名目录都存在，脚本会跳过该项，不覆盖。全密度批次脚本在新场景目录缺失时会回退读取旧的 `deployment/scene_fullres/`。

## 避障原型限制

`avoidance.enabled` 当前提供固定障碍物的离线绕行验证：在原路线中放置圆柱 footprint，以栅格 A* 在 NavMesh 上绕行，并按 D435 参数将圆柱投影到 RGB 关键帧。它不是在线检测后实时重规划；障碍物与原场景几何的空间容纳/碰撞检查尚未完成，RGB 合成也没有深度遮挡融合。使用前应检查障碍坐标及净距报告，不能将该原型当成完整动态避障系统。
