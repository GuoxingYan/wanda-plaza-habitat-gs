# Wanda Plaza / Habitat-GS

独立 Conda 环境：`cloudrobo-habitat-gs`，Python 3.12，CUDA toolkit 12.4，RTX 3090。

## 参数化运行

```bash
cd /data/projects/CloudRobo
conda activate cloudrobo-habitat-gs
./deployment/run_validation.sh
# 已有 NavMesh 时跳过烘焙
./deployment/run_validation.sh --skip-bake
# 只生成 NavMesh 并验证路径移动
./deployment/run_validation.sh --bake-only
# 指定 YAML 配置、场景或输出目录
./deployment/run_validation.sh --config deployment/config.yaml --scene-dir deployment/scene --output deployment/results_full
```

所有主要运行参数集中在 [config.yaml](config.yaml)，每个键都有中文说明。默认配置针对未裁剪的 `deployment/scene/`，线速度为 **3.0 m/s**，生成 **50 条**轨迹。可调整代理宽度/高度、坡度、爬升和 Recast 栅格参数，导航路线采样与距离限制，线速度（m/s）、角速度（rad/s）、帧率、D435 RGB 分辨率/FoV，以及输出项。代理半径按配置宽度的一半计算。指定起终点时两者都使用 Habitat Y-up 世界坐标 `[x,y,z]` 并且需要同时提供；`null` 时在可行走连通岛和 NavMesh 三角面中面积加权采样。

移动按 `linear_speed_mps / fps` 限制每帧前进距离，转向受 `angular_speed_radps / fps` 限制，位置经过 PathFinder 的 `try_step` 约束。逐帧姿态写入 `navigation_poses.json`：包括起点、终点、每帧代理/相机位置、时间戳、渲染耗时和相机朝向四元数。四元数顺序为 `x,y,z,w`，位置坐标系为 Habitat Y-up 世界坐标。

输入文件位于 `Wanda Plaza/`（链接到 `/data/data/Wanda Plaza`）。原文件不修改。

`prepare_scene.py` 默认用同一个源坐标包围盒 `X=[-50,10], Y=[-40,20]` 裁剪 mesh 和高斯，再将源 Z-up 转为 Habitat Y-up：`(x,y,z) -> (x,z,-y)`。mesh 三角面的三个顶点必须全部位于包围盒内；高斯中心也使用同一包围盒。位置、法线、高斯四元数、SH 1–3 阶颜色系数一起旋转。该 PLY 使用顶点色，没有 UV/外部贴图定义。

```bash
python deployment/prepare_scene.py
# 自定义同规格 crop：XMIN XMAX YMIN YMAX
python deployment/prepare_scene.py --crop -50 10 -40 20 --out deployment/scene_crop
```

完整场景 mesh 包含 11,479,664 个三角面。原始 Gaussian 有 35,225,246 个点；低密度预览版本按 stride=8 保留 4,403,156 个。轻量 crop 的实测资产包含 1,284,922 个 mesh 三角面和 870,725 个高斯，SH 不降阶。

## 验证输出

`scene/wanda.navmesh` 是完整场景实际烘焙结果。`results_full/route_*/validation.json` 为每条路线的实测报告；`trajectory_topdown.png` 展示 NavMesh、真实移动轨迹和起终点；`mesh_navmesh_topdown.png` 展示彩色 mesh 顶视并叠加半透明 NavMesh；`path.npy` 为 geodesic 转折点，`trajectory.npy` 为按速度控制且经过 `PathFinder.try_step` 过滤的移动轨迹；`navigation_poses.json` 记录完整逐帧位姿；`navigation_rgb.mp4` 和 `rgb_*.png` 来自 CUDA Gaussian Splatting。`scene_crop/` 与 `results_crop/` 保留轻量 crop 验证副本。

相机使用 D435 **RGB** 标称 HFOV=69°、640×360、安装高度1.5米，理想针孔，无畸变。不是深度相机的87°视场角，也不是实机逐台标定内参。默认坐标/mesh单位假定米。代理半径0.25米，高1.5米，爬升0.2米、坡度35°；NavMesh体素0.1×0.05米。路径导航是确定性geodesic跟随与NavMesh碰撞约束，不包含视觉策略、SLAM或运动控制器。

## 构建

`build.sh` 可重新构建当前源码。仅安装静态场景、导航与RGB所需依赖；关闭Bullet、GUI、Audio、Assimp，保留原生PLY/GLTF导入与CUDA渲染。GLM头文件解包在`system-deps/root`，不需要系统级安装。上游`src/cmake/dependencies.cmake`有一处补丁：仅当启用Assimp时将AssimpImporter加入Magnum静态插件列表。

下载、构建、转换及验证日志位于`logs/`。

参考：[Habitat-GS](https://github.com/zju3dv/habitat-gs)，[D435规格](https://www.realsenseai.com/products/stereo-depth-camera-d435/)。

## 批量生成与代码结构

`generation.trajectory_count` 控制生成的**轨迹条数**；每条轨迹输出至 `results_full/route_0000/`、`route_0001/` 等目录，种子依次为基础种子加索引。视频帧数由路径长度、线速度、角速度和 fps 自然决定；在转折点先原地转向，再沿最短路径前进。每个目录的 `effective_config.yaml` 是实际生效参数快照。固定起终点时多条结果会复用相同路线。

完整场景批量运行会在进程内共享一个 Gaussian 渲染 Simulator，避免每条路线重新加载约 1.1 GiB 的渲染 PLY。`--skip-bake` 复用已验证的 `scene/wanda.navmesh`；`--single` 只执行一条路线，`--count N` 临时覆盖 YAML 的生成条数，`--force` 重建已有结果。批量会识别有效的 `validation.json` 并跳过，失败条目写入 `failure.json` 后继续，全部成功时自动清除旧的 `failures.json`。

默认低密度完整空间范围 Gaussian 文件 `scene/wanda_crop.gs.ply` 来自原始 35,225,246 点的固定 stride=8 准备结果，共 4,403,156 点；mesh 保留约 1,148 万三角面。本地 `scene_fullres/wanda_crop.gs.ply` 保留全部 35,225,246 点，并完成同一路线 272 帧验证（见本地 `results_fullres/route_0000/validation.json`）。全密度地面视角的细节明显改善；全密度资产和验证图片均不上传 Git。

`visualization` 是可视化 YAML 配置分区，可设置轨迹颜色/线宽、NavMesh 颜色/透明度和输出 DPI；`output` 控制生成哪些图像和媒体。彩色 mesh 图是顶点颜色顶视投影用于对齐诊断，不是严格正射纠正的测绘 DOM，树冠和建筑会遮挡地面。

起终点算法先过滤面积不足的连通岛，再按岛面积选取岛、按 NavMesh 三角面面积加权抽样，并在三角形内用均匀重心坐标生成候选点。最后用 PathFinder 求真实测地距离，保留符合长度限制的路线。候选点都来自可行走表面；狭长弯曲场景仍可能需要增加 `sample_attempts`。手工坐标会投影到 NavMesh，且要求起终点在同一个连通岛。

目录职责：

- `deployment/config.yaml`：带详细中文注释的运行、生成和可视化参数。
- `deployment/wanda_nav/planning.py`：配置读取、面积加权路线选择、速度约束步进。
- `deployment/wanda_nav/visualization.py`：轨迹图和彩色模型叠加图。
- `deployment/wanda_nav/pipeline.py`：单条路线的烘焙、渲染、位姿和验证输出。
- `deployment/wanda_nav/cli.py`：参数入口和多条路线调度。
- `deployment/validate_navigation.py`：保留旧调用方式的兼容入口。
- `deployment/scene_crop/`、`deployment/results_crop/`、`deployment/logs/`：本地生成资产、结果及日志，均不上传原始数据或大型结果。

坐标约定：源模型 `(x,y,z)` 转为 Habitat `(x,z,-y)`，Y 轴向上；相机局部 -Z 朝前、+Y 朝上。JSON 朝向采用主动旋转的 `xyzw` 四元数，表示相机局部坐标到世界坐标的旋转；时间戳是模拟时间，不是渲染壁钟时间。


## 高分辨率可视化刷新

可复用已有路线，用 `bash deployment/run_validation.sh --visualize-only --scene-dir /data/projects/CloudRobo/deployment/scene_fullres --output /data/projects/CloudRobo/deployment/results_fullres` 单独刷新。不会重新烘焙 NavMesh 或运行视频循环。新增 Gaussian 图目前由该独立刷新入口生成。

YAML 中 `visualization.image_width_px/image_height_px` 默认 3600×3000；mesh 底图使用全部顶点按 0.06 米像素聚合颜色并做归一化邻域平滑，空白处保持白色。它是颜色诊断图，并非带遮挡测试的正射纹理产品，垂直方向不同表面的颜色可能混合。

Gaussian 底图使用真实 Habitat-GS 俯视透视相机；NavMesh 顶点根据自身高度投影到同一相机像素坐标，不可简单把透视图按 X/Z 平面拉伸叠加。顶部相机高度偏移和叠层透明度可在 YAML 中调整。图中被屋顶遮挡的地面仍可能绘制 NavMesh 诊断层，不代表地面可从相机直接看到。

**已知限制：**当前全密度高斯的高空俯视图存在严重拉丝，不作为合格的精细障碍底图；这不影响已验证的地面相机渲染。`gaussian_topdown_height_m` 表示相对于 NavMesh 最高点的高度偏移。独立刷新入口目前会逐条初始化高斯场景，批量刷新耗时较长。

mesh 底图第一行对应世界 Z 最大侧、第一列对应 X 最小侧，使用 `origin=upper` 与正常的米制 extent 配对；已修复此前只翻转底图造成的错位。另存 `mesh_topdown_base.png` 和 `mesh_topdown_coordinates.json`，记录完整栅格和坐标范围。`detail_regions_xz` 为局部放大范围列表，每项为 `[xmin,xmax,zmin,zmax]`，输出 `mesh_navmesh_detail_00.png` 等局部图。局部图复用相同世界坐标，不重新拉伸底图。

回归验证：`PYTHONPATH=deployment /data/anaconda3/envs/cloudrobo-habitat-gs/bin/python -m unittest discover -s deployment/tests -v`。四角颜色用例检查俯视方向及边界点；全场图已目视核对斜路、上/下转角和右侧通道。更多修改说明见 [CHANGELOG.md](CHANGELOG.md)。

此前 0.1m/0.05m 网格差异图：绿色表示共有区域，红色表示 0.05m 新增区域，蓝色表示仅 0.1m 存在区域；这不是可走/不可走的分类图。

低矮障碍需检查碰撞 mesh 是否包含柱体，以及柱体高度、宽度与 `max_climb_m`（当前 0.2m）、`cell_size_m`（0.1m）和 `cell_height_m`（0.05m）的关系。最大跨越高度以内的物体可能被当成台阶；区域面积过滤主要过滤可行走岛，不能直接用它解释柱体漏检。尚未定位的柱体不能仅凭底图颜色判断已被安全避让。
