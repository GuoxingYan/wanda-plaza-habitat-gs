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
./deployment/run_validation.sh --config deployment/config.yaml --scene-dir deployment/scene_crop --output deployment/results_crop
```

所有主要运行参数集中在 [config.yaml](config.yaml)，每个键都有中文说明。可调整代理宽度/高度、坡度、爬升和 Recast 栅格参数，导航路线采样与距离限制，线速度（m/s）、角速度（rad/s）、帧率、D435 RGB 分辨率/FoV，以及输出项。代理半径按配置宽度的一半计算。指定起终点时两者都使用 Habitat Y-up 世界坐标 `[x,y,z]` 并且需要同时提供；`null` 时在最大连通岛内确定性随机采样。

移动按 `linear_speed_mps / fps` 限制每帧前进距离，转向受 `angular_speed_radps / fps` 限制，位置经过 PathFinder 的 `try_step` 约束。逐帧姿态写入 `navigation_poses.json`：包括起点、终点、每帧代理/相机位置、时间戳、渲染耗时和相机朝向四元数。四元数顺序为 `x,y,z,w`，位置坐标系为 Habitat Y-up 世界坐标。

输入文件位于 `Wanda Plaza/`（链接到 `/data/data/Wanda Plaza`）。原文件不修改。

`prepare_scene.py` 默认用同一个源坐标包围盒 `X=[-50,10], Y=[-40,20]` 裁剪 mesh 和高斯，再将源 Z-up 转为 Habitat Y-up：`(x,y,z) -> (x,z,-y)`。mesh 三角面的三个顶点必须全部位于包围盒内；高斯中心也使用同一包围盒。位置、法线、高斯四元数、SH 1–3 阶颜色系数一起旋转。该 PLY 使用顶点色，没有 UV/外部贴图定义。

```bash
python deployment/prepare_scene.py
# 自定义同规格 crop：XMIN XMAX YMIN YMAX
python deployment/prepare_scene.py --crop -50 10 -40 20 --out deployment/scene_crop
```

默认验证资产包含 1,284,922 个 mesh 三角面和 870,725 个高斯，均来自上述同一 60×60 米 crop，SH 不降阶。

## 验证输出

`scene_crop/wanda_crop.navmesh` 是实际烘焙结果。`results_crop/validation.json` 为实测报告；`trajectory_topdown.png` 展示 NavMesh、真实移动轨迹和起终点；`mesh_navmesh_topdown.png` 展示裁剪 mesh 顶点颜色并叠加半透明 NavMesh；`path.npy` 为 geodesic 转折点，`trajectory.npy` 为按速度控制且经过 `PathFinder.try_step` 过滤的移动轨迹；`navigation_poses.json` 记录完整逐帧位姿；`navigation_rgb.mp4` 和 `rgb_*.png` 来自 CUDA Gaussian Splatting。

相机使用 D435 **RGB** 标称 HFOV=69°、640×360、安装高度1.5米，理想针孔，无畸变。不是深度相机的87°视场角，也不是实机逐台标定内参。默认坐标/mesh单位假定米。代理半径0.25米，高1.5米，爬升0.2米、坡度35°；NavMesh体素0.1×0.05米。路径导航是确定性geodesic跟随与NavMesh碰撞约束，不包含视觉策略、SLAM或运动控制器。

## 构建

`build.sh` 可重新构建当前源码。仅安装静态场景、导航与RGB所需依赖；关闭Bullet、GUI、Audio、Assimp，保留原生PLY/GLTF导入与CUDA渲染。GLM头文件解包在`system-deps/root`，不需要系统级安装。上游`src/cmake/dependencies.cmake`有一处补丁：仅当启用Assimp时将AssimpImporter加入Magnum静态插件列表。

下载、构建、转换及验证日志位于`logs/`。

参考：[Habitat-GS](https://github.com/zju3dv/habitat-gs)，[D435规格](https://www.realsenseai.com/products/stereo-depth-camera-d435/)。

## 批量生成与代码结构

`generation.trajectory_count` 控制生成的**轨迹条数**；每条轨迹输出至 `results_crop/route_0000/`、`route_0001/` 等目录，种子依次为基础种子加索引。视频帧数由路径长度、线速度、角速度和 fps 自然决定；在转折点先原地转向，再沿最短路径前进。每个目录的 `effective_config.yaml` 是实际生效参数快照。固定起终点时多条结果会复用相同路线。

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
