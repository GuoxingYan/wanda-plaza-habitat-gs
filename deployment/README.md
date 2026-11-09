# Wanda Plaza / Habitat-GS

室外场景导航数据生成与验证：从彩色 mesh 烘焙/加载 NavMesh，结合 `cameras.json` 采集轨迹筛选可用区域，生成带运动学约束的路线、D435 视角 Gaussian RGB 视频、位姿 JSON 和诊断可视化。

## 可视化示例

![采集轨迹密度过滤：原始 NavMesh、过滤 NavMesh 与采集路线独立视图](results_density_smoke/route_0000/camera_trajectory_density_comparison.png)

![单条路线在 NavMesh 中的运行轨迹](results_density_smoke/route_0000/trajectory_topdown.png)

![彩色 mesh 纹理与 NavMesh 叠加](results_density_smoke/route_0000/mesh_navmesh_topdown.png)

最新 50 条全密度结果完成后，总览图和 HTML 报告位于 `results_fullres/batch_20261109_50/`。HTML 顶部的总览将 50 条导航轨迹以不同颜色叠加在同一张彩色 mesh 底图中，页面下方列出逐条结果与视频链接。

独立 Conda 环境：`cloudrobo-habitat-gs`，Python 3.12，CUDA toolkit 12.4，RTX 3090。

首次阅读建议从本文件开始；参数及图像判读见 [VISUALIZATION.md](VISUALIZATION.md)，逐次代码变更见 [CHANGELOG.md](CHANGELOG.md)。原始数据位于 `Wanda Plaza/`，该目录被 Git 忽略。

## 参数化运行

```bash
cd /data/projects/CloudRobo
conda activate cloudrobo-habitat-gs
bash deployment/run_validation.sh
# 已有 NavMesh 时跳过烘焙
bash deployment/run_validation.sh --skip-bake
# 只生成 NavMesh 并验证路径移动
bash deployment/run_validation.sh --bake-only
# 指定 YAML 配置、场景或输出目录
bash deployment/run_validation.sh --config deployment/config.yaml --scene-dir deployment/scene --output deployment/results_full
```

### 生成全密度 50 条路线并汇总 HTML

```bash
bash deployment/run_validation.sh \
  --config deployment/config_fullres_50.yaml \
  --scene-dir deployment/scene_fullres \
  --output deployment/results_fullres/batch_20261109_50 \
  --skip-bake
/data/anaconda3/envs/cloudrobo-habitat-gs/bin/python deployment/summarize_results.py \
  --results deployment/results_fullres/batch_20261109_50 \
  --output deployment/results_fullres/batch_20261109_50/summary.html
```

汇总器会额外生成 `all_routes_mesh_overlay.png`：每条路线使用独立颜色，蓝色圆点表示起点、橙色三角表示终点；该总览图和各路线可视化以内嵌图片方式显示在 HTML 中，视频仍使用相对链接，避免复制媒体文件。

所有主要运行参数集中在 [config.yaml](config.yaml)，每个键都有中文说明。默认配置使用未裁剪的 `deployment/scene/`，线速度为 **3.0 m/s**，生成 **50 条**轨迹。指定起终点时使用 Habitat Y-up 世界坐标 `[x,y,z]`，必须同时提供；否则默认优先从 `cameras.json` 采集位置投影到 NavMesh 后抽取起终点。

默认采集轨迹先验开启：路线每 0.5 m 采样检查距 `cameras.json` 相机中心轨迹的 X/Z 平面距离，要求全路径都位于默认 **10 m 缓冲区**内。采集轨迹本身按源 Z-up 转为 Habitat `(x,z,-y)`。可以在 YAML 中关闭先验、改缓冲半径或关闭端点优先；详细解释见 [VISUALIZATION.md](VISUALIZATION.md)。

默认还开启连通岛采集密度后处理：将成功吸附到各 NavMesh 岛的 `cameras.json` 轨迹点计数，只有同时满足 `camera_trajectory_density_min_points` 和 `camera_trajectory_density_min_points_per_100m2` 的岛才参与路线起终点选择。该开关及两个阈值位于 `navigation` 配置；每次运行会输出 `camera_trajectory_density_comparison.png` 三列对比图：关闭/开启过滤的 NavMesh 视图分别显示导航轨迹与起终点，第三列单独显示采集轨迹；同一视频帧的多方向相机位置会先合并，轨迹按视频片段分别连接。浅红色标记会被过滤的稀疏区域。图中使用彩色 mesh 纹理底图，并在 `validation.json` 记录逐岛面积、点数、密度和保留状态。该后处理不重写 NavMesh 文件。

运动学使用 `motion.model: differential_drive`：每段直线采用固定帧率三角/梯形速度曲线，受最大线速度、加速度和制动减速度约束；路径转折处停车后原地转向，受最大角速度约束。位置仍经 PathFinder `try_step` 限制在 NavMesh 上。该模型适用于差速/全向底盘的简化仿真，不适用于汽车式最小转弯半径约束。`navigation_poses.json` 只包含相机信息：`coordinate_frame` 描述 Y-up 坐标和单位，`route.start` / `route.goal` 各保存相机 `position` 和 `xyzw`，`path_length` 记录路径长度（米）；`camera` 保存 D435 RGB 参数，`frames` 保存帧序号、模拟时间、相机 `position` 和 `xyzw` 四元数。四元数顺序为 x、y、z、w。代理位置和运动学参数只用于内部导航及可行性校验，不写入位姿 JSON。

输入文件位于 `Wanda Plaza/`（链接到 `/data/data/Wanda Plaza`）。原文件不修改。

`prepare_scene.py` 默认用同一个源坐标包围盒 `X=[-50,10], Y=[-40,20]` 裁剪 mesh 和高斯，再将源 Z-up 转为 Habitat Y-up：`(x,y,z) -> (x,z,-y)`。mesh 三角面的三个顶点必须全部位于包围盒内；高斯中心也使用同一包围盒。位置、法线、高斯四元数、SH 1–3 阶颜色系数一起旋转。该 PLY 使用顶点色，没有 UV/外部贴图定义。

```bash
python deployment/prepare_scene.py
# 自定义同规格 crop：XMIN XMAX YMIN YMAX
python deployment/prepare_scene.py --crop -50 10 -40 20 --out deployment/scene_crop
```

完整场景 mesh 包含 11,479,664 个三角面。原始 Gaussian 有 35,225,246 个点；低密度预览版本按 stride=8 保留 4,403,156 个。轻量 crop 的实测资产包含 1,284,922 个 mesh 三角面和 870,725 个高斯，SH 不降阶。

## 验证输出

`scene/wanda.navmesh` 是完整场景 NavMesh。`results_full/route_*/validation.json` 为逐路线报告；`trajectory_topdown.png` 展示 NavMesh、轨迹和起终点；`mesh_navmesh_topdown.png` 是彩色 mesh 聚合底图与 NavMesh 叠图；`mesh_topdown_base.png` 和 `mesh_topdown_coordinates.json` 提供纯底图及坐标映射；`mesh_navmesh_detail_*.png` 是 YAML 指定区域的放大图。将 `output.gaussian_navmesh_overlay` 打开后才额外生成 `gaussian_topdown_raw.png` 和 `gaussian_navmesh_topdown.png`，因为高分辨率俯视 Gaussian 会增加显存和渲染时间。`path.npy` 为 geodesic 转折点，`trajectory.npy` 为内部逐帧代理轨迹；`navigation_poses.json` 只记录逐帧相机位姿，不含代理位置；`navigation_rgb.mp4` 和 `rgb_*.png` 来自 CUDA Gaussian Splatting。

相机使用 D435 **RGB** 标称 HFOV=69°、1280×720、安装高度1.5米，理想针孔，无畸变。不是深度相机的87°视场角，也不是实机逐台标定内参。默认坐标/mesh单位假定米。NavMesh 栅格为 0.05×0.05 米，代理总宽度 0.75 米，额外障碍净距 0.15 米，因此烘焙半径为 `0.75/2+0.15=0.525 m`；代理高度1.5米、爬升0.2米、坡度35°。这会主动收缩窄路，使路线中心离墙面、低矮柱体等障碍更远。路径导航是确定性 geodesic 跟随与 NavMesh 碰撞约束，不包含视觉策略、SLAM 或运动控制器。

## 构建

`build.sh` 可重新构建当前源码。仅安装静态场景、导航与RGB所需依赖；关闭Bullet、GUI、Audio、Assimp，保留原生PLY/GLTF导入与CUDA渲染。GLM头文件解包在`system-deps/root`，不需要系统级安装。上游`src/cmake/dependencies.cmake`有一处补丁：仅当启用Assimp时将AssimpImporter加入Magnum静态插件列表。

下载、构建、转换及验证日志位于`logs/`。

参考：[Habitat-GS](https://github.com/zju3dv/habitat-gs)，[D435规格](https://www.realsenseai.com/products/stereo-depth-camera-d435/)。

## 批量生成与代码结构

`generation.trajectory_count` 控制生成的**轨迹条数**；每条轨迹输出至 `results_full/route_0000/`、`route_0001/` 等目录，种子依次为基础种子加索引。视频帧数由路径长度、线速度、角速度和 fps 自然决定；在转折点先原地转向，再沿最短路径前进。每个目录的 `effective_config.yaml` 是实际生效参数快照。固定起终点时多条结果会复用相同路线。

批量续跑会根据 `output` 开关判断必需文件：关闭视频时不要求 MP4 或非零 RGB 帧计数，关闭位姿或某类俯视图时也不会因缺少对应文件而重复生成。场景级 NavMesh 三角面、mesh 彩色栅格和采集轨迹吸附候选会按输入文件指纹缓存；修改 NavMesh、mesh、`cameras.json` 或相关可视化参数后自动失效重建。隐藏 `.npz` 缓存保存在场景目录，不属于可交付结果。

完整场景批量运行会在进程内共享一个 Gaussian 渲染 Simulator，避免每条路线重新加载约 1.1 GiB 的渲染 PLY。`--skip-bake` 复用已验证的 `scene/wanda.navmesh`；`--single` 只执行一条路线，`--count N` 临时覆盖 YAML 的生成条数，`--force` 重建已有结果。批量会识别有效的 `validation.json` 并跳过，失败条目写入 `failure.json` 后继续，全部成功时自动清除旧的 `failures.json`。

默认低密度完整空间范围 Gaussian 文件 `scene/wanda_crop.gs.ply` 来自原始 35,225,246 点的固定 stride=8 准备结果，共 4,403,156 点；mesh 保留约 1,148 万三角面。本地 `scene_fullres/wanda_crop.gs.ply` 保留全部 35,225,246 点，并完成同一路线 272 帧验证（见本地 `results_fullres/route_0000/validation.json`）。全密度地面视角的细节明显改善；全密度资产和验证图片均不上传 Git。

`visualization` 是可视化 YAML 配置分区，可设置图像 3600×3000 分辨率、局部放大范围、颜色/透明度和输出 DPI；`output` 控制生成哪些图像和媒体。彩色 mesh 图是顶点颜色栅格聚合，树冠和建筑会与地面同平面投影，不是带遮挡计算的正射影像。图片判读和图层含义见 [VISUALIZATION.md](VISUALIZATION.md)。

路线算法默认在相机采集轨迹位置中选起终点、吸附到可行走面，并用 PathFinder 求测地路线；路线必须满足配置距离范围并全程处于轨迹缓冲区内。关闭端点优先时回退到 NavMesh 三角面面积加权采样，但启用先验时路径仍须在缓冲区内。手工坐标会投影到 NavMesh，且要求起终点同岛、路线长度合格及先验条件满足。

目录职责：

- `deployment/config.yaml`：带详细中文注释的运行、生成和可视化参数。
- `deployment/wanda_nav/planning.py`：配置读取、面积加权路线选择、速度约束步进。
- `deployment/wanda_nav/visualization.py`：轨迹图和彩色模型叠加图。
- `deployment/wanda_nav/pipeline.py`：单条路线的烘焙、渲染、位姿和验证输出。
- `deployment/wanda_nav/cli.py`：参数入口和多条路线调度。
- `deployment/validate_navigation.py`：保留旧调用方式的兼容入口。
- `deployment/CHANGELOG.md`：日期化功能修改、bug 修复和验证记录。
- `deployment/VISUALIZATION.md`：图层说明、局部图判读和采集轨迹先验。
- `deployment/tests/`：坐标方向、透视投影和先验逻辑的回归测试。
- `deployment/scene_crop/`、`deployment/results_crop/`、`deployment/logs/`：本地生成资产、结果及日志，均不上传原始数据或大型结果。

坐标约定：源模型 `(x,y,z)` 转为 Habitat `(x,z,-y)`，Y 轴向上；相机局部 -Z 朝前、+Y 朝上。JSON 朝向采用主动旋转的 `xyzw` 四元数，表示相机局部坐标到世界坐标的旋转；时间戳是模拟时间，不是渲染壁钟时间。


## 高分辨率可视化刷新

可复用已有路线，用 `bash deployment/run_validation.sh --visualize-only --scene-dir /data/projects/CloudRobo/deployment/scene_fullres --output /data/projects/CloudRobo/deployment/results_fullres` 单独刷新。它读取每条路线的 `effective_config.yaml`，不重新烘焙 NavMesh 或运行行走视频；会为 Gaussian 高空底图单独初始化一次场景。正常导航 pipeline 也会生成同样的 Gaussian 图。

YAML 中 `visualization.image_width_px/image_height_px` 默认 3600×3000；mesh 底图使用全部顶点按 0.06 米像素聚合颜色并做归一化邻域平滑，空白处保持白色。它是颜色诊断图，并非带遮挡测试的正射纹理产品，垂直方向不同表面的颜色可能混合。

Gaussian 底图使用真实 Habitat-GS 俯视透视相机；NavMesh 顶点根据自身高度投影到同一相机像素坐标，不可简单把透视图按 X/Z 平面拉伸叠加。顶部相机高度偏移和叠层透明度可在 YAML 中调整。图中被屋顶遮挡的地面仍可能绘制 NavMesh 诊断层，不代表地面可从相机直接看到。

**已知限制：**当前全密度高斯的高空俯视图存在严重拉丝，不作为合格的精细障碍底图；这不影响已验证的地面相机渲染。`gaussian_topdown_height_m` 表示相对于 NavMesh 最高点的高度偏移。正常批量导航会在共享 Simulator 中共用俯视与地面 sensor；独立刷新模式每条路线单独载入一次场景，批量刷新耗时较长。

mesh 底图第一行对应世界 Z 最大侧、第一列对应 X 最小侧，使用 `origin=upper` 与正常的米制 extent 配对；已修复此前只翻转底图造成的错位。另存 `mesh_topdown_base.png` 和 `mesh_topdown_coordinates.json`，记录完整栅格和坐标范围。`detail_regions_xz` 为局部放大范围列表，每项为 `[xmin,xmax,zmin,zmax]`，输出 `mesh_navmesh_detail_00.png` 等局部图。局部图复用相同世界坐标，不重新拉伸底图。

回归验证：`PYTHONPATH=deployment /data/anaconda3/envs/cloudrobo-habitat-gs/bin/python -m unittest discover -s deployment/tests -v`。四角颜色用例检查俯视方向及边界点；全场图已目视核对斜路、上/下转角和右侧通道。更多修改说明见 [CHANGELOG.md](CHANGELOG.md)。

此前 0.1m/0.05m 网格差异图：绿色表示共有区域，红色表示 0.05m 新增区域，蓝色表示仅 0.1m 存在区域；这不是可走/不可走的分类图。

低矮障碍需检查碰撞 mesh 是否包含柱体，以及柱体高度、宽度与 `max_climb_m`（当前 0.2m）、`cell_size_m`（0.1m）和 `cell_height_m`（0.05m）的关系。最大跨越高度以内的物体可能被当成台阶；区域面积过滤主要过滤可行走岛，不能直接用它解释柱体漏检。尚未定位的柱体不能仅凭底图颜色判断已被安全避让。
