# Wanda Plaza / Habitat-GS

独立 Conda 环境：`cloudrobo-habitat-gs`，Python 3.12，CUDA toolkit 12.4，RTX 3090。

## 运行

```bash
cd /data/projects/CloudRobo
conda activate cloudrobo-habitat-gs
./deployment/run_validation.sh
# 已有 NavMesh 时跳过烘焙
./deployment/run_validation.sh --skip-bake
# 只生成 NavMesh 并验证路径移动
./deployment/run_validation.sh --bake-only
```

输入文件位于 `Wanda Plaza/`（链接到 `/data/data/Wanda Plaza`）。原文件不修改。

`prepare_scene.py` 默认用同一个源坐标包围盒 `X=[-50,10], Y=[-40,20]` 裁剪 mesh 和高斯，再将源 Z-up 转为 Habitat Y-up：`(x,y,z) -> (x,z,-y)`。mesh 三角面的三个顶点必须全部位于包围盒内；高斯中心也使用同一包围盒。位置、法线、高斯四元数、SH 1–3 阶颜色系数一起旋转。该 PLY 使用顶点色，没有 UV/外部贴图定义。

```bash
python deployment/prepare_scene.py
# 自定义同规格 crop：XMIN XMAX YMIN YMAX
python deployment/prepare_scene.py --crop -50 10 -40 20 --out deployment/scene_crop
```

默认验证资产包含 1,284,922 个 mesh 三角面和 870,725 个高斯，均来自上述同一 60×60 米 crop，SH 不降阶。

## 验证输出

`scene_crop/wanda_crop.navmesh` 是实际烘焙结果。`results_crop/validation.json` 为实测报告；`navmesh_route.png` 为实际 NavMesh 与路径，`path.npy` 为 geodesic 转折点，`trajectory.npy` 为经过 `PathFinder.try_step` 过滤的移动轨迹；`navigation_rgb.mp4` 和 `rgb_*.png` 来自 CUDA Gaussian Splatting。

相机使用 D435 **RGB** 标称 HFOV=69°、640×360、安装高度1.5米，理想针孔，无畸变。不是深度相机的87°视场角，也不是实机逐台标定内参。默认坐标/mesh单位假定米。代理半径0.25米，高1.5米，爬升0.2米、坡度35°；NavMesh体素0.1×0.05米。路径导航是确定性geodesic跟随与NavMesh碰撞约束，不包含视觉策略、SLAM或运动控制器。

## 构建

`build.sh` 可重新构建当前源码。仅安装静态场景、导航与RGB所需依赖；关闭Bullet、GUI、Audio、Assimp，保留原生PLY/GLTF导入与CUDA渲染。GLM头文件解包在`system-deps/root`，不需要系统级安装。上游`src/cmake/dependencies.cmake`有一处补丁：仅当启用Assimp时将AssimpImporter加入Magnum静态插件列表。

下载、构建、转换及验证日志位于`logs/`。

参考：[Habitat-GS](https://github.com/zju3dv/habitat-gs)，[D435规格](https://www.realsenseai.com/products/stereo-depth-camera-d435/)。
