# Wanda Plaza Habitat-GS 导航验证

这个仓库保存 Wanda Plaza 场景接入 Habitat-GS 的可复现脚本和配置。原始场景数据不在仓库中，需要在本地准备：

```text
/data/projects/CloudRobo/Wanda Plaza/point_cloud.ply
/data/projects/CloudRobo/Wanda Plaza/mesh.ply
```

其中 Gaussian PLY 与 mesh 使用同一个源坐标系。脚本默认裁剪源坐标 `X=[-50,10]、Y=[-40,20]` 的 60×60 米区域；mesh 只保留三个顶点都在范围内的三角面，高斯中心使用完全相同的范围。随后将源 Z-up 坐标转换到 Habitat Y-up：`(x,y,z) -> (x,z,-y)`，同时旋转法线、四元数和球谐颜色系数。

## 安装环境

机器上已经验证的环境名称为 `cloudrobo-habitat-gs`，Python 3.12，Habitat-GS 0.3.3，CUDA 12.4，GPU 为 RTX 3090。源码构建使用 `deployment/build.sh`；首次编译会花较长时间。

## 运行验证

```bash
cd /data/projects/CloudRobo
conda activate cloudrobo-habitat-gs
python deployment/prepare_scene.py
./deployment/run_validation.sh
```

验证会完成：mesh NavMesh 烘焙、最大连通区域内的 geodesic 路径规划、`PathFinder.try_step` 移动检查，以及 D435 RGB 参数下的 CUDA Gaussian Splatting 渲染。验证报告写入 `deployment/results_crop/validation.json`；生成的二进制场景和媒体文件被 `.gitignore` 排除。

当前 crop 的实测结果保留在 `deployment/results_crop/validation.json`：NavMesh 可导航面积 771.76 m²，路径长度 30.46 m，119 个移动步，终点误差 0，RGB 120 帧、640×360、约 6.4 ms/帧。

## 文件说明

- `deployment/prepare_scene.py`：读取二进制 PLY，按同一包围盒裁剪 mesh 与高斯并完成坐标/SH 变换。
- `deployment/validate_navigation.py`：烘焙 NavMesh、规划路线、检查移动并渲染 RGB。
- `deployment/run_validation.sh`：设置 CUDA 与动态库路径后运行验证。
- `deployment/build.sh`：在独立 Conda 环境中构建 CUDA Habitat-GS。
- `deployment/habitat-gs-minimal-build.patch`：关闭 Assimp 时修正 Magnum 静态插件列表的补丁。

该 mesh PLY 实际包含顶点颜色，没有 UV 或外部纹理文件；因此渲染验证使用 Gaussian 的 SH 颜色。D435 使用 RGB 彩色相机标称水平视场角 69°，不是深度相机视场角，也不是某台实机逐台标定的内参。
