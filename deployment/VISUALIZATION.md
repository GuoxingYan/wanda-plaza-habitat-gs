# 结果图像与先验说明

本说明对应默认全场输出目录 `deployment/results_full/route_0000/`。当前环境中的全密度示例位于本机忽略目录 `deployment/results_fullres/route_0000/`；PNG、PLY 和视频均不纳入 Git。

## 图像索引

| 文件 | 内容 | 判读重点 |
|---|---|---|
| `trajectory_topdown.png` | 全部 NavMesh 连通区域、当前路线、起点和终点 | 查看路线是否落在目标可行走区域；完整岛均绘制，不等于不同岛互通 |
| `mesh_navmesh_topdown.png` | 彩色 mesh 顶视颜色聚合，叠加绿色 NavMesh | 对齐道路和场景范围；mesh 是所有高度顶点投影，树冠可能覆盖其树干周围的地面 |
| `mesh_topdown_base.png` | 未叠 NavMesh 的聚合底图 | 检查纹理/颜色和模型覆盖范围 |
| `mesh_navmesh_detail_00.png` 等 | 按配置区域输出的局部放大图 | 用 `detail_regions_xz` 按 `[xmin,xmax,zmin,zmax]` 设定米制范围 |
| `gaussian_topdown_raw.png` | Habitat-GS 实际高空俯视渲染 | 当前高空视角有拉丝，暂不用于精细障碍判定 |
| `gaussian_navmesh_topdown.png` | 高空 Gaussian RGB 与逐顶点投影 NavMesh | 投影按透视深度和 Y-up 坐标计算；建筑遮挡处仍可能画有 NavMesh |

mesh 顶视栅格的坐标定义保存在 `mesh_topdown_coordinates.json`：第一列对应世界 X 最小侧，第一行对应世界 Z 最大侧。图像 `origin=upper`；世界坐标仍按 X/Z 正向标尺显示，不应再额外翻转图层。

## 采集轨迹先验

`Wanda Plaza/cameras.json` 的 30,276 条记录含原始 Z-up 相机中心。系统只使用 `position`，不使用图像朝向；先按 `(x,y,z) -> (x,z,-y)` 转为 Habitat Y-up，再在 X/Z 平面建立距离查询。

默认 YAML 设置：

- `use_camera_trajectory_prior: true`：启用相机采集路线附近区域约束。
- `camera_trajectory_buffer_m: 10.0`：路径点到最近相机中心的最大平面距离。
- `prefer_camera_trajectory_endpoints: true`：起终点优先由采集相机 XYZ 位置吸附到 NavMesh 得到，姿态不复用。

路线每 0.5 m 检查一次；实际候选仍需要 NavMesh 连通、测地距离符合要求且路径全段位于缓冲区。这个先验表达“曾经采集经过附近”的弱可行性，不代表完整机器人宽度、动态障碍或法律意义的道路通行保证。`validation.json` 记录输入文件、去重位置数量、缓冲距离和路线检查结果。

## 当前已知限制

1. 顶点颜色聚合底图没有深度缓冲，树冠与地面可能重叠显示；需要结合树干高度层、NavMesh 高程、最近障碍距离判断误提取。
2. 高空 Gaussian 视角存在显著拉丝；地面 RGB 视角已验证正常，全场俯视仍有画质限制。
3. 右上树列临时探针找到少数相邻于绿色高处几何的可行走点，最近障碍距离低于 0.25 m 代理半径；降低 climb 并细化至 5 cm 后仍有点被判可走。这提示要继续核对树干，但尚不能证明高处几何是树干或整片树木已得到正确碰撞处理。正式 NavMesh 未修改。后续可评估在碰撞几何中加入树干或为绿色植被建立专门障碍层；两者都会改变导航行为，需先由用户选择。

完整运行命令、YAML 参数和验证入口见 [README.md](README.md)，功能历史见 [CHANGELOG.md](CHANGELOG.md)。
