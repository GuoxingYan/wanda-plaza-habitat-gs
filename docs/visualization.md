# 可视化与结果判读

输出图和视频由 `output` 与 `visualization` 两个 YAML 区块控制。默认不启用昂贵的 Gaussian 高空俯视渲染；导航 RGB 仍使用配置的 D435 相机参数。

## 常见图片

- `trajectory_topdown.png`：NavMesh 连通区域、执行轨迹、抽样车头朝向、起点和终点。明显拐弯处会额外补一个同样式的稍长箭头；箭头从逐帧姿态四元数计算，倒车时仍表示车头方向。若开启避障实验，也显示障碍物 footprint。
- `mesh_navmesh_topdown.png`：mesh 顶点颜色按米制网格聚合形成底图，叠加 NavMesh 可行走面。
- `camera_trajectory_density_comparison.png`：采集轨迹密度过滤关闭/开启后的可行区域，并单独展示 cameras.json 轨迹。浅红色表示被密度条件过滤的候选岛。
- `gaussian_topdown_raw.png` 与 `gaussian_navmesh_topdown.png`：可选的 Habitat-GS 俯视渲染及投影叠图。大场景开销较高，且高空 Gaussian 可能出现拉丝，不应单独作为精确障碍判据。
- `rgb_*.png` 和 `navigation_rgb.mp4`：D435 参数的地面视角 Gaussian 渲染。
- `summary.html` 与 `all_routes_mesh_overlay.png`：批次汇总报告及彩色 mesh 纹理底图上的路线叠加图。每条路线使用独立颜色，箭头为按路程抽样的车头朝向，显著拐弯处额外加箭；蓝色圆点为起点，橙色三角为终点，不叠加 NavMesh。

## 坐标约定

所有俯视图用 Habitat 世界 X/Z 米制坐标。mesh 底图第一行对应世界 Z 最大侧，第一列对应 X 最小侧；像素映射见同目录 `mesh_topdown_coordinates.json`。顶视 mesh 颜色是栅格聚合，不做深度遮挡，建筑和树冠可能投影到地面底图中。NavMesh 覆盖层代表几何可行区域，不保证该处在 RGB 相机中可见。

`visualization.image_width_px`、`image_height_px` 和 `dpi` 控制导出尺寸；`topdown_pixel_size_m` 控制底图的米制像素大小；`topdown_smoothing_px` 只用于平滑可视化采样空洞，不改变碰撞 mesh 或导航结果。`detail_regions_xz` 可列出局部范围 `[xmin,xmax,zmin,zmax]`，生成放大图。

## 网格差异图

此前 NavMesh cell size 差异图中，绿色代表两种网格结果共有区域，红色代表细网格新增区域，蓝色代表粗网格独有区域。它不是可行走/不可行走分类图。
