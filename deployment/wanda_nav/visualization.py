"""生成高分辨率俯视诊断图，并尽量保留 mesh/Gaussian 底图的空间细节。"""
import json
from pathlib import Path

import numpy as np


def _pixel_colors(xz, rgb, bounds, pixel_size, max_points=None, seed=0, smoothing_px=1):
    """将不规则点云颜色聚合到米制像素网格，避免直接散点造成的碎屑观感。"""
    from scipy.ndimage import gaussian_filter

    (xmin, xmax), (zmin, zmax) = bounds
    width = max(1, int(np.ceil((xmax - xmin) / pixel_size)))
    height = max(1, int(np.ceil((zmax - zmin) / pixel_size)))
    if max_points is not None and len(xz) > max_points:
        chosen = np.random.default_rng(seed).choice(len(xz), max_points, replace=False)
        xz, rgb = xz[chosen], rgb[chosen]
    # extent 与像素分箱使用同一实际边长；最后一行/列也完整包含边界点。
    col = np.floor((xz[:, 0] - xmin) / (xmax - xmin) * width).astype(np.int64)
    row = np.floor((zmax - xz[:, 1]) / (zmax - zmin) * height).astype(np.int64)
    col = np.where(xz[:, 0] == xmax, width - 1, col)
    row = np.where(xz[:, 1] == zmin, height - 1, row)
    valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    row, col, rgb = row[valid], col[valid], rgb[valid]

    counts = np.zeros((height, width), dtype=np.float32)
    sums = np.zeros((height, width, 3), dtype=np.float32)
    np.add.at(counts, (row, col), 1)
    for channel in range(3):
        np.add.at(sums[..., channel], (row, col), rgb[:, channel])

    # 高斯平滑只用於填補采样形成的微小空洞，按颜色总和/点数分别滤波，
    # 避免直接平均 RGB 和空白背景而把底图洗成灰白色。
    smooth_radius = max(0, float(smoothing_px))
    if smooth_radius == 0:
        smooth_radius = 1e-6
    weighted_counts = gaussian_filter(counts, smooth_radius)
    weighted_sums = np.stack([gaussian_filter(sums[..., c], smooth_radius)
                              for c in range(3)], axis=-1)
    image = np.divide(weighted_sums, weighted_counts[..., None],
                      out=np.zeros_like(weighted_sums),
                      where=weighted_counts[..., None] > 1e-6)
    occupied = weighted_counts > 0.015
    image[~occupied] = 1.0
    return image, occupied, (xmin, xmax, zmin, zmax)


def _finish_axis(ax, bounds):
    xmin, xmax, zmin, zmax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal")
    ax.set_xlabel("Habitat X (m)")
    ax.set_ylabel("Habitat Z (m)")


def _save_figure(fig, output_path, config):
    visual = config["visualization"]
    fig.set_size_inches(visual.get("image_width_px", 3600) / visual["dpi"],
                        visual.get("image_height_px", 3000) / visual["dpi"])
    fig.savefig(output_path, dpi=visual["dpi"], facecolor="white", pil_kwargs={"compress_level": 1})
    import matplotlib.pyplot as plt
    plt.close(fig)


def save_gaussian_navmesh_overlay(rgb, island_triangles, bounds, output_path, config):
    """将 Habitat-GS 真实俯视相机 RGB 与全岛 NavMesh 按同一范围叠加。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    image = np.asarray(rgb)[..., :3]
    xmin, xmax, zmin, zmax = bounds
    fig, ax = plt.subplots()
    ax.imshow(image, extent=(xmin, xmax, zmax, zmin), origin="upper")
    for _, triangles in island_triangles:
        ax.add_collection(PolyCollection(triangles,
                                         facecolors=config["visualization"]["navmesh_color"],
                                         edgecolors="none",
                                         alpha=config["visualization"]["gaussian_overlay_alpha"],
                                         rasterized=True))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_xlabel("图像列 (px)")
    ax.set_ylabel("图像行 (px)")
    ax.set_title("Habitat-GS 高斯俯视透视渲染 + NavMesh（逐顶点投影）")
    ax.legend(handles=[Line2D([0], [0], color=config["visualization"]["navmesh_color"], lw=8,
                              label="可行走 NavMesh")], loc="upper right")
    fig.tight_layout()
    _save_figure(fig, output_path, config)


def project_navmesh_to_topdown_pixels(triangles, camera_y, focal, center_x, center_z,
                                      width, height):
    """将世界 XYZ NavMesh 三角面投影到向下看的 Habitat 相机像素 XY。"""
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    depth = camera_y - triangles[..., 1]
    if np.any(depth <= 0):
        raise ValueError("俯视相机必须高于所有 NavMesh 顶点")
    pixel_x = focal * (triangles[..., 0] - center_x) / depth + (width - 1) / 2
    pixel_y = (height - 1) / 2 - focal * (triangles[..., 2] - center_z) / depth
    return np.stack([pixel_x, pixel_y], axis=-1)


def topdown_plots(pathfinder, points, frames, scene_dir, output_dir, config, island):
    """輸出軌跡、彩色 mesh/NavMesh 及 Gaussian/NavMesh 高分辨率頂視圖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from plyfile import PlyData

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    output_dir = Path(output_dir)
    scene_dir = Path(scene_dir)
    visual, output = config["visualization"], config["output"]

    island_triangles = []
    for island_id in range(pathfinder.num_islands):
        vertices = np.asarray(pathfinder.build_navmesh_vertices(island_id)).reshape(-1, 3, 3)
        if len(vertices):
            island_triangles.append((island_id, vertices))
    trajectory = np.asarray([frame[0] for frame in frames])
    all_triangles = np.concatenate([tri for _, tri in island_triangles], axis=0)
    nav_bounds = (float(all_triangles[:, :, 0].min()), float(all_triangles[:, :, 0].max()),
                  float(all_triangles[:, :, 2].min()), float(all_triangles[:, :, 2].max()))

    if output["trajectory_topdown"]:
        fig, ax = plt.subplots()
        for island_id, island_tri in island_triangles:
            color = "#75c99a" if island_id == island else "#aab8b0"
            alpha = 0.62 if island_id == island else 0.30
            ax.add_collection(PolyCollection(island_tri[:, :, [0, 2]], facecolors=color,
                                             edgecolors="none", alpha=alpha, rasterized=True))
        ax.plot(trajectory[:, 0], trajectory[:, 2], color=config["visualization"]["trajectory_color"],
                linewidth=visual["trajectory_linewidth"], label="運行軌跡", zorder=4)
        ax.scatter(*trajectory[0, [0, 2]], color="#1674d1", s=70, label="起點", zorder=5)
        ax.scatter(*trajectory[-1, [0, 2]], color="#e33c32", s=70, label="終點", zorder=5)
        _finish_axis(ax, nav_bounds)
        ax.legend(loc="upper right", framealpha=0.95)
        ax.set_title("NavMesh 全連通區與導航軌跡")
        fig.tight_layout()
        _save_figure(fig, output_dir / "trajectory_topdown.png", config)

    stage = json.loads((scene_dir / "wanda.stage_config.json").read_text(encoding="utf-8"))
    mesh_path = Path(stage["collision_asset"])
    if not mesh_path.is_absolute():
        mesh_path = scene_dir / mesh_path
    max_points = output.get("topdown_max_points")
    pixel_size = visual.get("topdown_pixel_size_m", 0.06)
    bounds = (nav_bounds[0], nav_bounds[1], nav_bounds[2], nav_bounds[3])

    if output["mesh_navmesh_overlay"]:
        mesh = PlyData.read(str(mesh_path))["vertex"].data
        coords = np.column_stack([mesh["x"], mesh["z"]]).astype(np.float32)
        colors = np.column_stack([mesh["red"], mesh["green"], mesh["blue"]]).astype(np.float32) / 255.0
        mesh_bounds = ((float(mesh["x"].min()), float(mesh["x"].max())),
                       (float(mesh["z"].min()), float(mesh["z"].max())))
        base, occupied, extent = _pixel_colors(coords, colors, mesh_bounds, pixel_size,
                                                max_points, config["navigation"]["seed"],
                                                visual.get("topdown_smoothing_px", 1))
        # 原始米制栅格独立保存：放大查看不受绘图页边距/图例挤压影响。
        from PIL import Image
        Image.fromarray(np.uint8(np.clip(base, 0, 1) * 255)).save(output_dir / "mesh_topdown_base.png")
        (output_dir / "mesh_topdown_coordinates.json").write_text(json.dumps({
            "coordinate_frame": "Habitat X/Z, meters", "extent_xmin_xmax_zmin_zmax": list(extent),
            "origin": "upper", "shape_hw": list(base.shape[:2]),
            "pixel_size_x_m": (extent[1]-extent[0])/base.shape[1],
            "pixel_size_z_m": (extent[3]-extent[2])/base.shape[0],
            "note": "第一行对应 Z 最大侧，第一列对应 X 最小侧；颜色聚合不改变碰撞 mesh。"
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        fig, ax = plt.subplots()
        ax.imshow(base, extent=extent, origin="upper", interpolation="nearest")
        for _, tri in island_triangles:
            ax.add_collection(PolyCollection(tri[:, :, [0, 2]], facecolors=visual["navmesh_color"],
                                             edgecolors="none", alpha=visual.get("mesh_overlay_alpha", 0.28),
                                             rasterized=True))
        _finish_axis(ax, (extent[0], extent[1], extent[2], extent[3]))
        ax.set_title("彩色 Mesh 頂視顏色聚合 + NavMesh")
        ax.legend(handles=[Line2D([0], [0], color=visual["navmesh_color"], lw=8,
                                  label="可行走 NavMesh")], loc="upper right")
        fig.tight_layout()
        _save_figure(fig, output_dir / "mesh_navmesh_topdown.png", config)
        # 局部图复用相同世界坐标映射，不裁切拉伸底图，方便检查窄柱和网格孔洞。
        for index, region in enumerate(visual.get("detail_regions_xz", [])):
            x0, x1, z0, z1 = region
            fig, ax = plt.subplots()
            ax.imshow(base, extent=extent, origin="upper", interpolation="nearest")
            for _, tri in island_triangles:
                mask = ((tri[..., 0].max(axis=1) >= x0) & (tri[..., 0].min(axis=1) <= x1)
                        & (tri[..., 2].max(axis=1) >= z0) & (tri[..., 2].min(axis=1) <= z1))
                ax.add_collection(PolyCollection(tri[mask][:, :, [0, 2]],
                    facecolors=visual["navmesh_color"], edgecolors="none",
                    alpha=visual.get("mesh_overlay_alpha", 0.28)))
            _finish_axis(ax, region)
            ax.set_title(f"局部 {index+1}：彩色 Mesh + NavMesh（米制坐标）")
            fig.tight_layout()
            _save_figure(fig, output_dir / f"mesh_navmesh_detail_{index:02d}.png", config)

