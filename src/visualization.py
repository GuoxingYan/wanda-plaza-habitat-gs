"""生成高分辨率俯视诊断图，并尽量保留 mesh/Gaussian 底图的空间细节。"""
import json
import hashlib
from pathlib import Path

import numpy as np

_GAUSSIAN_TOPDOWN_CACHE = {}


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


def _mesh_image_extent(extent):
    """返回 origin=upper 的 mesh 栅格对应的标准 X/Z 世界范围。

    栅格第 0 行已经是最大 Z；Matplotlib 的 extent 参数仍按
    (left, right, bottom, top) 传入，即 (xmin, xmax, zmin, zmax)。
    """
    xmin, xmax, zmin, zmax = extent
    return xmin, xmax, zmin, zmax


def _save_figure(fig, output_path, config):
    visual = config["visualization"]
    fig.set_size_inches(visual.get("image_width_px", 3600) / visual["dpi"],
                        visual.get("image_height_px", 3000) / visual["dpi"])
    fig.savefig(output_path, dpi=visual["dpi"], facecolor="white", pil_kwargs={"compress_level": 1})
    import matplotlib.pyplot as plt
    plt.close(fig)


def _file_signature(path):
    """用路径、大小和修改时间识别静态输入是否变化。"""
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _static_visualization_key(navmesh_path, mesh_path, config, include_navmesh=True):
    """分别为 mesh 栅格或 NavMesh 缓存生成稳定指纹。"""
    visual = config["visualization"]
    output = config["output"]
    payload = {
        "mesh": _file_signature(mesh_path),
        "pixel_size": visual.get("topdown_pixel_size_m", 0.06),
        "smoothing": visual.get("topdown_smoothing_px", 1),
        "max_points": output.get("topdown_max_points"),
    }
    if include_navmesh:
        payload["navmesh"] = _file_signature(navmesh_path)
        payload.update({"width": visual.get("image_width_px", 3600),
                        "height": visual.get("image_height_px", 3000),
                        "dpi": visual["dpi"], "navmesh_color": visual["navmesh_color"],
                        "alpha": visual.get("mesh_overlay_alpha", 0.28),
                        "detail_regions": visual.get("detail_regions_xz", [])})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"key": digest, "inputs": payload}


def _load_cached_navmesh(pathfinder, navmesh_path):
    """以场景 NavMesh 文件指纹缓存连通岛三角面和包围范围。"""
    signature = _file_signature(navmesh_path)
    cache_path = Path(navmesh_path).with_name(".wanda_navmesh_visual_cache.npz")
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["signature"].item()) == json.dumps(signature, sort_keys=True):
                islands = []
                count = int(cache["island_count"].item())
                for index in range(count):
                    islands.append((int(cache[f"island_id_{index}"].item()),
                                    cache[f"triangles_{index}"].copy()))
                return islands, tuple(cache["bounds"].tolist())
    except (OSError, ValueError, KeyError):
        pass

    islands = []
    for island_id in range(pathfinder.num_islands):
        vertices = np.asarray(pathfinder.build_navmesh_vertices(island_id)).reshape(-1, 3, 3)
        if len(vertices):
            islands.append((island_id, vertices))
    all_triangles = np.concatenate([triangles for _, triangles in islands], axis=0)
    bounds = (float(all_triangles[:, :, 0].min()), float(all_triangles[:, :, 0].max()),
              float(all_triangles[:, :, 2].min()), float(all_triangles[:, :, 2].max()))
    arrays = {"signature": np.asarray(json.dumps(signature, sort_keys=True)),
              "island_count": np.asarray(len(islands)), "bounds": np.asarray(bounds)}
    for index, (island_id, triangles) in enumerate(islands):
        arrays[f"island_id_{index}"] = np.asarray(island_id)
        arrays[f"triangles_{index}"] = triangles
    try:
        np.savez_compressed(cache_path, **arrays)
    except OSError:
        # 只读场景目录仍可照常运行，缓存只是性能优化。
        pass
    return islands, bounds


def _load_static_base(mesh_path, config, cache_dir):
    """读取或重建 mesh 颜色栅格缓存，供批次内每条路线共享。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _static_visualization_key(cache_dir / "wanda.navmesh", mesh_path, config,
                                    include_navmesh=False)
    cache_path = cache_dir / f".mesh_topdown_{key['key'][:20]}.npz"
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            base = cache["base"].copy()
            extent = tuple(cache["extent"].tolist())
            coordinates = json.loads(str(cache["coordinates"].item()))
            return base, extent, coordinates
    except (OSError, ValueError, KeyError):
        pass

    from plyfile import PlyData
    mesh = PlyData.read(str(mesh_path))["vertex"].data
    coords = np.column_stack([mesh["x"], mesh["z"]]).astype(np.float32)
    colors = np.column_stack([mesh["red"], mesh["green"], mesh["blue"]]).astype(np.float32) / 255.0
    mesh_bounds = ((float(mesh["x"].min()), float(mesh["x"].max())),
                   (float(mesh["z"].min()), float(mesh["z"].max())))
    visual, output = config["visualization"], config["output"]
    base, _, extent = _pixel_colors(coords, colors, mesh_bounds,
                                    visual.get("topdown_pixel_size_m", 0.06),
                                    output.get("topdown_max_points"),
                                    0,
                                    visual.get("topdown_smoothing_px", 1))
    coordinates = {"coordinate_frame": "Habitat X/Z, meters",
                   "extent_xmin_xmax_zmin_zmax": list(extent), "origin": "upper",
                   "shape_hw": list(base.shape[:2]),
                   "pixel_size_x_m": (extent[1] - extent[0]) / base.shape[1],
                   "pixel_size_z_m": (extent[3] - extent[2]) / base.shape[0],
                   "note": "第一行对应 Z 最大侧，第一列对应 X 最小侧；颜色聚合不改变碰撞 mesh。"}
    try:
        np.savez_compressed(cache_path, base=base, extent=np.asarray(extent),
                            coordinates=np.asarray(json.dumps(coordinates, ensure_ascii=False)))
    except OSError:
        pass
    return base, extent, coordinates


def save_gaussian_navmesh_overlay(rgb, island_triangles, bounds, output_path, config):
    """将 Gaussian 俯视 RGB 与同坐标系 NavMesh 像素三角面叠加。"""
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
    ax.set_title("Habitat-GS 正交分块地面切片 + NavMesh")
    ax.legend(handles=[Line2D([0], [0], color=config["visualization"]["navmesh_color"], lw=8,
                              label="可行走 NavMesh")], loc="upper right")
    fig.tight_layout()
    _save_figure(fig, output_path, config)


def save_camera_density_comparison(island_triangles, density_stats, trajectory, camera_trajectory,
                                   mesh_path, scene_dir, output_path, config):
    """输出采集轨迹密度后处理开关的并排对比图。

    前两列代表关闭/开启密度过滤，第三列单独显示 cameras.json 采集轨迹。
    三图严格复用同一米制坐标范围，便于对照网格、导航路线和原始采集覆盖。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib import patheffects

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    keep = np.asarray(density_stats["keep"], dtype=bool)
    visual = config["visualization"]
    base, mesh_extent, _ = _load_static_base(mesh_path, config, scene_dir)
    image = np.uint8(np.clip(base, 0, 1) * 255)
    trajectory = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    camera_segments = [np.asarray(segment, dtype=np.float64).reshape(-1, 3)
                       for segment in camera_trajectory if len(segment)]
    bounds = (mesh_extent[0], mesh_extent[1], mesh_extent[2], mesh_extent[3])
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    trajectory_effects = [patheffects.Stroke(linewidth=4.5, foreground="white", alpha=0.95),
                          patheffects.Normal()]
    for panel_index, (axis, title) in enumerate(zip(
            axes, ("关闭：原始 NavMesh", "开启：稀疏区域过滤", "cameras.json 采集轨迹"))):
        filtered = panel_index == 1
        capture_only = panel_index == 2
        # _pixel_colors 已将世界最大 Z 放在图像第 0 行；配合 origin=upper 时
        # 使用正常的米制 extent（bottom=zmin, top=zmax）才能让图像行正确回到世界 Z。
        axis.imshow(image, extent=_mesh_image_extent(mesh_extent),
                    origin="upper", interpolation="nearest", zorder=0)
        for island_id, triangles in island_triangles:
            is_sparse = island_id >= len(keep) or not keep[island_id]
            if filtered and is_sparse:
                continue
            if capture_only:
                color = visual["navmesh_color"] if not is_sparse else "#f4a6a6"
                alpha = 0.34 if is_sparse else 0.28
            elif filtered:
                color = visual["navmesh_color"]
                alpha = 0.58
            else:
                color = "#f4a6a6" if is_sparse else visual["navmesh_color"]
                alpha = 0.48 if is_sparse else 0.58
            axis.add_collection(PolyCollection(
                triangles[:, :, [0, 2]], facecolors=color, edgecolors="none",
                alpha=alpha, rasterized=True, zorder=2))
        if not capture_only and len(trajectory):
            route_line, = axis.plot(trajectory[:, 0], trajectory[:, 2], color="#d62728",
                                    linewidth=visual.get("trajectory_linewidth", 1.7) + 1.0,
                                    label="导航轨迹", zorder=5)
            route_line.set_path_effects(trajectory_effects)
            axis.scatter(trajectory[0, 0], trajectory[0, 2], s=90, marker="o",
                         facecolor="#1674d1", edgecolor="white", linewidth=1.5,
                         label="起点", zorder=6)
            axis.scatter(trajectory[-1, 0], trajectory[-1, 2], s=120, marker="*",
                         facecolor="#ff8c00", edgecolor="white", linewidth=1.2,
                         label="终点", zorder=6)
        if capture_only and camera_segments:
            all_positions = []
            for segment in camera_segments:
                axis.plot(segment[:, 0], segment[:, 2], color="#174ea6",
                          linewidth=1.0, alpha=0.8, zorder=5)
                all_positions.append(segment)
            all_positions = np.concatenate(all_positions, axis=0)
            sample_step = max(1, len(all_positions) // 500)
            sampled = all_positions[::sample_step]
            axis.scatter(sampled[:, 0], sampled[:, 2], s=3, color="#174ea6",
                         alpha=0.55, label="采集相机位置", zorder=5)
        if filtered:
            axis.set_title(title + f"（保留 {int(np.count_nonzero(keep))}/{len(keep)} 岛）")
        else:
            axis.set_title(title)
        _finish_axis(axis, bounds)
    axes[0].legend(handles=[
        Line2D([0], [0], color=visual["navmesh_color"], lw=8, label="轨迹充分，保留"),
        Line2D([0], [0], color="#f4a6a6", lw=8, label="轨迹稀疏，开启时过滤"),
        Line2D([0], [0], color="#d62728", lw=2.5, label="导航轨迹"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1674d1",
               markeredgecolor="white", markersize=9, label="起点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#ff8c00",
               markeredgecolor="white", markersize=12, label="终点"),
        Line2D([0], [0], color="#174ea6", lw=2, label="cameras.json 采集轨迹"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#174ea6",
               markersize=4, label="采集相机位置"),
    ], loc="upper right", framealpha=0.95)
    fig.suptitle(
        "cameras.json 采集轨迹密度后处理对比（点数/100 m²："
        f"≥{density_stats['min_points_per_100m2']:.3g}，且点数≥{density_stats['min_points']}）")
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


def render_ground_sliced_gaussian_topdown(sim, triangles_by_island, config, world_bounds):
    """分块正交俯视渲染，并按每块 NavMesh 高程裁掉远离地面的 Gaussian。

    每块相机只看局部高度薄层，避免从单一高空相机观察整片场景时把树冠、
    立面和重建离群 Gaussian 叠投到地面。这里仅筛渲染深度，不改写输入 PLY。
    """
    from PIL import Image
    import magnum as mn
    import habitat_sim as hs
    from .planning import yaw_quaternion

    visual = config["visualization"]
    scene_dir = Path(config["scene"]["directory"])
    stage_path = scene_dir / "wanda.stage_config.json"
    navmesh_path = scene_dir / "wanda.navmesh"
    key_data = {
        "stage": _file_signature(stage_path),
        "navmesh": _file_signature(navmesh_path),
        "width": visual["image_width_px"], "height": visual["image_height_px"],
        "tile_size": visual.get("gaussian_topdown_tile_size_m", 64.0),
        "tile_resolution": visual.get("gaussian_topdown_tile_resolution_px", 768),
        "camera_height": visual.get("gaussian_topdown_camera_height_m", 8.0),
        "above": visual.get("gaussian_topdown_height_above_walkable_m", 3.0),
        "below": visual.get("gaussian_topdown_depth_below_walkable_m", 0.5),
    }
    cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    if cache_key in _GAUSSIAN_TOPDOWN_CACHE:
        image, extent = _GAUSSIAN_TOPDOWN_CACHE[cache_key]
        return image.copy(), extent
    xmin, zmin = float(world_bounds[0, 0]), float(world_bounds[0, 2])
    xmax, zmax = float(world_bounds[1, 0]), float(world_bounds[1, 2])
    width = int(visual["image_width_px"])
    height = int(visual["image_height_px"])
    span = max(xmax - xmin, (zmax - zmin) * width / height) * 1.10
    center_x, center_z = (xmin + xmax) / 2, (zmin + zmax) / 2
    meters_per_pixel = span / width
    tile_size = float(visual.get("gaussian_topdown_tile_size_m", 64.0))
    tile_pixels = int(visual.get("gaussian_topdown_tile_resolution_px", 768))
    camera_height = float(visual.get("gaussian_topdown_camera_height_m", 8.0))
    above = float(visual.get("gaussian_topdown_height_above_walkable_m", 3.0))
    below = float(visual.get("gaussian_topdown_depth_below_walkable_m", 0.5))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    # 先建立每块覆盖的可行走三角面，再由局部中位高程放置相机切片。
    # Sensor 为正方形；其正交视野覆盖 tile_size × tile_size 米。
    x_edges = np.arange(xmin, xmax, tile_size).tolist() + [xmax]
    z_edges = np.arange(zmin, zmax, tile_size).tolist() + [zmax]
    for tile_x0, tile_x1 in zip(x_edges[:-1], x_edges[1:]):
        for tile_z0, tile_z1 in zip(z_edges[:-1], z_edges[1:]):
            local_heights = []
            for _, triangles in triangles_by_island:
                tri_min = triangles[:, :, [0, 2]].min(axis=1)
                tri_max = triangles[:, :, [0, 2]].max(axis=1)
                intersects = ((tri_max[:, 0] >= tile_x0) & (tri_min[:, 0] <= tile_x1) &
                              (tri_max[:, 1] >= tile_z0) & (tri_min[:, 1] <= tile_z1))
                if np.any(intersects):
                    local_heights.append(triangles[intersects, :, 1].reshape(-1))
            if not local_heights:
                continue

            ground_y = float(np.median(np.concatenate(local_heights)))
            state = hs.AgentState()
            state.position = mn.Vector3(tile_x0 + tile_size / 2,
                                        ground_y + camera_height,
                                        tile_z0 + tile_size / 2)
            state.rotation = yaw_quaternion(0.0)
            sim.initialize_agent(0, state)
            tile = sim.get_sensor_observations()["topdown_rgb"][:, :, :3]

            # Habitat 俯视画面上方对应 +Z；按统一世界范围写入目标像素区域。
            tile_mpp = tile_size / tile_pixels
            source_width = int(round((tile_x1 - tile_x0) / tile_mpp))
            source_height = int(round((tile_z1 - tile_z0) / tile_mpp))
            px0 = int(np.floor((tile_x0 - (center_x - span / 2)) / meters_per_pixel))
            px1 = int(np.ceil((tile_x1 - (center_x - span / 2)) / meters_per_pixel))
            image_z_span = span * height / width
            pixel_size_z = image_z_span / height
            image_zmax = center_z + span * height / width
            py0 = int(np.floor((image_zmax - (tile_z0 + tile_size)) / pixel_size_z))
            py1 = int(np.ceil((image_zmax - tile_z0) / pixel_size_z))
            dst_x0, dst_y0 = max(0, px0), max(0, py0)
            dst_x1, dst_y1 = min(width, px1), min(height, py1)
            if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
                continue
            crop_x = max(1, min(tile_pixels, source_width))
            crop_y = max(1, min(tile_pixels, source_height))
            # 末端 X/Z 块分别保留画面左侧和顶部，对应世界较小 X 和较大 Z。
            source_y0 = tile_pixels - crop_y if tile_z1 < tile_z0 + tile_size else 0
            source_tile = tile[source_y0:source_y0 + crop_y, :crop_x]
            resized = np.asarray(Image.fromarray(source_tile).resize(
                (max(1, px1 - px0), max(1, py1 - py0)), Image.Resampling.BILINEAR))
            canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[
                dst_y0 - py0:dst_y1 - py0, dst_x0 - px0:dst_x1 - px0]

    extent = (-span / 2 + center_x, span / 2 + center_x,
              center_z - span * height / width / 2,
              center_z + span * height / width / 2)
    _GAUSSIAN_TOPDOWN_CACHE[cache_key] = (canvas, extent)
    return canvas, extent


def project_navmesh_world_to_topdown_pixels(triangles, extent, width, height):
    """把世界 X/Z 三角面投影到等比例俯视拼接画布像素坐标。"""
    xmin, xmax, zmin, zmax = extent
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    pixel_x = (triangles[..., 0] - xmin) / (xmax - xmin) * width - 0.5
    pixel_y = (zmax - triangles[..., 2]) / (zmax - zmin) * height - 0.5
    return np.stack([pixel_x, pixel_y], axis=-1)


def topdown_plots(pathfinder, points, frames, scene_dir, output_dir, config, island, obstacle=None):
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

    island_triangles, nav_bounds = _load_cached_navmesh(pathfinder, scene_dir / "wanda.navmesh")
    trajectory = np.asarray([frame[0] for frame in frames])

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
        if obstacle and obstacle.get("triggered"):
            from matplotlib.patches import Circle
            center = obstacle["center"]
            ax.add_patch(Circle((center[0], center[2]), obstacle["radius_m"],
                                facecolor="#e53935", edgecolor="#8b1111", alpha=0.8,
                                label="臨時障礙物", zorder=6))
        _finish_axis(ax, nav_bounds)
        ax.legend(loc="upper right", framealpha=0.95)
        ax.set_title("NavMesh 全連通區與導航軌跡")
        fig.tight_layout()
        _add_sampled_heading_arrows(ax, frames, trajectory,
                                    visual["trajectory_color"])
        _save_figure(fig, output_dir / "trajectory_topdown.png", config)

    stage = json.loads((scene_dir / "wanda.stage_config.json").read_text(encoding="utf-8"))
    mesh_path = Path(stage["collision_asset"])
    if not mesh_path.is_absolute():
        mesh_path = scene_dir / mesh_path
    bounds = (nav_bounds[0], nav_bounds[1], nav_bounds[2], nav_bounds[3])

    if output["mesh_navmesh_overlay"]:
        base, extent, coordinates = _load_static_base(mesh_path, config, scene_dir)
        # 原始米制栅格独立保存：放大查看不受绘图页边距/图例挤压影响。
        from PIL import Image
        if output.get("mesh_topdown_base", True):
            Image.fromarray(np.uint8(np.clip(base, 0, 1) * 255)).save(output_dir / "mesh_topdown_base.png")
        if output.get("mesh_coordinates_json", True):
            (output_dir / "mesh_topdown_coordinates.json").write_text(
                json.dumps(coordinates, ensure_ascii=False, indent=2), encoding="utf-8")
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
        detail_regions = (visual.get("detail_regions_xz", [])
                          if output.get("mesh_navmesh_details", True) else [])
        for index, region in enumerate(detail_regions):
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


def _add_sampled_heading_arrows(ax, frames, trajectory, trajectory_color):
    """沿路线距离和显著拐点抽样绘制统一样式的车头朝向箭头。"""
    from matplotlib.patches import FancyArrowPatch

    if len(frames) < 2:
        return
    positions = np.asarray([frame[0] for frame in frames], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
        np.diff(positions[:, [0, 2]], axis=0), axis=1))))
    distance = float(cumulative[-1])
    sample_count = min(9, max(2, int(distance // 8.0)))
    samples = set(np.searchsorted(
        cumulative, np.linspace(distance * 0.08, distance * 0.92, sample_count)
    ).clip(0, len(frames) - 1).tolist())

    yaw = np.unwrap(np.asarray([frame[1] for frame in frames], dtype=np.float64))
    window = max(2, int(round(0.5 * 10)))
    turn_mask = np.zeros(len(frames), dtype=bool)
    for index in range(window, len(frames) - window):
        turn_mask[index] = abs(yaw[index + window] - yaw[index - window]) >= np.deg2rad(20)
    turn_indices = np.flatnonzero(turn_mask)
    runs = []
    if len(turn_indices):
        start = previous = int(turn_indices[0])
        for index in turn_indices[1:]:
            index = int(index)
            if index - previous > 15:
                runs.append((start, previous))
                start = index
            previous = index
        runs.append((start, previous))
    turn_samples = set()
    for start, end in runs:
        if abs(yaw[end + window] - yaw[start - window]) >= np.deg2rad(25):
            turn_samples.add((start + end) // 2)
    samples.update(turn_samples)

    for index in sorted(samples):
        x, y, z = positions[index]
        heading = yaw[index]
        length_m = 2.6 if index in turn_samples else 1.6
        end = (x - np.sin(heading) * length_m,
               z - np.cos(heading) * length_m)
        is_turn = index in turn_samples
        for color, linewidth, scale, zorder in (
                ("white", 3.5 if is_turn else 2.8,
                 17 if is_turn else 13, 7),
                ("#18211d", 1.7 if is_turn else 1.3,
                 13 if is_turn else 9, 8)):
            ax.add_patch(FancyArrowPatch(
                (x, z), end, arrowstyle="-|>", mutation_scale=scale,
                linewidth=linewidth, color=color, shrinkA=0, shrinkB=0,
                transform=ax.transData, clip_on=True, zorder=zorder))

