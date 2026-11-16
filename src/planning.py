"""按 YAML 配置运行 Habitat-GS 的 NavMesh、导航与 Gaussian RGB 闭环验证。"""
import math
from functools import lru_cache
from pathlib import Path

import habitat_sim as hs
import numpy as np
import yaml
from habitat_sim.utils.common import quat_from_angle_axis
from src.motion import build_linear_segment, yaw_steps
from src.motion_primitives import (ACTION_TYPES, assign_batch_actions,
                                   generate_curve_action, smooth_clothoid_polyline,
                                   time_parameterize_path)


ROOT = Path(__file__).resolve().parents[1]
_SNAPPED_CANDIDATE_CACHE = {}


def navmesh_agent_radius(nav_config):
    """计算 NavMesh 烘焙用的代理半径，并叠加额外障碍物净距。"""
    width = float(nav_config["agent_width_m"])
    clearance = float(nav_config.get("obstacle_clearance_m", 0.0))
    if not np.isfinite(width) or width <= 0:
        raise ValueError("navmesh.agent_width_m 必须是有限正数")
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("navmesh.obstacle_clearance_m 必须是有限非负数")
    return width / 2.0 + clearance


def resolve_path(value):
    """将配置中的相对路径解释为项目根目录下的路径。"""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    """读取并检查必要的运行参数，避免错误配置进入昂贵的渲染过程。"""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    motion, camera = config["motion"], config["camera"]
    nav = config["navigation"]
    assert motion["linear_speed_mps"] > 0 and motion["angular_speed_radps"] > 0
    for key in ("max_acceleration_mps2", "max_deceleration_mps2"):
        assert motion.get(key, 0) > 0, f"motion.{key} 必须为正数"
    assert motion["fps"] > 0 and camera["width"] > 0 and camera["height"] > 0
    assert config["navmesh"]["agent_width_m"] > 0 and config["navmesh"]["agent_height_m"] > 0
    avoidance = config.get("avoidance", {})
    if avoidance.get("enabled", False):
        assert avoidance.get("obstacle_radius_m", 0) > 0
        assert avoidance.get("obstacle_height_m", 0) > 0.5
        assert avoidance.get("trigger_distance_m", -1) >= 0
        assert avoidance.get("safety_margin_m", -1) >= 0
        assert avoidance.get("grid_resolution_m", 0) > 0
        assert avoidance.get("max_search_cells", 0) > 0
        assert 0 < avoidance.get("placement_fraction", 0.5) < 1
    assert nav["sample_attempts"] > 0
    topdown_max_points = config["output"].get("topdown_max_points")
    assert topdown_max_points is None or topdown_max_points > 0
    assert nav["min_route_length_m"] >= 0 and nav["max_route_length_m"] > nav["min_route_length_m"]
    assert (nav["start"] is None) == (nav["goal"] is None), "start/goal 必须同时指定或同时留空"
    if nav.get("use_camera_trajectory_prior", False):
        assert nav.get("camera_trajectory_buffer_m", 0) > 0
        assert nav.get("camera_trajectory_file")
    motion = config["motion"]
    assert motion.get("generation_profile", "rich") in ("rich", "legacy")
    assert motion.get("model", "differential_drive") in (
        "differential_drive", "ackermann")
    assert motion.get("minimum_turn_radius_m", 1.0) > 0
    assert motion.get("max_lateral_acceleration_mps2", 1.0) > 0
    assert motion.get("curve_sample_spacing_m", 0.05) > 0
    if motion.get("generation_profile", "rich") == "rich":
        assert 0 <= motion.get("min_generated_route_length_m", 50.0) < motion.get(
            "max_generated_route_length_m", 100.0)
        generation = config["generation"]
        assert generation.get("composition_mode", "mixed") in (
            "mixed", "single", "composite")
        assert 0 <= generation.get("composite_fraction", 0.70) <= 1
        assert generation.get("action_attempts_per_scale", 20) > 0
    tangent_probability = config.get("motion", {}).get("actions", {}).get(
        "tangent_connector_probability", 0.35)
    assert 0.0 <= tangent_probability <= 1.0
    return config


def yaw_quaternion(yaw):
    """Habitat 绕 Y 轴旋转；JSON 导出时统一转换为 x,y,z,w。"""
    return quat_from_angle_axis(float(yaw), np.array([0.0, 1.0, 0.0]))


def quaternion_xyzw(rotation):
    """Habitat 使用 numpy-quaternion，其属性为 x、y、z、w。"""
    return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]


def sample_triangle_point(triangles, weights, rng):
    """按三角形面积抽取 NavMesh 点，避免在不可行空间做盲目坐标采样。"""
    index = int(rng.choice(len(triangles), p=weights))
    a, b, c = triangles[index]
    r1, r2 = np.sqrt(rng.random()), rng.random()
    return (1.0 - r1) * a + r1 * (1.0 - r2) * b + r1 * r2 * c


def load_camera_trajectory(config):
    """读取源 Z-up cameras.json 的相机中心并转换为 Habitat Y-up 坐标。"""
    nav = config["navigation"]
    if not nav.get("use_camera_trajectory_prior", False):
        return None
    path = resolve_path(nav["camera_trajectory_file"])
    path = path.resolve()
    stat = path.stat()
    # 将文件大小和修改时间纳入缓存键，避免用户替换 cameras.json 后继续使用旧轨迹。
    return _load_camera_trajectory_file(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=4)
def _load_camera_trajectory_file(path_string, file_size, file_mtime_ns):
    """批次中复用已解析的 JSON 与 KD-tree，避免每条路线重复读取大文件。"""
    import json
    from scipy.spatial import cKDTree

    path = Path(path_string)
    with path.open(encoding="utf-8") as stream:
        cameras = json.load(stream)
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"相机轨迹文件为空或格式无效: {path}")
    source = np.asarray([camera["position"] for camera in cameras], dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError(f"cameras.json position 必须是有限的 N×3 数组: {path}")
    # prepare_scene.py 使用同一旋转：(x, y, z) -> (x, z, -y)。
    # 保留原始顺序供采集轨迹可视化使用；KD-tree 使用去重位置以减少查询开销。
    ordered_habitat = source[:, [0, 2, 1]].copy()
    ordered_habitat[:, 2] *= -1.0
    habitat = np.unique(np.round(ordered_habitat, decimals=3), axis=0)
    # 一个视频帧由多个方向相机组成；按 img_name 的“视频段_帧号_相机号”
    # 分组并合并相机中心，避免把不同朝向的同帧记录或不同视频段硬连成线。
    capture_groups = {}
    for camera, position in zip(cameras, ordered_habitat):
        name = str(camera.get("img_name", ""))
        parts = name.rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            sequence, frame_index = parts[0], int(parts[1])
            capture_groups.setdefault(sequence, {}).setdefault(frame_index, []).append(position)
    capture_segments = []
    for sequence in capture_groups.values():
        frames = [np.mean(positions, axis=0) for _, positions in sorted(sequence.items())]
        segment = np.asarray(frames, dtype=np.float64)
        if len(segment):
            capture_segments.append(segment)
    if not capture_segments:
        capture_segments = [ordered_habitat]
    return {"positions": habitat, "ordered_positions": ordered_habitat,
            "capture_segments": capture_segments,
            "tree": cKDTree(habitat[:, [0, 2]]),
            "source_file": str(path.resolve()), "source_count": len(cameras),
            "unique_count": len(habitat)}


def route_within_camera_buffer(points, camera_prior, buffer_m, sample_spacing_m=0.5):
    """沿路径每隔固定距离检查 X/Z 到最近采集相机中心轨迹的距离。"""
    if camera_prior is None:
        return True
    points = np.asarray(points, dtype=np.float64)
    samples = []
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        count = max(1, int(np.ceil(length / sample_spacing_m)))
        samples.append(np.linspace(start, end, count + 1))
    if len(points) == 1:
        samples = [points]
    query = np.concatenate(samples, axis=0)[:, [0, 2]]
    distances, _ = camera_prior["tree"].query(query, k=1)
    return bool(np.all(distances <= buffer_m + 1e-6))


def _snapped_camera_candidates(pathfinder, camera_prior, navmesh_path, snap_limit):
    """缓存相机轨迹吸附结果；NavMesh 或先验文件变化时自动换缓存键。"""
    nav_path = Path(navmesh_path).resolve()
    camera_path = Path(camera_prior["source_file"]).resolve()
    nav_stat, camera_stat = nav_path.stat(), camera_path.stat()
    key = (str(nav_path), nav_stat.st_size, nav_stat.st_mtime_ns,
           str(camera_path), camera_stat.st_size, camera_stat.st_mtime_ns,
           round(float(snap_limit), 6))
    cached = _SNAPPED_CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached[0].copy(), cached[1].copy()

    points = []
    for position in camera_prior["positions"]:
        point = np.asarray(pathfinder.snap_point(position.astype(np.float32)), dtype=np.float64)
        if (np.isfinite(point).all()
                and np.linalg.norm(point[[0, 2]] - position[[0, 2]]) <= snap_limit
                and pathfinder.get_island(point) >= 0):
            points.append(point)
    if not points:
        raise RuntimeError("cameras.json 轨迹点无法吸附到 NavMesh；请检查坐标变换/高度或关闭轨迹端点优先")
    points = np.asarray(points)
    islands = np.asarray([pathfinder.get_island(point) for point in points], dtype=np.int64)
    if len(_SNAPPED_CANDIDATE_CACHE) >= 4:
        _SNAPPED_CANDIDATE_CACHE.pop(next(iter(_SNAPPED_CANDIDATE_CACHE)))
    _SNAPPED_CANDIDATE_CACHE[key] = (points, islands)
    return points.copy(), islands.copy()


def classify_camera_trajectory_islands(island_areas, candidate_islands,
                                       min_points=20,
                                       min_points_per_100m2=0.5):
    """根据采集轨迹覆盖量判断哪些连通岛可以参与导航。

    `candidate_islands` 来自已经吸附到 NavMesh 的相机位置，因此不会把
    NavMesh 外的道路点误计入统计。面积密度用“每 100 平方米的轨迹点数”
    表示，同时要求绝对点数达到阈值，避免一个很小的偶然岛屿被保留。
    返回布尔保留掩码、各岛点数和密度，供路线选择与可视化共用。
    """
    areas = np.asarray(island_areas, dtype=np.float64)
    islands = np.asarray(candidate_islands, dtype=np.int64)
    counts = np.bincount(islands[islands >= 0], minlength=len(areas)).astype(np.int64)
    density = np.divide(counts * 100.0, areas, out=np.zeros_like(areas), where=areas > 1e-9)
    keep = (counts >= int(min_points)) & (density >= float(min_points_per_100m2))
    return keep, counts, density


def camera_trajectory_island_stats(pathfinder, camera_prior, navmesh_path, config):
    """计算相机轨迹对每个 NavMesh 岛的覆盖统计，并给出过滤结果。"""
    nav = config["navigation"]
    areas = np.asarray([pathfinder.island_area(i) for i in range(pathfinder.num_islands)],
                       dtype=np.float64)
    points, islands = _snapped_camera_candidates(
        pathfinder, camera_prior, navmesh_path,
        max(float(nav.get("camera_trajectory_buffer_m", 0.0)), 1.0))
    keep, counts, density = classify_camera_trajectory_islands(
        areas, islands,
        nav.get("camera_trajectory_density_min_points", 20),
        nav.get("camera_trajectory_density_min_points_per_100m2", 0.5))
    return {
        "areas_m2": areas,
        "counts": counts,
        "density_per_100m2": density,
        "keep": keep,
        "candidate_points": points,
        "candidate_islands": islands,
        "min_points": int(nav.get("camera_trajectory_density_min_points", 20)),
        "min_points_per_100m2": float(nav.get("camera_trajectory_density_min_points_per_100m2", 0.5)),
    }


def choose_route(pathfinder, config, navmesh_path=None):
    """按岛面积和三角面采样起终点，再以真实测地距离筛选路线。"""
    nav = config["navigation"]
    camera_prior = load_camera_trajectory(config)
    prefer_trajectory_endpoints = bool(
        camera_prior is not None and nav.get("prefer_camera_trajectory_endpoints", True))
    island_areas = np.asarray([pathfinder.island_area(i) for i in range(pathfinder.num_islands)])
    if nav["island_index"] is not None:
        islands = [int(nav["island_index"])]
    else:
        islands = [int(i) for i in np.argsort(island_areas)[::-1]
                   if island_areas[i] >= nav["min_island_area_m2"]]
    if not islands:
        raise RuntimeError("没有达到 min_island_area_m2 的可行走连通岛")
    density_stats = None
    if (camera_prior is not None
            and nav.get("use_camera_trajectory_density_filter", True)):
        density_stats = camera_trajectory_island_stats(pathfinder, camera_prior, navmesh_path, config)
        islands = [island for island in islands if bool(density_stats["keep"][island])]
        if not islands:
            raise RuntimeError(
                "采集轨迹密度过滤后没有可行走连通岛；请降低 "
                "camera_trajectory_density_min_points 或 "
                "camera_trajectory_density_min_points_per_100m2，或关闭过滤")
    if nav["start"] is not None:
        start = pathfinder.snap_point(np.asarray(nav["start"], dtype=np.float32))
        goal = pathfinder.snap_point(np.asarray(nav["goal"], dtype=np.float32))
        if pathfinder.get_island(start) != pathfinder.get_island(goal):
            raise RuntimeError("配置的起点和终点不属于同一个 NavMesh 连通岛")
        if density_stats is not None and not bool(density_stats["keep"][pathfinder.get_island(start)]):
            raise RuntimeError("配置的起点/终点位于采集轨迹稀疏而被过滤的连通岛")
        candidates = [(start, goal)]
    else:
        rng = np.random.default_rng(nav["seed"])
        trajectory_candidates = []
        if camera_prior is not None and nav.get("prefer_camera_trajectory_endpoints", True):
            # 轨迹相机中心只用 XYZ 位置，吸附结果按 NavMesh/轨迹文件状态缓存。
            snap_limit = max(float(nav["camera_trajectory_buffer_m"]), 1.0)
            navmesh_path = (Path(navmesh_path) if navmesh_path is not None else
                            resolve_path(config["scene"]["directory"]) / "wanda.navmesh")
            trajectory_candidates, trajectory_islands = _snapped_camera_candidates(
                pathfinder, camera_prior, navmesh_path, snap_limit)
            allowed_islands = np.asarray(islands, dtype=np.int64)
            keep = np.isin(trajectory_islands, allowed_islands)
            trajectory_candidates = trajectory_candidates[keep]
            trajectory_islands = trajectory_islands[keep]
            if len(trajectory_candidates) < 2:
                raise RuntimeError("相机轨迹在所选 NavMesh 岛上不足两个候选点")
        else:
            trajectory_islands = None
        # 用该岛实际三角面跨度排除明显无法达到目标测地长度的小岛。
        viable = []
        for island in islands:
            triangles = np.asarray(pathfinder.build_navmesh_vertices(island)).reshape(-1, 3, 3)
            if not len(triangles):
                continue
            span = np.linalg.norm(np.ptp(triangles.reshape(-1, 3)[:, [0, 2]], axis=0))
            if span >= nav["min_route_length_m"]:
                areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                                      triangles[:, 2] - triangles[:, 0]), axis=1)
                viable.append((island, triangles, areas / areas.sum()))
        if not viable:
            raise RuntimeError("没有几何跨度足以达到 min_route_length_m 的 NavMesh 连通岛")
        def candidates_from_navmesh():
            """逐一生成候选点，找到合适路线后立即停止采样。"""
            for _ in range(nav["sample_attempts"]):
                if prefer_trajectory_endpoints:
                    start = trajectory_candidates[int(rng.integers(len(trajectory_candidates)))]
                    same_island = trajectory_candidates[trajectory_islands == pathfinder.get_island(start)]
                    if len(same_island) < 2:
                        continue
                    goal = same_island[int(rng.integers(len(same_island)))]
                    yield start, goal
                    continue
                island, triangles, weights = viable[int(rng.choice(len(viable), p=np.asarray(
                    [island_areas[x[0]] for x in viable]) / sum(island_areas[x[0]] for x in viable)))]
                a = sample_triangle_point(triangles, weights, rng)
                b = sample_triangle_point(triangles, weights, rng)
                yield a, b
        candidates = candidates_from_navmesh()
    for start, goal in candidates:
        path = hs.ShortestPath()
        path.requested_start, path.requested_end = start, goal
        if (pathfinder.find_path(path)
                and nav["min_route_length_m"] < path.geodesic_distance < nav["max_route_length_m"]
                and route_within_camera_buffer(
                    path.points, camera_prior, nav.get("camera_trajectory_buffer_m", 0.0))):
            # 路径几何可达不代表按当前 fps 和 try_step 投影后仍能通过每个转折点。
            # 在接受候选前使用实际运动学模拟筛掉这种路线，批量生成时继续采样。
            try:
                build_motion(pathfinder, path.points, config)
            except RuntimeError as error:
                if "NavMesh 投影导致轨迹无法到达路径转折点" not in str(error):
                    raise
                if nav["start"] is not None:
                    raise RuntimeError("配置路线无法通过 NavMesh 逐帧运动学验证") from error
                continue
            return path, int(pathfinder.get_island(path.requested_start))
        if nav["start"] is not None:
            raise RuntimeError("配置的起终点无法构成满足距离及采集轨迹缓冲区条件的 NavMesh 路径")
    raise RuntimeError("起终点采样达到上限，仍未找到符合距离与采集轨迹缓冲区条件的路线；可扩大缓冲区或增加采样次数")


def _legacy_build_motion(pathfinder, points, config):
    """沿旧折线路径生成停车原地转向与直线运动帧。"""
    motion = config["motion"]
    dt = 1.0 / motion["fps"]
    max_speed = motion["linear_speed_mps"]
    acceleration = motion["max_acceleration_mps2"]
    deceleration = motion["max_deceleration_mps2"]
    max_turn = motion["angular_speed_radps"] * dt
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("导航路径至少需要两个点")
    position = points[0].copy()
    delta = points[1] - points[0]
    yaw = math.atan2(-delta[0], -delta[2])
    frames = [(position.copy(), yaw, 0.0)]
    for target in points[1:]:
        target = np.asarray(target, dtype=np.float64)
        delta = target - position
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            continue
        desired = math.atan2(-delta[0], -delta[2])
        angle = (desired - yaw + math.pi) % (2 * math.pi) - math.pi
        for yaw in yaw_steps(yaw, desired, motion["angular_speed_radps"], dt):
            frames.append((position.copy(), yaw, len(frames) * dt))
        # 固定 fps 的梯形速度序列，起终速度为零，满足线加速度与制动上限。
        delta = target - position
        distance = float(np.linalg.norm(delta))
        direction = delta / distance
        segment_frames, position, _ = build_linear_segment(
            position, target, yaw, frames[-1][2], motion["fps"], max_speed,
            acceleration, deceleration, pathfinder.try_step, pathfinder.is_navigable)
        frames.extend(segment_frames)
    return frames


def _shortest_path(pathfinder, start, goal):
    """获取同一 NavMesh 连通岛内两点间的 Habitat 测地连接路径。"""
    path = hs.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(goal, dtype=np.float32)
    if not pathfinder.find_path(path):
        raise RuntimeError("动作锚点之间不存在 NavMesh 连接路径")
    return np.asarray(path.points, dtype=np.float64)


def _simplify_navmesh_path(pathfinder, points, spacing_m=0.1):
    """用 NavMesh 线段可见性贪心删去冗余路径点，保留真实不可直连的转折。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 2:
        return points.copy()
    island = int(pathfinder.get_island(points[0]))
    simplified = [points[0]]
    current = 0
    while current < len(points) - 1:
        selected = current + 1
        for candidate in range(len(points) - 1, current + 1, -1):
            start, end = points[current], points[candidate]
            distance = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
            count = max(1, int(math.ceil(distance / spacing_m)))
            samples = np.linspace(start, end, count + 1)
            if all(pathfinder.is_navigable(sample.astype(np.float32), 0.15)
                   and pathfinder.get_island(sample.astype(np.float32)) == island
                   for sample in samples):
                selected = candidate
                break
        simplified.append(points[selected])
        current = selected
    return np.asarray(simplified, dtype=np.float64)


def _smooth_navmesh_connector(pathfinder, points, config):
    """简化并圆滑 Habitat 连接路线，随后重新确认曲线全段落在同一 NavMesh 岛。"""
    motion = config["motion"]
    simplified = _simplify_navmesh_path(pathfinder, points)
    smoothed = smooth_clothoid_polyline(
        simplified, float(motion.get("minimum_turn_radius_m", 0.3)),
        float(motion.get("curve_sample_spacing_m", 0.05)))
    if not _validate_rich_path(pathfinder, smoothed, None, config):
        raise RuntimeError("平滑后的 NavMesh 连接路线超出可行走区域")
    return smoothed


def _path_endpoint_tangent(points, at_end):
    """从折线路径指定端点附近提取 XZ 平面行进切线。"""
    points = np.asarray(points, dtype=np.float64)
    pairs = (zip(range(len(points) - 1, 0, -1), range(len(points) - 2, -1, -1))
             if at_end else zip(range(1, len(points)), range(len(points) - 1)))
    for first, second in pairs:
        delta = points[first, [0, 2]] - points[second, [0, 2]]
        length = float(np.linalg.norm(delta))
        if length > 1e-6:
            return delta / length
    raise RuntimeError("运动段端点附近没有可用的水平切线")


def _maximum_discrete_curvature(points):
    """估算稠密折线路径的最大离散平面曲率。"""
    points = np.asarray(points, dtype=np.float64)
    delta = np.diff(points[:, [0, 2]], axis=0)
    distances = np.linalg.norm(delta, axis=1)
    valid = distances > 1e-6
    if np.count_nonzero(valid) < 2:
        return float("inf")
    courses = np.unwrap(np.arctan2(delta[valid, 1], delta[valid, 0]))
    valid_distances = distances[valid]
    if len(courses) < 2:
        return 0.0
    interval_distance = (valid_distances[:-1] + valid_distances[1:]) * 0.5
    curvature = np.abs(np.diff(courses)) / np.maximum(interval_distance, 1e-8)
    return float(curvature.max(initial=0.0))


def _try_tangent_connector(pathfinder, start, end, start_tangent, end_tangent,
                           island, camera_prior, config):
    """尝试沿端点切线构造 NavMesh 连接；任何几何或约束失败均返回 None。"""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    start_tangent = np.asarray(start_tangent, dtype=np.float64)
    start_tangent /= max(float(np.linalg.norm(start_tangent)), 1e-12)
    if end_tangent is not None:
        end_tangent = np.asarray(end_tangent, dtype=np.float64)
        end_tangent /= max(float(np.linalg.norm(end_tangent)), 1e-12)

    motion = config["motion"]
    spacing = float(motion.get("curve_sample_spacing_m", 0.05))
    radius = float(motion.get("minimum_turn_radius_m", 1.0))
    if end_tangent is None:
        stub_pairs = [(length, None) for length in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0)]
    else:
        stub_pairs = [(0.75, 0.75), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0),
                      (3.0, 3.0), (4.0, 4.0), (0.75, 1.5), (1.5, 0.75),
                      (1.0, 2.0), (2.0, 1.0)]

    for start_length, end_length in stub_pairs:
        start_stub_raw = start.copy()
        start_stub_raw[[0, 2]] += start_tangent * start_length
        start_stub = np.asarray(pathfinder.snap_point(
            start_stub_raw.astype(np.float32)), dtype=np.float64)
        if (not np.isfinite(start_stub).all()
                or np.linalg.norm(start_stub[[0, 2]] - start_stub_raw[[0, 2]]) > 0.15
                or pathfinder.get_island(start_stub) != island
                or not pathfinder.is_navigable(start_stub.astype(np.float32), 0.15)):
            continue

        end_stub = None
        if end_tangent is not None:
            end_stub_raw = end.copy()
            end_stub_raw[[0, 2]] -= end_tangent * float(end_length)
            end_stub = np.asarray(pathfinder.snap_point(
                end_stub_raw.astype(np.float32)), dtype=np.float64)
            if (not np.isfinite(end_stub).all()
                    or np.linalg.norm(end_stub[[0, 2]] - end_stub_raw[[0, 2]]) > 0.15
                    or pathfinder.get_island(end_stub) != island
                    or not pathfinder.is_navigable(end_stub.astype(np.float32), 0.15)):
                continue

        try:
            middle = _simplify_navmesh_path(
                pathfinder, _shortest_path(
                    pathfinder, start_stub,
                    end_stub if end_stub is not None else end))
            pieces = [start, start_stub, *middle[1:]]
            if end_stub is not None:
                pieces.extend((end_stub, end))
            candidate = [pieces[0]]
            for point in pieces[1:]:
                if np.linalg.norm(point - candidate[-1]) > 1e-5:
                    candidate.append(point)
            candidate = smooth_clothoid_polyline(
                np.asarray(candidate), radius, spacing)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue

        if not _validate_rich_path(pathfinder, candidate, camera_prior, config):
            continue
        curvature_limit = 1.0 / radius + 0.08
        if _maximum_discrete_curvature(candidate) > curvature_limit:
            continue
        try:
            actual_start_tangent = _path_endpoint_tangent(candidate, at_end=False)
            if np.dot(actual_start_tangent, start_tangent) < math.cos(math.radians(8.0)):
                continue
            if end_tangent is not None:
                actual_end_tangent = _path_endpoint_tangent(candidate, at_end=True)
                if np.dot(actual_end_tangent, end_tangent) < math.cos(math.radians(8.0)):
                    continue
        except RuntimeError:
            continue
        return candidate
    return None


def _merge_forward_operations(previous, connector, following=None):
    """合并已验证的切向连接，使连续段共享一次速度规划且保留事件距离。"""
    previous_points = np.asarray(previous["points"], dtype=np.float64)
    connector = np.asarray(connector, dtype=np.float64)
    pieces = [*previous_points, *connector[1:]]
    distance_offset = _path_length(previous_points) + _path_length(connector)
    events = [dict(event) for event in previous.get("events", [])]
    connector_count = int(previous.get("tangent_connector_count", 0)) + 1
    if following is not None:
        following_points = np.asarray(following["points"], dtype=np.float64)
        pieces.extend(following_points[1:])
        for event in following.get("events", []):
            shifted = dict(event)
            shifted["start_distance_m"] += distance_offset
            shifted["end_distance_m"] += distance_offset
            events.append(shifted)
        connector_count += int(following.get("tangent_connector_count", 0))
    return {"points": np.asarray(pieces), "reverse": False,
            "type": "continuous_forward", "events": events,
            "tangent_connector_count": connector_count}


def _path_length(points):
    """计算三维折线的实际几何弧长。"""
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _validate_rich_path(pathfinder, points, camera_prior, config):
    """对稠密曲线路径逐点和逐段检查 NavMesh、连通岛与采集轨迹缓冲。"""
    nav = config["navigation"]
    island = int(pathfinder.get_island(points[0]))
    if island < 0:
        return False
    for start, end in zip(points[:-1], points[1:]):
        horizontal_length = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        count = max(1, int(math.ceil(horizontal_length / 0.05)))
        samples = np.linspace(start, end, count + 1)
        for point in samples:
            point32 = point.astype(np.float32)
            if (not pathfinder.is_navigable(point32, 0.15)
                    or pathfinder.get_island(point32) != island):
                return False
        if not route_within_camera_buffer(
                np.asarray([start, end]), camera_prior,
                nav.get("camera_trajectory_buffer_m", 0.0), sample_spacing_m=0.25):
            return False
    return True


def _sample_dead_end(pathfinder, rng, island, config, camera_prior,
                    anchor_candidates=None):
    """在真实 NavMesh 采样前探测出可驶入且能原路倒出的盲端直廊。"""
    motion = config["motion"]
    settings = motion.get("actions", {})
    minimum_length = float(settings.get("dead_end_min_length_m", 8.0))
    maximum_length = float(settings.get("dead_end_max_length_m", 20.0))
    probe_spacing = float(settings.get("dead_end_probe_spacing_m", 0.5))
    side_clearance = (float(config["navmesh"]["agent_width_m"]) / 2.0
                      + float(config["navmesh"].get("obstacle_clearance_m", 0.0)))
    start = (np.asarray(anchor_candidates[int(rng.integers(len(anchor_candidates)))],
                        dtype=np.float64) if anchor_candidates is not None
             and len(anchor_candidates) else np.asarray(
                 pathfinder.get_random_navigable_point(island_index=island), dtype=np.float64))
    yaw = float(rng.uniform(-math.pi, math.pi))
    direction = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    last_valid = start.copy()
    travelled = 0.0
    for distance in np.arange(probe_spacing, maximum_length + probe_spacing,
                              probe_spacing):
        raw = start + direction * distance
        snapped = np.asarray(pathfinder.snap_point(raw.astype(np.float32)), dtype=np.float64)
        if (not np.isfinite(snapped).all()
                or np.linalg.norm(snapped[[0, 2]] - raw[[0, 2]]) > 0.15
                or pathfinder.get_island(snapped) != island
                or not pathfinder.is_navigable(snapped.astype(np.float32), 0.15)):
            break
        last_valid, travelled = snapped, float(distance)
    if not minimum_length <= travelled <= maximum_length:
        raise RuntimeError("本次采样没有找到长度合适的真实盲端")
    ahead = last_valid + direction * probe_spacing
    ahead_snap = np.asarray(pathfinder.snap_point(ahead.astype(np.float32)), dtype=np.float64)
    if (np.isfinite(ahead_snap).all()
            and np.linalg.norm(ahead_snap[[0, 2]] - ahead[[0, 2]]) <= 0.15
            and pathfinder.get_island(ahead_snap) == island
            and pathfinder.is_navigable(ahead_snap.astype(np.float32), 0.15)):
        raise RuntimeError("前向探测未确认真实盲端")

    right = np.array([direction[2], 0.0, -direction[0]])
    for sign in (-1.0, 1.0):
        side = last_valid + right * sign * side_clearance
        if (not pathfinder.is_navigable(side.astype(np.float32), 0.15)
                or pathfinder.get_island(side) != island):
            raise RuntimeError("盲端两侧净宽不足以容纳代理安全倒车")
    if not route_within_camera_buffer(
            np.asarray([start, last_valid]), camera_prior,
            config["navigation"].get("camera_trajectory_buffer_m", 0.0)):
        raise RuntimeError("盲端超出 cameras.json 轨迹缓冲区")
    outbound = np.asarray([start, last_valid], dtype=np.float64)
    return [{"points": outbound, "reverse": False},
            {"points": outbound[::-1].copy(), "reverse": True}], yaw


def _sample_action_segment(pathfinder, action, rng, island, scale,
                           camera_prior, config, anchor_candidates=None):
    """按动作类型在指定 NavMesh 连通岛生成并验证一个动作模板。"""
    if action == "dead_end_reverse":
        operations, _ = _sample_dead_end(
            pathfinder, rng, island, config, camera_prior, anchor_candidates)
        return operations, [{"type": action}]

    settings = config["motion"].get("actions", {})
    start = (np.asarray(anchor_candidates[int(rng.integers(len(anchor_candidates)))],
                        dtype=np.float64) if anchor_candidates is not None
             and len(anchor_candidates) else np.asarray(
                 pathfinder.get_random_navigable_point(island_index=island), dtype=np.float64))
    yaw = float(rng.uniform(-math.pi, math.pi))
    radius = float(settings.get("curve_radius_m", 4.0)) * scale
    if action == "arc_u_turn":
        angle = math.pi * scale
    elif action == "s_curve":
        angle = float(settings.get("s_curve_angle_deg", 55.0)) * math.pi / 180.0 * scale
    else:
        angle = float(settings.get("tangent_curve_angle_deg", 70.0)) * math.pi / 180.0 * scale
    segment = generate_curve_action(start, yaw, action, radius, angle,
                                    float(config["motion"].get(
                                        "curve_sample_spacing_m", 0.05)))
    projected = []
    for point in segment:
        snapped = np.asarray(pathfinder.snap_point(point.astype(np.float32)),
                             dtype=np.float64)
        if (not np.isfinite(snapped).all()
                or np.linalg.norm(snapped[[0, 2]] - point[[0, 2]]) > 0.15
                or pathfinder.get_island(snapped) != island):
            raise RuntimeError(f"{action} 曲线无法连续吸附到 NavMesh")
        projected.append(snapped)
    segment = np.asarray(projected, dtype=np.float64)
    if not _validate_rich_path(pathfinder, segment, camera_prior, config):
        raise RuntimeError(f"{action} 曲线未通过 NavMesh/轨迹缓冲验证")
    frames = time_parameterize_path(segment, config)
    events = [{"type": action, "start_frame": 0, "end_frame": len(frames) - 1}]
    return [{"points": segment, "reverse": False}], events


def _connect_segments(pathfinder, action_operations, rng, island, config):
    """保留动作模板曲率，仅平滑段间 NavMesh 连接路并补足目标路程。"""
    motion = config["motion"]
    minimum = float(motion.get("min_generated_route_length_m", 50.0))
    maximum = float(motion.get("max_generated_route_length_m", 100.0))
    settings = motion.get("actions", {})
    tangent_probability = float(settings.get("tangent_connector_probability", 0.35))
    operations = []
    for group in action_operations:
        for operation in group:
            points = np.asarray(operation["points"], dtype=np.float64)
            if operations and np.linalg.norm(operations[-1]["points"][-1] - points[0]) > 1e-4:
                previous = operations[-1]
                following = {**operation, "points": points}
                tangent_connector = None
                if (not previous.get("reverse", False)
                        and not following.get("reverse", False)
                        and rng.random() < tangent_probability):
                    try:
                        tangent_connector = _try_tangent_connector(
                            pathfinder, previous["points"][-1], points[0],
                            _path_endpoint_tangent(previous["points"], at_end=True),
                            _path_endpoint_tangent(points, at_end=False), island,
                            load_camera_trajectory(config), config)
                    except RuntimeError:
                        tangent_connector = None
                if tangent_connector is not None:
                    operations[-1] = _merge_forward_operations(
                        previous, tangent_connector, following)
                    continue
                connector = _smooth_navmesh_connector(
                    pathfinder, _shortest_path(
                        pathfinder, previous["points"][-1], points[0]), config)
                operations.append({"points": connector, "reverse": False,
                                   "type": "connector", "events": []})
            operations.append({**operation, "points": points})

    def total_length(items):
        """累计正向与倒车操作的实际移动长度。"""
        return sum(_path_length(item["points"]) for item in items)

    length = total_length(operations)
    if length > maximum:
        raise RuntimeError("动作模板与连接路径已经超过最大累计路程")
    for _ in range(int(settings.get("route_extension_attempts", 40))):
        if length >= minimum:
            return operations
        remaining = maximum - length
        anchor = operations[-1]["points"][-1]
        goal = np.asarray(pathfinder.get_random_navigable_point_near(
            anchor.astype(np.float32), max(remaining, minimum - length), 250, island),
            dtype=np.float64)
        tangent_connector = None
        previous = operations[-1]
        if (not previous.get("reverse", False)
                and rng.random() < tangent_probability):
            try:
                tangent_connector = _try_tangent_connector(
                    pathfinder, anchor, goal,
                    _path_endpoint_tangent(previous["points"], at_end=True), None,
                    island, load_camera_trajectory(config), config)
            except RuntimeError:
                tangent_connector = None
        try:
            if tangent_connector is not None:
                connector = tangent_connector
            else:
                connector = _shortest_path(pathfinder, anchor, goal)
                if _path_length(connector) <= 1.0:
                    continue
                connector = _smooth_navmesh_connector(pathfinder, connector, config)
        except RuntimeError:
            continue
        extension = {"points": connector, "reverse": False,
                     "type": "connector", "events": []}
        candidate_length = length + _path_length(connector)
        if candidate_length <= maximum:
            if tangent_connector is not None:
                operations[-1] = _merge_forward_operations(previous, connector)
            else:
                operations.append(extension)
            length = candidate_length
    raise RuntimeError(f"无法将动作模板扩展到 {minimum:.1f}–{maximum:.1f} 米实际路程")


def _build_operation_frames(operations, config):
    """按动作段分别时间参数化并保持停启处朝向连续、倒车时车头朝向不变。"""
    motion = config["motion"]
    frames = []
    motion_events = []
    route_points = []
    current_time = 0.0
    current_yaw = None
    for operation in operations:
        points = np.asarray(operation["points"], dtype=np.float64)
        if len(points) < 2:
            continue
        segment_frames = time_parameterize_path(
            points, config, reverse=bool(operation.get("reverse", False)))
        if current_yaw is None:
            frames.extend(segment_frames)
            current_time = float(frames[-1][2])
            segment_anchor = 0
        else:
            start_yaw = float(segment_frames[0][1])
            angle = abs((start_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi)
            if angle > math.radians(3.0):
                if motion.get("model", "differential_drive") == "ackermann":
                    raise RuntimeError("Ackermann 底盘无法在原地完成连接段朝向调整")
                turn_frames = yaw_steps(current_yaw, start_yaw,
                                        float(motion["angular_speed_radps"]),
                                        1.0 / float(motion["fps"]))
                for yaw in turn_frames:
                    current_time += 1.0 / float(motion["fps"])
                    frames.append((frames[-1][0].copy(), float(yaw), current_time))
            elif operation.get("reverse", False):
                for _ in range(int(round(float(motion.get("actions", {}).get(
                        "dead_end_pause_s", 0.5)) * float(motion["fps"])))):
                    current_time += 1.0 / float(motion["fps"])
                    frames.append((frames[-1][0].copy(), current_yaw, current_time))
            segment_anchor = len(frames) - 1
            frames.extend((frame[0], frame[1], frame[2] + current_time)
                          for frame in segment_frames[1:])
            current_time = float(frames[-1][2])
        if not route_points:
            route_points.extend(points)
        elif np.linalg.norm(route_points[-1] - points[0]) > 1e-5:
            route_points.extend(points)
        else:
            route_points.extend(points[1:])
        for event in operation.get("events", []):
            positions = np.asarray([frame[0][[0, 2]] for frame in frames], dtype=np.float64)
            local_positions = positions[segment_anchor:]
            local_progress = np.concatenate(([0.0], np.cumsum(
                np.linalg.norm(np.diff(local_positions, axis=0), axis=1))))
            event_start_frame = segment_anchor + int(np.argmin(np.abs(
                local_progress - event["start_distance_m"])))
            event_end_frame = segment_anchor + int(np.argmin(np.abs(
                local_progress - event["end_distance_m"])))
            motion_events.append({"type": event["type"],
                                  "start_frame": event_start_frame,
                                  "end_frame": max(event_start_frame + 1,
                                                   event_end_frame)})
        if operation.get("type") == "dead_end_reverse":
            motion_events.append({"type": "dead_end_reverse",
                                  "start_frame": segment_anchor,
                                  "end_frame": len(frames) - 1})
        current_yaw = float(frames[-1][1])
    if len(frames) < 2:
        raise RuntimeError("运动操作未生成足够帧")
    return np.asarray(route_points, dtype=np.float64), frames, motion_events


def build_rich_motion(pathfinder, config, batch_index=0, batch_size=1):
    """排程动作模板、构造长距离连接路线并输出可复现的运动与诊断记录。"""
    nav = config["navigation"]
    motion = config["motion"]
    generation = config["generation"]
    batch_seed = int(generation.get("batch_seed", nav["seed"]))
    rng = np.random.default_rng(np.random.SeedSequence([
        int(nav["seed"]), int(batch_index), int(batch_size), 0xC10A]))
    action_plan = assign_batch_actions(
        batch_seed, int(batch_size), int(batch_index),
        generation.get("composition_mode", "mixed"),
        float(generation.get("composite_fraction", 0.70)))
    camera_prior = load_camera_trajectory(config)
    islands = [index for index in range(pathfinder.num_islands)
               if pathfinder.island_area(index) >= nav["min_island_area_m2"]]
    anchor_candidates = {}
    if camera_prior is not None and nav.get("prefer_camera_trajectory_endpoints", True):
        navmesh_path = resolve_path(config["scene"]["directory"]) / "wanda.navmesh"
        camera_points, camera_islands = _snapped_camera_candidates(
            pathfinder, camera_prior, navmesh_path,
            max(float(nav["camera_trajectory_buffer_m"]), 1.0))
        density_stats = None
        if nav.get("use_camera_trajectory_density_filter", True):
            density_stats = camera_trajectory_island_stats(
                pathfinder, camera_prior, navmesh_path, config)
            islands = [island for island in islands if density_stats["keep"][island]]
        for island in islands:
            anchor_candidates[island] = camera_points[camera_islands == island]
        islands = [island for island in islands
                   if len(anchor_candidates.get(island, [])) > 0]
    if nav.get("island_index") is not None:
        islands = [int(nav["island_index"])]
        if camera_prior is not None and nav.get("prefer_camera_trajectory_endpoints", True):
            if not len(anchor_candidates.get(islands[0], [])):
                raise RuntimeError("指定 NavMesh 岛没有可用采集轨迹锚点")
    if not islands:
        raise RuntimeError("没有面积符合条件的 NavMesh 连通岛")

    diagnostics = {"generation_profile": "rich", "composition": action_plan["composition"],
                   "actions": [], "attempts": 0}
    scale_factors = (1.0, 0.85, 0.70)
    attempts_per_scale = int(generation.get("action_attempts_per_scale", 20))
    for scale in scale_factors:
        for _ in range(attempts_per_scale):
            diagnostics["attempts"] += 1
            island = int(rng.choice(islands))
            action_operations, event_records = [], []
            try:
                for action in action_plan["actions"]:
                    operations, events = _sample_action_segment(
                        pathfinder, action, rng, island, scale,
                        camera_prior, config, anchor_candidates.get(island))
                    for operation in operations:
                        operation["type"] = action
                        operation["events"] = ([{
                            "type": action,
                            "start_distance_m": 0.0,
                            "end_distance_m": _path_length(operation["points"]),
                        }] if operation is operations[0] and action != "dead_end_reverse"
                            else [])
                        if not _validate_rich_path(pathfinder, operation["points"],
                                                   camera_prior, config):
                            raise RuntimeError(f"{action} 操作未通过 NavMesh/先验验证")
                    action_operations.append(operations)
                    event_records.extend(events)
                operations = _connect_segments(pathfinder, action_operations,
                                               rng, island, config)
                points, frames, motion_events = _build_operation_frames(operations, config)
                if not _validate_rich_path(pathfinder, points, camera_prior, config):
                    raise RuntimeError("完整动作路径未通过 NavMesh/轨迹缓冲验证")
                route_length = float(sum(_path_length(operation["points"])
                                         for operation in operations))
                if not float(motion.get("min_generated_route_length_m", 50.0)) <= route_length <= float(
                        motion.get("max_generated_route_length_m", 100.0)):
                    raise RuntimeError("动作路线累计路程不在配置范围内")
                actual_length = float(sum(np.linalg.norm(
                    frames[index][0] - frames[index - 1][0])
                    for index in range(1, len(frames))))
                minimum = float(motion.get("min_generated_route_length_m", 50.0))
                maximum = float(motion.get("max_generated_route_length_m", 100.0))
                if not minimum <= actual_length <= maximum:
                    raise RuntimeError("时间参数化后的实际逐帧路程超出目标范围")
                diagnostics.update({"actions": event_records,
                                    "action_types": action_plan["actions"],
                                    "action_scale": scale,
                                    "cumulative_distance_m": actual_length,
                                    "route_island": island,
                                    "tangent_connector_count": sum(
                                        int(operation.get("tangent_connector_count", 0))
                                        for operation in operations)})
                diagnostics["motion_events"] = motion_events
                return points.astype(np.float32), island, frames, diagnostics
            except RuntimeError as error:
                diagnostics["last_error"] = str(error)
                continue
    raise RuntimeError(
        f"丰富运动路线在三个幅度等级及 {attempts_per_scale} 次重采样后仍不可行："
        f"{diagnostics.get('last_error', 'unknown')}；动作={action_plan['actions']}")


def build_motion(pathfinder, points, config):
    """按运动配置选择旧折线模式或曲率受控的连续前进轨迹。"""
    if config.get("motion", {}).get("generation_profile", "rich") == "legacy":
        return _legacy_build_motion(pathfinder, points, config)
    return time_parameterize_path(points, config)
