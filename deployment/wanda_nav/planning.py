"""按 YAML 配置运行 Habitat-GS 的 NavMesh、导航与 Gaussian RGB 闭环验证。"""
import math
from functools import lru_cache
from pathlib import Path

import habitat_sim as hs
import numpy as np
import yaml
from habitat_sim.utils.common import quat_from_angle_axis
from .motion import build_linear_segment, yaw_steps


ROOT = Path(__file__).resolve().parents[2]
_SNAPPED_CANDIDATE_CACHE = {}


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
    assert nav["sample_attempts"] > 0
    topdown_max_points = config["output"].get("topdown_max_points")
    assert topdown_max_points is None or topdown_max_points > 0
    assert nav["min_route_length_m"] >= 0 and nav["max_route_length_m"] > nav["min_route_length_m"]
    assert (nav["start"] is None) == (nav["goal"] is None), "start/goal 必须同时指定或同时留空"
    if nav.get("use_camera_trajectory_prior", False):
        assert nav.get("camera_trajectory_buffer_m", 0) > 0
        assert nav.get("camera_trajectory_file")
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
    habitat = source[:, [0, 2, 1]].copy()
    habitat[:, 2] *= -1.0
    habitat = np.unique(np.round(habitat, decimals=3), axis=0)
    return {"positions": habitat, "tree": cKDTree(habitat[:, [0, 2]]),
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
    if nav["start"] is not None:
        start = pathfinder.snap_point(np.asarray(nav["start"], dtype=np.float32))
        goal = pathfinder.snap_point(np.asarray(nav["goal"], dtype=np.float32))
        if pathfinder.get_island(start) != pathfinder.get_island(goal):
            raise RuntimeError("配置的起点和终点不属于同一个 NavMesh 连通岛")
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
            return path, int(pathfinder.get_island(path.requested_start))
        if nav["start"] is not None:
            raise RuntimeError("配置的起终点无法构成满足距离及采集轨迹缓冲区条件的 NavMesh 路径")
    raise RuntimeError("起终点采样达到上限，仍未找到符合距离与采集轨迹缓冲区条件的路线；可扩大缓冲区或增加采样次数")


def build_motion(pathfinder, points, config):
    """差速/全向底盘运动：直线梯形速度曲线，停车后限角速度原地转向。"""
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
