"""按 YAML 配置运行 Habitat-GS 的 NavMesh、导航与 Gaussian RGB 闭环验证。"""
import math
from functools import lru_cache
from pathlib import Path

import habitat_sim as hs
import numpy as np
import yaml
from habitat_sim.utils.common import quat_from_angle_axis


ROOT = Path(__file__).resolve().parents[2]


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
    return _load_camera_trajectory_file(str(path.resolve()))


@lru_cache(maxsize=4)
def _load_camera_trajectory_file(path_string):
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


def choose_route(pathfinder, config):
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
            # 轨迹相机中心只用 XYZ 位置，逐点吸附到可走面后丢弃超出邻近阈值的点。
            snap_limit = max(float(nav["camera_trajectory_buffer_m"]), 1.0)
            for position in camera_prior["positions"]:
                point = np.asarray(pathfinder.snap_point(position.astype(np.float32)), dtype=np.float64)
                if (np.isfinite(point).all()
                        and np.linalg.norm(point[[0, 2]] - position[[0, 2]]) <= snap_limit
                        and pathfinder.get_island(point) >= 0):
                    trajectory_candidates.append(point)
            if not trajectory_candidates:
                raise RuntimeError("cameras.json 轨迹点无法吸附到 NavMesh；请检查坐标变换/高度或关闭轨迹端点优先")
            trajectory_candidates = np.asarray(trajectory_candidates)
            trajectory_islands = np.asarray([pathfinder.get_island(p) for p in trajectory_candidates])
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
    """按线速度和角速度限制推进代理，并以 try_step 将每步约束在 NavMesh 上。"""
    motion = config["motion"]
    dt = 1.0 / motion["fps"]
    max_step = motion["linear_speed_mps"] * dt
    max_turn = motion["angular_speed_radps"] * dt
    position = np.asarray(points[0], dtype=np.float64)
    delta = points[1] - points[0]
    yaw = math.atan2(-delta[0], -delta[2])
    frames = [(position.copy(), yaw, 0.0)]
    for target in points[1:]:
        target = np.asarray(target, dtype=np.float64)
        delta = target - position
        desired = math.atan2(-delta[0], -delta[2])
        # 转折点先原地转向，再沿测地线段前进；不会因平滑转向而切入障碍物。
        angle = (desired - yaw + math.pi) % (2 * math.pi) - math.pi
        turns = int(math.ceil(abs(angle) / max_turn))
        for _ in range(turns):
            yaw += angle / turns
            frames.append((position.copy(), yaw, len(frames) * dt))
        remaining = float(np.linalg.norm(target - position))
        # 以小于速度上限的增量推进；发生投影时保留实际位置，不跨越障碍物强行吸附。
        while remaining > 0.03:
            delta = target - position
            distance = float(np.linalg.norm(delta))
            increment = min(max_step * 0.8, distance)
            wanted = position + delta * (increment / distance)
            actual = np.asarray(pathfinder.try_step(position.astype(np.float32), wanted.astype(np.float32)), dtype=np.float64)
            moved = float(np.linalg.norm(actual - position))
            if moved < 1e-4:
                raise RuntimeError("NavMesh 阻挡路径推进；此路线不适合当前代理宽度")
            if moved > max_step + 1e-6:
                raise RuntimeError("NavMesh 单步投影超过速度上限")
            if not pathfinder.is_navigable(actual.astype(np.float32), 0.15):
                raise RuntimeError("移动点离开 NavMesh")
            position = actual
            frames.append((position.copy(), yaw, len(frames) * dt))
            remaining = float(np.linalg.norm(target - position))
            if len(frames) > 200000:
                raise RuntimeError("路线推进超过安全帧数上限")
    return frames


