"""用于验证静态障碍触发绕行的轻量 NavMesh 栅格规划。"""
import heapq
import math
from functools import lru_cache

import numpy as np
import habitat_sim as hs
from PIL import Image, ImageDraw


def _cell(point, origin_xz, resolution):
    return tuple(np.rint((np.asarray(point)[[0, 2]] - origin_xz) / resolution).astype(int))


def _world(cell, origin_xz, y, resolution):
    xz = origin_xz + np.asarray(cell, dtype=np.float64) * resolution
    return np.array([xz[0], y, xz[1]], dtype=np.float64)


def _astar(pathfinder, start, goal, obstacle_center, blocked_radius, resolution,
           max_expansions, corridor_points, corridor_width):
    """在 NavMesh 可行单元上搜索八邻接路径，障碍圆盘连代理净距一起膨胀。"""
    bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float64)
    origin = bounds[0, [0, 2]]
    shape = np.ceil((bounds[1, [0, 2]] - origin) / resolution).astype(int) + 1
    start_cell, goal_cell = _cell(start, origin, resolution), _cell(goal, origin, resolution)
    obstacle_xz = np.asarray(obstacle_center)[[0, 2]]
    corridor = np.asarray(corridor_points, dtype=np.float64)[:, [0, 2]]
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    @lru_cache(maxsize=250000)
    def usable(cell):
        if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
            return False
        point = _world(cell, origin, float(start[1]), resolution)
        # 只在原路线周围有限宽度的走廊中绕行，避免搜索整片场景并走出采集区域。
        nearest_route_distance = math.inf
        for first, second in zip(corridor[:-1], corridor[1:]):
            delta = second - first
            fraction = np.clip(np.dot(point[[0, 2]] - first, delta)
                               / max(float(np.dot(delta, delta)), 1e-12), 0.0, 1.0)
            nearest_route_distance = min(nearest_route_distance, float(np.linalg.norm(
                point[[0, 2]] - (first + fraction * delta))))
        if nearest_route_distance > corridor_width:
            return False
        if np.linalg.norm(point[[0, 2]] - obstacle_xz) <= blocked_radius:
            return False
        snapped = np.asarray(pathfinder.snap_point(point.astype(np.float32)), dtype=np.float64)
        return (np.linalg.norm(snapped[[0, 2]] - point[[0, 2]]) <= resolution * 1.5
                and pathfinder.is_navigable(snapped.astype(np.float32), 0.15))

    # 起点/终点贴近网格边界时，吸附到附近最近的可行网格单元。
    def nearest(cell):
        candidates = []
        radius = max(1, int(math.ceil(1.0 / resolution)))
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                candidate = (cell[0] + dx, cell[1] + dz)
                if usable(candidate):
                    candidates.append((dx * dx + dz * dz, candidate))
        if not candidates:
            return None
        return min(candidates)[1]

    start_cell, goal_cell = nearest(start_cell), nearest(goal_cell)
    if start_cell is None or goal_cell is None:
        return None
    frontier = [(0.0, start_cell)]
    came_from = {start_cell: None}
    costs = {start_cell: 0.0}
    expansions = 0
    while frontier and expansions < max_expansions:
        _, current = heapq.heappop(frontier)
        if current == goal_cell:
            cells = [current]
            while came_from[cells[-1]] is not None:
                cells.append(came_from[cells[-1]])
            cells.reverse()
            points = [np.asarray(pathfinder.snap_point(
                _world(cell, origin, float(start[1]), resolution).astype(np.float32)),
                dtype=np.float64) for cell in cells]
            points[0] = np.asarray(start, dtype=np.float64)
            points[-1] = np.asarray(goal, dtype=np.float64)
            # 只压缩同方向共线网格边，绝不调用会穿越临时障碍的 NavMesh 直线捷径。
            points = np.asarray(points, dtype=np.float32)
            simplified = [points[0]]
            previous_direction = None
            for first, second in zip(points, points[1:]):
                direction = np.sign((second - first)[[0, 2]]).astype(int)
                if previous_direction is not None and not np.array_equal(
                        direction, previous_direction):
                    simplified.append(first)
                previous_direction = direction
            simplified.append(points[-1])
            simplified[0] = np.asarray(start, dtype=np.float32)
            simplified[-1] = np.asarray(goal, dtype=np.float32)
            return np.asarray(simplified, dtype=np.float32)
        expansions += 1
        for dx, dz in neighbors:
            neighbor = (current[0] + dx, current[1] + dz)
            if not usable(neighbor):
                continue
            step = resolution * (math.sqrt(2.0) if dx and dz else 1.0)
            new_cost = costs[current] + step
            if new_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = new_cost
            came_from[neighbor] = current
            heuristic = resolution * math.hypot(goal_cell[0] - neighbor[0],
                                                goal_cell[1] - neighbor[1])
            heapq.heappush(frontier, (new_cost + heuristic, neighbor))
    return None


def plan_around_synthetic_obstacle(pathfinder, original_points, config):
    """在原路线中段放置圆柱 footprint；触发距离内用 A* 生成绕行折线。"""
    settings = config.get("avoidance", {})
    if not settings.get("enabled", False):
        return np.asarray(original_points, dtype=np.float32), None
    points = np.asarray(original_points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 0:
        raise RuntimeError("原始路线长度为零，无法放置避障物")
    target_distance = total * float(settings.get("placement_fraction", 0.5))
    segment = min(int(np.searchsorted(cumulative, target_distance, side="right") - 1),
                  len(points) - 2)
    ratio = (target_distance - cumulative[segment]) / max(segment_lengths[segment], 1e-9)
    center = points[segment] + ratio * (points[segment + 1] - points[segment])
    radius = float(settings["obstacle_radius_m"])
    height = float(settings["obstacle_height_m"])
    trigger = float(settings["trigger_distance_m"])
    if height <= 0.5 or radius <= 0:
        raise ValueError("避障物高度必须大于 0.5 m，半径必须为正数")
    # 路线与障碍中心位于同一条原始折线上，固定居中时必然触发；保留距离阈值
    # 用于运行时逻辑和报告，也可在 YAML 中关闭该条件进行强制测试。
    center_xz = center[[0, 2]]
    route_distance = math.inf
    for first, second in zip(points[:-1], points[1:]):
        start_xz, delta_xz = first[[0, 2]], second[[0, 2]] - first[[0, 2]]
        fraction = np.clip(np.dot(center_xz - start_xz, delta_xz)
                           / max(float(np.dot(delta_xz, delta_xz)), 1e-12), 0.0, 1.0)
        route_distance = min(route_distance, float(np.linalg.norm(
            center_xz - (start_xz + fraction * delta_xz))))
    triggered = bool(settings.get("force", False) or route_distance <= trigger + radius)
    descriptor = {"type": "procedural_cylinder", "center": center.tolist(),
                  "radius_m": radius, "height_m": height,
                  "trigger_distance_m": trigger, "triggered": triggered,
                  "placement_fraction": float(settings.get("placement_fraction", 0.5))}
    if not triggered:
        return points.astype(np.float32), descriptor
    safe_radius = radius + float(config["navmesh"]["agent_width_m"]) / 2.0 + float(
        settings.get("safety_margin_m", 0.2))
    # 栅格 A* 允许对角移动；加入半个单元对角线裕量，防止线段穿过被占用圆盘的角落。
    route = _astar(pathfinder, points[0], points[-1], center,
                   safe_radius + float(settings["grid_resolution_m"]) * math.sqrt(2.0),
                   float(settings["grid_resolution_m"]),
                   int(settings["max_search_cells"]), points,
                   float(settings.get("corridor_width_m", 4.0)))
    if route is None:
        raise RuntimeError("临时障碍物附近没有找到满足净距的 NavMesh 绕行路线")
    descriptor["planning_clearance_radius_m"] = safe_radius
    descriptor["detour_waypoints"] = len(route)
    return route, descriptor


def add_render_obstacle(sim, obstacle):
    """在 Habitat 场景中显示与规划 footprint 对应的程序化圆柱。"""
    if not obstacle or not obstacle.get("triggered"):
        return None
    import habitat_sim

    import tempfile

    # 用 NumPy 写一个低面数圆柱 OBJ，避免引入额外几何库或大型外部模型。
    sections = 24
    radius, height = float(obstacle["radius_m"]), float(obstacle["height_m"])
    vertices = []
    for y in (-height / 2.0, height / 2.0):
        vertices.extend((radius * math.cos(2 * math.pi * index / sections), y,
                         radius * math.sin(2 * math.pi * index / sections))
                        for index in range(sections))
    faces = []
    for index in range(sections):
        nxt = (index + 1) % sections
        lower_a, lower_b = index + 1, nxt + 1
        upper_a, upper_b = sections + index + 1, sections + nxt + 1
        faces.extend(((lower_a, lower_b, upper_b), (lower_a, upper_b, upper_a),
                      (1, lower_b, lower_a), (sections + 1, upper_a, upper_b)))
    temporary_mesh = tempfile.NamedTemporaryFile(prefix="wanda_nav_obstacle_",
                                                 suffix=".obj", delete=False)
    temporary_mesh.close()
    with open(temporary_mesh.name, "w", encoding="ascii") as stream:
        stream.write("o temporary_navigation_obstacle\n")
        for vertex in vertices:
            stream.write("v %.8f %.8f %.8f\n" % vertex)
        for face in faces:
            stream.write("f %d %d %d\n" % face)
    object_template = habitat_sim.attributes.ObjectAttributes()
    object_template.handle = "wanda_nav_temporary_obstacle"
    object_template.render_asset_handle = temporary_mesh.name
    object_template.collision_asset_handle = temporary_mesh.name
    object_template.scale = np.ones(3, dtype=np.float32)
    object_template.is_collidable = False
    object_template.is_visibile = True
    template_id = sim.get_object_template_manager().register_template(object_template)
    obj = sim.get_rigid_object_manager().add_object_by_template_id(template_id)
    center = np.asarray(obstacle["center"], dtype=np.float32)
    center[1] += height / 2.0
    obj.translation = center
    import habitat_sim
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    return obj, temporary_mesh.name


def composite_obstacle_rgb(rgb, obstacle, camera_position, yaw, camera_config):
    """将临时圆柱按当前 D435 针孔位姿投影到 Gaussian RGB 帧。"""
    if not obstacle or not obstacle.get("triggered"):
        return rgb
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8).copy())
    draw = ImageDraw.Draw(image)
    center = np.asarray(obstacle["center"], dtype=np.float64)
    camera = np.asarray(camera_position, dtype=np.float64)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    forward = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    width, height = int(camera_config["width"]), int(camera_config["height"])
    focal = width / (2.0 * math.tan(math.radians(float(camera_config["hfov_deg"])) / 2.0))
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0

    def project(point):
        delta = np.asarray(point, dtype=np.float64) - camera
        depth = float(np.dot(delta, forward))
        if depth <= 0.05:
            return None, depth
        px = cx + focal * float(np.dot(delta, right)) / depth
        py = cy - focal * float(np.dot(delta, up)) / depth
        return (int(round(px)), int(round(py))), depth

    radius, obstacle_height = float(obstacle["radius_m"]), float(obstacle["height_m"])
    camera_to_center = camera[[0, 2]] - center[[0, 2]]
    if float(np.linalg.norm(camera_to_center)) < radius * 0.25:
        camera_to_center = np.array([1.0, 0.0])
    camera_to_center /= np.linalg.norm(camera_to_center)
    camera_angle = math.atan2(camera_to_center[1], camera_to_center[0])
    sections = 32
    faces = []
    for index in range(sections):
        angle = 2.0 * math.pi * (index + 0.5) / sections
        normal = np.array([math.cos(angle), math.sin(angle)])
        if np.dot(normal, camera_to_center) <= 0:
            continue
        a0, a1 = 2.0 * math.pi * index / sections, 2.0 * math.pi * (index + 1) / sections
        world = [
            center + np.array([radius * math.cos(a0), 0, radius * math.sin(a0)]),
            center + np.array([radius * math.cos(a1), 0, radius * math.sin(a1)]),
            center + np.array([radius * math.cos(a1), obstacle_height, radius * math.sin(a1)]),
            center + np.array([radius * math.cos(a0), obstacle_height, radius * math.sin(a0)]),
        ]
        projected = [project(point) for point in world]
        if any(pixel is None for pixel, _ in projected):
            continue
        shade = 0.72 + 0.28 * max(0.0, math.cos(angle - camera_angle))
        color = (int(220 * shade), int(48 * shade), int(34 * shade))
        mean_depth = sum(depth for _, depth in projected) / 4.0
        faces.append((mean_depth, [pixel for pixel, _ in projected], color))
    if not faces:
        return rgb
    # 先画远侧面，再画近侧面，构成有宽度与高度的实体，而非路线图上的圆形标记。
    for _, polygon, color in sorted(faces, reverse=True):
        draw.polygon(polygon, fill=color, outline=(76, 24, 20))
    top_ring = []
    for index in range(sections):
        angle = 2.0 * math.pi * index / sections
        pixel, _ = project(center + np.array([radius * math.cos(angle),
                                               obstacle_height,
                                               radius * math.sin(angle)]))
        if pixel is not None:
            top_ring.append(pixel)
    if len(top_ring) >= 3:
        draw.polygon(top_ring, fill=(244, 90, 57), outline=(76, 24, 20))
    return np.asarray(image)
