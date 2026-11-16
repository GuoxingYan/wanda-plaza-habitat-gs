"""生成曲率连续的导航运动原语、时间参数化轨迹及轨迹级语义标签。"""
import math

import numpy as np


ACTION_TYPES = ("arc_u_turn", "dead_end_reverse", "s_curve", "tangent_curve")
TAG_ORDER = (
    "in_place_rotation_minor", "in_place_rotation_moderate", "in_place_rotation_major",
    "curved_trajectory_minor", "curved_trajectory_moderate", "curved_trajectory_major",
    "straight_forward", "straight_backward", "elevation_change",
)


def assign_batch_actions(seed, batch_size, route_index, composition_mode="mixed",
                         composite_fraction=0.7):
    """根据批次索引确定性地均衡分配专题轨迹与复合动作。"""
    if batch_size < 1 or not 0 <= route_index < batch_size:
        raise ValueError("路线索引必须位于有效批次范围内")
    if composition_mode not in ("mixed", "single", "composite"):
        raise ValueError("generation.composition_mode 必须是 mixed、single 或 composite")
    if not 0.0 <= composite_fraction <= 1.0:
        raise ValueError("generation.composite_fraction 必须位于 [0, 1]")

    schedule_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(batch_size), 0xA11C]))
    if composition_mode == "single":
        composite_count = 0
    elif composition_mode == "composite":
        composite_count = batch_size
    elif batch_size == 1:
        composite_count = 1
    else:
        composite_count = int(math.floor(batch_size * composite_fraction + 0.5))
        composite_count = min(batch_size - 1, max(1, composite_count))

    route_order = schedule_rng.permutation(batch_size)
    composite_indices = set(int(value) for value in route_order[:composite_count])
    single_order = [int(value) for value in route_order[composite_count:]]
    single_types = [ACTION_TYPES[index % len(ACTION_TYPES)]
                    for index in range(len(single_order))]
    schedule_rng.shuffle(single_types)

    route_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(batch_size), int(route_index), 0xAC710]))
    if route_index in composite_indices:
        count = int(route_rng.integers(2, min(3, len(ACTION_TYPES)) + 1))
        actions = [str(value) for value in route_rng.choice(
            ACTION_TYPES, count, replace=False)]
        if "dead_end_reverse" in actions:
            actions.remove("dead_end_reverse")
            actions.append("dead_end_reverse")
        return {"composition": "composite", "actions": actions}
    single_index = single_order.index(route_index)
    return {"composition": "single", "actions": [single_types[single_index]]}


def _integrate_turn(start_xz, heading, signed_angle, radius, spacing):
    """以线性曲率渐变积分一段入口、恒曲率和出口连续的 Euler 螺线。"""
    angle = abs(float(signed_angle))
    if angle <= 1e-8:
        return np.asarray([start_xz], dtype=np.float64), float(heading)
    if radius <= 0 or spacing <= 0:
        raise ValueError("转弯半径和曲线采样间距必须为正数")

    sign = 1.0 if signed_angle > 0 else -1.0
    max_curvature = 1.0 / float(radius)
    transition_angle = min(angle / 2.0, math.pi / 4.0)
    transition_length = 2.0 * transition_angle / max_curvature
    constant_length = max(0.0, (angle - 2.0 * transition_angle) / max_curvature)
    total_length = 2.0 * transition_length + constant_length
    position = np.asarray(start_xz, dtype=np.float64).copy()
    points = [position.copy()]
    travelled = 0.0
    current_heading = float(heading)
    while travelled < total_length - 1e-9:
        step = min(float(spacing), total_length - travelled)
        middle = travelled + step / 2.0
        if middle < transition_length:
            curvature = sign * max_curvature * middle / transition_length
        elif middle < transition_length + constant_length:
            curvature = sign * max_curvature
        else:
            curvature = sign * max_curvature * (
                1.0 - (middle - transition_length - constant_length) / transition_length)
        next_heading = current_heading + curvature * step
        midpoint_heading = (current_heading + next_heading) / 2.0
        position += step * np.array([math.cos(midpoint_heading), math.sin(midpoint_heading)])
        points.append(position.copy())
        current_heading = next_heading
        travelled += step
    return np.asarray(points), current_heading


def generate_curve_action(start, yaw, action, radius_m, angle_rad, spacing_m=0.05):
    """生成掉头、S 弯或切向弧线的三维折线，起终点曲率均为零。"""
    start = np.asarray(start, dtype=np.float64)
    if start.shape != (3,) or not np.isfinite(start).all():
        raise ValueError("曲线起点必须是有限 XYZ 坐标")
    if action not in ("arc_u_turn", "s_curve", "tangent_curve"):
        raise ValueError(f"不支持的曲线动作: {action}")
    if radius_m <= 0 or angle_rad <= 0:
        raise ValueError("曲线半径和转角必须为正数")

    direction = np.array([-math.sin(float(yaw)), -math.cos(float(yaw))])
    heading = math.atan2(direction[1], direction[0])
    xz = start[[0, 2]].copy()
    if action == "s_curve":
        first, heading = _integrate_turn(xz, heading, angle_rad, radius_m, spacing_m)
        second, heading = _integrate_turn(first[-1], heading, -angle_rad, radius_m, spacing_m)
        xz = np.concatenate((first, second[1:]), axis=0)
    else:
        xz, heading = _integrate_turn(
            xz, heading, angle_rad, radius_m, spacing_m)
    points = np.repeat(start[None, :], len(xz), axis=0)
    points[:, 0], points[:, 2] = xz[:, 0], xz[:, 1]
    return points


def smooth_clothoid_polyline(points, minimum_radius_m, spacing_m=0.05,
                             minimum_corner_angle_rad=0.12):
    """以 clothoid 圆角平滑折线路径；空间不足或近 180 度折返时显式拒绝。"""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("折线路径必须至少包含两个 XYZ 点")
    if minimum_radius_m <= 0 or spacing_m <= 0:
        raise ValueError("最小转弯半径和采样间距必须为正数")

    clean = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point[[0, 2]] - clean[-1][[0, 2]]) > 1e-5:
            clean.append(point)
    points = np.asarray(clean, dtype=np.float64)
    if len(points) < 3:
        return points.copy()

    corners = []
    for index in range(1, len(points) - 1):
        incoming = points[index, [0, 2]] - points[index - 1, [0, 2]]
        outgoing = points[index + 1, [0, 2]] - points[index, [0, 2]]
        incoming_length, outgoing_length = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        incoming /= incoming_length
        outgoing /= outgoing_length
        angle = math.atan2(incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
                           float(np.dot(incoming, outgoing)))
        if abs(angle) < minimum_corner_angle_rad:
            corners.append(None)
            continue
        if abs(angle) > math.radians(165.0):
            raise RuntimeError("折线路径包含无法用前进曲线安全平滑的近 180 度转折")
        curve, _ = _integrate_turn(np.zeros(2), 0.0, angle,
                                   float(minimum_radius_m), float(spacing_m))
        local_displacement = curve[-1]
        incoming_heading = math.atan2(incoming[1], incoming[0])
        rotation = np.array([[math.cos(incoming_heading), -math.sin(incoming_heading)],
                             [math.sin(incoming_heading), math.cos(incoming_heading)]])
        displacement = rotation @ local_displacement
        matrix = np.column_stack((incoming, outgoing))
        try:
            trim_in, trim_out = np.linalg.solve(matrix, displacement)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("转角几何退化，无法构造 clothoid 过渡") from error
        if trim_in <= 0 or trim_out <= 0:
            raise RuntimeError("当前转角与曲率方向不匹配，无法构造前进圆角")
        corners.append((float(trim_in), float(trim_out), angle))

    for segment in range(len(points) - 1):
        consumed = 0.0
        if segment > 0 and corners[segment - 1] is not None:
            consumed += corners[segment - 1][1]
        if segment < len(corners) and corners[segment] is not None:
            consumed += corners[segment][0]
        available = float(np.linalg.norm(points[segment + 1, [0, 2]]
                                         - points[segment, [0, 2]]))
        if consumed > available - 2.0 * spacing_m:
            raise RuntimeError("相邻折点间距不足以满足最小转弯半径")

    result = [points[0].copy()]
    for index in range(1, len(points) - 1):
        corner = corners[index - 1]
        if corner is None:
            result.append(points[index].copy())
            continue
        trim_in, trim_out, angle = corner
        incoming_vector = points[index, [0, 2]] - points[index - 1, [0, 2]]
        incoming_length = float(np.linalg.norm(incoming_vector))
        incoming = incoming_vector / incoming_length
        start_xz = points[index, [0, 2]] - trim_in * incoming
        heading = math.atan2(incoming[1], incoming[0])
        curve, _ = _integrate_turn(start_xz, heading, angle,
                                   float(minimum_radius_m), float(spacing_m))
        start_fraction = (incoming_length - trim_in) / incoming_length
        start_y = points[index - 1, 1] + start_fraction * (
            points[index, 1] - points[index - 1, 1])
        outgoing_length = float(np.linalg.norm(
            points[index + 1, [0, 2]] - points[index, [0, 2]]))
        end_fraction = trim_out / outgoing_length
        end_y = points[index, 1] + end_fraction * (
            points[index + 1, 1] - points[index, 1])
        curve_3d = np.empty((len(curve), 3), dtype=np.float64)
        curve_3d[:, 0], curve_3d[:, 2] = curve[:, 0], curve[:, 1]
        curve_3d[:, 1] = np.linspace(start_y, end_y, len(curve))
        result.extend(curve_3d[1:])
    result.append(points[-1].copy())
    return np.asarray(result, dtype=np.float64)


def resample_polyline(points, spacing_m=0.05):
    """按近似等弧长间隔重采样三维折线，并保留精确首尾位置。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2 or spacing_m <= 0:
        raise ValueError("重采样需要至少两个点及正数间距")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > 1e-8))
    points = points[keep]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-8:
        raise ValueError("路径长度必须大于零")
    count = max(1, int(math.ceil(cumulative[-1] / spacing_m)))
    samples = np.linspace(0.0, cumulative[-1], count + 1)
    output = np.column_stack([
        np.interp(samples, cumulative, points[:, axis]) for axis in range(3)
    ])
    output[0], output[-1] = points[0], points[-1]
    return output


def _path_course(points):
    """由采样点计算连续展开的 XZ 行进方向角。"""
    delta = np.empty((len(points), 2), dtype=np.float64)
    delta[0] = points[1, [0, 2]] - points[0, [0, 2]]
    delta[-1] = points[-1, [0, 2]] - points[-2, [0, 2]]
    if len(points) > 2:
        delta[1:-1] = points[2:, [0, 2]] - points[:-2, [0, 2]]
    course = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    return course


def time_parameterize_path(points, config, reverse=False):
    """按速度、加减速度、角速度、横向加速度及底盘曲率限制生成固定帧率位姿。"""
    motion = config["motion"]
    spacing = float(motion.get("curve_sample_spacing_m", 0.05))
    points = resample_polyline(points, spacing)
    delta = np.diff(points, axis=0)
    distances = np.linalg.norm(delta, axis=1)
    course = _path_course(points)
    yaw = np.arctan2(-np.cos(course), -np.sin(course))
    if reverse:
        yaw += math.pi
    yaw = np.unwrap(yaw)
    yaw_delta = np.diff(yaw)
    curvature = np.zeros(len(points), dtype=np.float64)
    curvature[1:-1] = (yaw_delta[1:] + yaw_delta[:-1]) / (
        distances[1:] + distances[:-1] + 1e-12)
    curvature[0], curvature[-1] = curvature[1], curvature[-2]

    max_speed = float(motion["linear_speed_mps"])
    max_angular = float(motion["angular_speed_radps"])
    lateral_acceleration = float(motion.get("max_lateral_acceleration_mps2", 1.0))
    minimum_radius = float(motion.get("minimum_turn_radius_m", 1.0))
    model = motion.get("model", "differential_drive")
    limits = np.full(len(points), max_speed, dtype=np.float64)
    nonzero_curvature = np.abs(curvature) > 1e-8
    limits[nonzero_curvature] = np.minimum(
        limits[nonzero_curvature],
        max_angular / np.abs(curvature[nonzero_curvature]))
    limits[nonzero_curvature] = np.minimum(
        limits[nonzero_curvature],
        np.sqrt(lateral_acceleration / np.abs(curvature[nonzero_curvature])))
    if model == "ackermann":
        if np.any(np.abs(curvature) > 1.0 / minimum_radius + 0.04):
            raise RuntimeError("轨迹曲率超过 Ackermann 最小转弯半径限制")
    elif model != "differential_drive":
        raise ValueError("motion.model 必须是 differential_drive 或 ackermann")

    speeds = limits.copy()
    speeds[0] = speeds[-1] = 0.0
    acceleration = float(motion["max_acceleration_mps2"])
    deceleration = float(motion["max_deceleration_mps2"])
    for index in range(1, len(speeds)):
        speeds[index] = min(speeds[index], math.sqrt(
            speeds[index - 1] ** 2 + 2.0 * acceleration * distances[index - 1]))
    for index in range(len(speeds) - 2, -1, -1):
        speeds[index] = min(speeds[index], math.sqrt(
            speeds[index + 1] ** 2 + 2.0 * deceleration * distances[index]))
    interval_time = 2.0 * distances / np.maximum(speeds[:-1] + speeds[1:], 1e-9)
    cumulative_time = np.concatenate(([0.0], np.cumsum(interval_time)))
    total_time = float(cumulative_time[-1])
    dt = 1.0 / float(motion["fps"])

    def sample(scale):
        """按给定总时长比例采样，并让末帧严格落在轨迹终点。"""
        frame_count = max(1, int(math.ceil(total_time * scale / dt)))
        output_duration = frame_count * dt
        source_time = cumulative_time * (output_duration / total_time)
        sample_times = np.arange(frame_count + 1, dtype=np.float64) * dt
        sampled_points = np.column_stack([
            np.interp(sample_times, source_time, points[:, axis])
            for axis in range(3)
        ])
        sampled_yaw = np.interp(sample_times, source_time, yaw)
        sampled_points[-1], sampled_yaw[-1] = points[-1], yaw[-1]
        return [(sampled_points[index], float(sampled_yaw[index]), float(index * dt))
                for index in range(frame_count + 1)]

    def discrete_limits(frames):
        """计算固定帧率输出真正呈现的速度、角速度和加减速度峰值。"""
        positions = np.asarray([frame[0] for frame in frames], dtype=np.float64)
        frame_yaw = np.unwrap(np.asarray([frame[1] for frame in frames], dtype=np.float64))
        frame_speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
        frame_angular = np.abs(np.diff(frame_yaw)) / dt
        frame_acceleration = np.diff(frame_speed) / dt
        return (float(frame_speed.max(initial=0.0)),
                float(frame_angular.max(initial=0.0)),
                float(frame_acceleration.max(initial=0.0)),
                float((-frame_acceleration).max(initial=0.0)))

    # 几何曲率约束还会受到固定 FPS 采样误差影响；逐帧复核并只延长总时长，
    # 确保最终落盘的离散轨迹也满足配置上限。
    scale = 1.0
    for _ in range(32):
        frames = sample(scale)
        frame_speed, frame_angular, frame_acceleration, frame_deceleration = discrete_limits(frames)
        ratios = [
            frame_speed / max_speed,
            frame_angular / max_angular,
            math.sqrt(frame_acceleration / acceleration) if frame_acceleration > 0 else 0.0,
            math.sqrt(frame_deceleration / deceleration) if frame_deceleration > 0 else 0.0,
        ]
        required_scale = max(ratios)
        if required_scale <= 1.0 + 1e-8:
            return frames
        scale *= max(1.02, required_scale * 1.01)
    raise RuntimeError("固定帧率重采样后仍无法满足速度、角速度或加减速度约束")


def classify_trajectory_tags(frames, motion_events=None, elevation_threshold_m=0.15):
    """按现有标注规范从实际执行位姿汇总整条轨迹的 Tag 集合。"""
    if len(frames) < 2:
        return []
    positions = np.asarray([frame[0] for frame in frames], dtype=np.float64)
    yaws = np.unwrap(np.asarray([frame[1] for frame in frames], dtype=np.float64))
    displacement = np.diff(positions, axis=0)
    distance = np.linalg.norm(displacement[:, [0, 2]], axis=1)
    moving = distance > 1e-4
    tags = set()

    if np.ptp(positions[:, 1]) >= float(elevation_threshold_m):
        tags.add("elevation_change")

    forward = np.array([-np.sin(yaws[:-1]), -np.cos(yaws[:-1])]).T
    signed_distance = np.sum(displacement[:, [0, 2]] * forward, axis=1)
    for is_reverse, tag in ((False, "straight_forward"),
                            (True, "straight_backward")):
        selected = moving & ((signed_distance < -1e-5) if is_reverse
                             else (signed_distance > 1e-5))
        run_distance = 0.0
        run_yaws = []
        for index, active in enumerate(selected):
            if active:
                run_distance += float(distance[index])
                run_yaws.extend((float(yaws[index]), float(yaws[index + 1])))
            else:
                if run_distance >= 2.0 and run_yaws and np.ptp(run_yaws) < math.radians(5):
                    tags.add(tag)
                run_distance, run_yaws = 0.0, []
        if run_distance >= 2.0 and run_yaws and np.ptp(run_yaws) < math.radians(5):
            tags.add(tag)

    for event in motion_events or []:
        if event.get("type") not in ("arc_u_turn", "tangent_curve", "s_curve"):
            continue
        start = max(0, min(len(yaws) - 1, int(event.get("start_frame", 0))))
        end = max(start + 1, min(len(yaws) - 1, int(event.get("end_frame", len(yaws) - 1))))
        net_angle = abs(float(yaws[end] - yaws[start]))
        if net_angle < math.radians(15):
            continue
        if net_angle < math.radians(45):
            tags.add("curved_trajectory_minor")
        elif net_angle < math.radians(90):
            tags.add("curved_trajectory_moderate")
        else:
            tags.add("curved_trajectory_major")

    stationary = np.linalg.norm(displacement, axis=1) <= 1e-4
    rotation_start = None
    for index, is_stationary in enumerate(stationary):
        if is_stationary and rotation_start is None:
            rotation_start = index
        if (not is_stationary or index == len(stationary) - 1) and rotation_start is not None:
            rotation_end = index if not is_stationary else index + 1
            rotation = abs(float(yaws[min(rotation_end, len(yaws) - 1)]
                                - yaws[rotation_start]))
            if rotation >= math.radians(90):
                tags.add("in_place_rotation_major")
            elif rotation >= math.radians(45):
                tags.add("in_place_rotation_moderate")
            elif rotation >= math.radians(15):
                tags.add("in_place_rotation_minor")
            rotation_start = None
    return [tag for tag in TAG_ORDER if tag in tags]
