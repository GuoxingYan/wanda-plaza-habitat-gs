"""不依赖 Habitat 的离散差速/全向运动学速度规划。"""
import math

import numpy as np


def yaw_steps(start_yaw, target_yaw, max_angular_speed, dt):
    """生成绕 Y 轴的原地转向帧，保证相邻 yaw 变化不超角速度上限。"""
    if max_angular_speed <= 0 or dt <= 0:
        raise ValueError("角速度和时间步长都必须为正数")
    angle = (target_yaw - start_yaw + math.pi) % (2 * math.pi) - math.pi
    count = int(math.ceil(abs(angle) / (max_angular_speed * dt)))
    if count == 0:
        return []
    return [start_yaw + angle * index / count for index in range(1, count + 1)]


def trapezoidal_speed_profile(distance, max_speed, acceleration, deceleration, dt):
    """生成固定帧率、两端静止的梯形/三角速度剖面。"""
    if distance <= 0 or min(max_speed, acceleration, deceleration, dt) <= 0:
        raise ValueError("距离、速度、加减速度和时间步长都必须为正数")

    peak = min(max_speed, math.sqrt(
        2 * distance * acceleration * deceleration / (acceleration + deceleration)))
    accel_time = peak / acceleration
    brake_time = peak / deceleration
    cruise_time = max(0.0, (distance - peak**2 / (2 * acceleration)
                            - peak**2 / (2 * deceleration)) / peak)
    total_time = accel_time + cruise_time + brake_time
    frame_count = max(2, math.ceil(total_time / dt))

    # 固定帧数下逐步构造速度上限，再用前后向传播满足离散加减速约束。
    speeds = np.full(frame_count + 1, peak, dtype=np.float64)
    speeds[0] = speeds[-1] = 0.0
    for index in range(1, frame_count + 1):
        speeds[index] = min(speeds[index], speeds[index - 1] + acceleration * dt)
    for index in range(frame_count - 1, -1, -1):
        speeds[index] = min(speeds[index], speeds[index + 1] + deceleration * dt)

    # 按梯形积分位移做小幅归一化；若缩放会超出约束，则增加帧数重算。
    for _ in range(1000):
        intervals = 0.5 * (speeds[:-1] + speeds[1:]) * dt
        traveled = float(intervals.sum())
        scale = distance / traveled
        candidate = speeds * scale
        if (candidate.max() <= max_speed + 1e-9
                and np.all(np.diff(candidate) <= acceleration * dt + 1e-9)
                and np.all(-np.diff(candidate) <= deceleration * dt + 1e-9)):
            candidate_intervals = (candidate[:-1] + candidate[1:]) * 0.5 * dt
            if candidate_intervals.max() <= max_speed * dt + 1e-9:
                # 返回每帧对应的平均线速度；首尾零速端点只用于约束曲线边界。
                return candidate_intervals / dt
        frame_count += 1
        speeds = np.full(frame_count + 1, peak, dtype=np.float64)
        speeds[0] = speeds[-1] = 0.0
        for index in range(1, frame_count + 1):
            speeds[index] = min(speeds[index], speeds[index - 1] + acceleration * dt)
        for index in range(frame_count - 1, -1, -1):
            speeds[index] = min(speeds[index], speeds[index + 1] + deceleration * dt)
    raise RuntimeError("无法在帧数上限内构造满足约束的速度剖面")


def build_linear_segment(start, target, start_yaw, start_time, fps,
                         max_speed, acceleration, deceleration, try_step, is_navigable):
    """通过 NavMesh 步进器执行一段直线路径，返回帧记录与末端状态。"""
    dt = 1.0 / fps
    position = np.asarray(start, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64)
    delta = target - position
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-8:
        return [], position, start_time
    direction = delta / distance
    speeds = trapezoidal_speed_profile(distance, max_speed, acceleration, deceleration, dt)
    frames = []
    elapsed = float(start_time)
    for speed in speeds:
        remaining = float(np.linalg.norm(target - position))
        step = min(float(speed * dt), max_speed * dt, remaining)
        wanted = position + direction * step
        actual = np.asarray(try_step(position.astype(np.float32), wanted.astype(np.float32)), dtype=np.float64)
        moved = float(np.linalg.norm(actual - position))
        if moved < 1e-5 and remaining > 1e-5:
            raise RuntimeError("NavMesh 阻挡路径推进；此路线不适合当前代理宽度")
        if moved > max_speed * dt + 1e-6:
            raise RuntimeError(f"NavMesh 单步投影超过速度上限: {moved:.6f} > {max_speed * dt:.6f}")
        if not is_navigable(actual.astype(np.float32), 0.15):
            raise RuntimeError("移动点离开 NavMesh")
        position = actual
        elapsed += dt
        frames.append((position.copy(), float(start_yaw), elapsed))
    if np.linalg.norm(position - target) > 0.03:
        raise RuntimeError("NavMesh 投影导致轨迹无法到达路径转折点")
    return frames, position, elapsed
