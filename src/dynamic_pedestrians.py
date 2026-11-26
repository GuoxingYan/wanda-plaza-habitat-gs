"""中文文件头：提供动态行人代理胶囊碰撞检测和 driver 数据校验。"""

import pickle
from pathlib import Path

import numpy as np


def minimum_capsule_clearance(first, second):
    """计算两组逐帧代理胶囊的最小表面净距，负值表示发生碰撞。"""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.ndim != 3 or second.ndim != 3 or first.shape[0] != second.shape[0]:
        raise ValueError("capsules 必须为帧数一致的 [T,C,7] 数组")
    if first.shape[2] != 7 or second.shape[2] != 7:
        raise ValueError("capsule 最后一维必须是两个端点与半径 [7]")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("capsule 数据必须全为有限值")
    if np.any(first[..., 6] < 0) or np.any(second[..., 6] < 0):
        raise ValueError("capsule 半径不能为负值")

    a0, a1, ar = first[..., :3], first[..., 3:6], first[..., 6]
    b0, b1, br = second[..., :3], second[..., 3:6], second[..., 6]
    minimum = np.inf
    for frame in range(first.shape[0]):
        p0, u = a0[frame], a1[frame] - a0[frame]
        q0, v = b0[frame], b1[frame] - b0[frame]
        w = p0[:, None, :] - q0[None, :, :]
        aa = np.einsum("ij,ij->i", u, u)[:, None]
        cc = np.einsum("ij,ij->i", v, v)[None, :]
        bb = u @ v.T
        dd = np.einsum("ijk,ik->ij", w, u)
        ee = np.einsum("ijk,jk->ij", w, v)

        def point_segment_distance(point, start, direction, length_squared):
            """计算逐点至线段的距离，并令零长度线段退化为点。"""
            offset = point - start
            projection = np.sum(offset * direction, axis=-1)
            parameter = np.clip(projection / np.maximum(length_squared, 1e-12), 0.0, 1.0)
            delta = offset - parameter[..., None] * direction
            return np.linalg.norm(delta, axis=-1)

        candidates = [
            point_segment_distance(w, np.zeros_like(q0)[None, :, :], v[None, :, :], cc),
            point_segment_distance(w + u[:, None, :], np.zeros_like(q0)[None, :, :], v[None, :, :], cc),
            point_segment_distance(-w, np.zeros_like(p0)[:, None, :], u[:, None, :], aa),
            point_segment_distance(-w + v[None, :, :], np.zeros_like(p0)[:, None, :], u[:, None, :], aa),
        ]
        denominator = aa * cc - bb * bb
        safe = np.maximum(denominator, 1e-12)
        s = (bb * ee - cc * dd) / safe
        t = (aa * ee - bb * dd) / safe
        interior = (denominator > 1e-12) & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
        interior_delta = w + s[..., None] * u[:, None, :] - t[..., None] * v[None, :, :]
        interior_distance = np.where(interior, np.linalg.norm(interior_delta, axis=-1), np.inf)
        center_distance = np.minimum.reduce([*candidates, interior_distance])
        surface = center_distance - ar[frame, :, None] - br[frame, None, :]
        minimum = min(minimum, float(surface.min()))
        if minimum < 0:
            return minimum
    return float(minimum)


def load_driver(path):
    """读取 driver，并检查渲染矩阵和碰撞胶囊按相同帧对齐。"""
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    mats = np.asarray(data["joint_mats"])
    capsules = np.asarray(data["proxy_capsules"])
    if mats.shape != (len(capsules), 55, 4, 4) or capsules.shape[1:] != (54, 7):
        raise ValueError(f"driver 形状不符：{mats.shape}, {capsules.shape}")
    if not np.isfinite(mats).all() or not np.isfinite(capsules).all():
        raise ValueError("driver 含非有限值")
    if not np.allclose(mats[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("joint_mats 齐次矩阵格式异常")
    return data
