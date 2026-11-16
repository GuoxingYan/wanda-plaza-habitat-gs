"""曲线运动原语、运动学约束、批次动作排程与语义标签回归测试。"""
import math
import unittest
from unittest.mock import patch

import numpy as np

from src.motion_primitives import (ACTION_TYPES, assign_batch_actions,
                                   classify_trajectory_tags, generate_curve_action,
                                   smooth_clothoid_polyline, time_parameterize_path)
from src.planning import (_build_operation_frames, _maximum_discrete_curvature,
                          _connect_segments, _merge_forward_operations,
                          _path_endpoint_tangent, _path_length,
                          _try_tangent_connector)


class MotionPrimitivesTest(unittest.TestCase):
    """验证丰富运动生成器的几何连续性和输出约束。"""

    class OpenNavMesh:
        """提供可控的平面可行区，用于隔离测试切向连接几何。"""

        def snap_point(self, point):
            """在测试平面中将输入点原样视为 NavMesh 投影结果。"""
            return np.asarray(point, dtype=np.float64)

        def get_island(self, point):
            """将测试中的所有点视为同一连通岛。"""
            return 0

        def is_navigable(self, point, margin):
            """允许测试候选连接经过的所有点。"""
            return True

    def test_batch_schedule_is_reproducible_and_balanced(self):
        """验证默认批次约七成复合样本且专题动作类别均衡。"""
        plans = [assign_batch_actions(17, 50, index) for index in range(50)]
        self.assertEqual(plans, [assign_batch_actions(17, 50, index) for index in range(50)])
        composite = sum(plan["composition"] == "composite" for plan in plans)
        self.assertEqual(composite, 35)
        singles = [plan["actions"][0] for plan in plans if plan["composition"] == "single"]
        self.assertEqual(set(singles), set(ACTION_TYPES))
        self.assertLessEqual(max(singles.count(action) for action in ACTION_TYPES)
                             - min(singles.count(action) for action in ACTION_TYPES), 1)

    def test_s_curve_returns_to_initial_heading_with_curvature_reversal(self):
        """验证 S 弯前后切向一致且曲率确实改变符号。"""
        points = generate_curve_action([0, 0, 0], 0.0, "s_curve", 4.0,
                                       math.radians(55), 0.025)
        course = np.unwrap(np.arctan2(np.diff(points[:, 2]), np.diff(points[:, 0])))
        course_change = np.diff(course)
        self.assertGreater(np.count_nonzero(course_change > 1e-5), 10)
        self.assertGreater(np.count_nonzero(course_change < -1e-5), 10)
        self.assertLess(abs(float(course[-1] - course[0])), math.radians(2.0))

    def test_u_turn_has_large_heading_change_and_finite_geometry(self):
        """验证弧线掉头形成有限轨迹并实现近 180 度航向变化。"""
        points = generate_curve_action([1, 2, 3], 0.3, "arc_u_turn", 3.0,
                                       math.pi, 0.025)
        course = np.unwrap(np.arctan2(np.diff(points[:, 2]), np.diff(points[:, 0])))
        self.assertTrue(np.isfinite(points).all())
        self.assertAlmostEqual(abs(float(course[-1] - course[0])), math.pi, delta=0.02)

    def test_clothoid_smoothing_respects_radius_and_heading_continuity(self):
        """验证直角折线路径被圆滑且离散曲率不超过指定上限。"""
        points = smooth_clothoid_polyline(
            [[0, 0, 0], [10, 0, 0], [10, 0, -10]], 2.0, 0.02)
        delta = np.diff(points[:, [0, 2]], axis=0)
        courses = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        ds = np.linalg.norm(delta, axis=1)
        curvature = np.abs(np.diff(courses)) / ((ds[:-1] + ds[1:]) / 2)
        self.assertGreater(len(points), 4)
        self.assertLessEqual(float(curvature.max()), 1.0 / 2.0 + 0.08)
        self.assertTrue(np.allclose(points[0], [0, 0, 0]))
        self.assertTrue(np.allclose(points[-1], [10, 0, -10]))

    def test_tangent_connector_matches_both_endpoints_and_radius(self):
        """验证可行 NavMesh 连接能匹配两端切线并满足最小转弯半径。"""
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([10.0, 0.0, 10.0])
        start_stub = start + np.array([0.75, 0.0, 0.0])
        end_stub = end - np.array([0.0, 0.0, 0.75])
        config = {"motion": {"curve_sample_spacing_m": 0.05,
                             "minimum_turn_radius_m": 0.3},
                  "navigation": {"camera_trajectory_buffer_m": 0.0}}
        with patch("src.planning._shortest_path",
                   return_value=np.asarray([start_stub, end_stub])):
            connector = _try_tangent_connector(
                self.OpenNavMesh(), start, end, np.array([1.0, 0.0]),
                np.array([0.0, 1.0]), 0, None, config)

        self.assertIsNotNone(connector)
        start_tangent = _path_endpoint_tangent(connector, at_end=False)
        end_tangent = _path_endpoint_tangent(connector, at_end=True)
        self.assertGreater(float(np.dot(start_tangent, [1.0, 0.0])), 0.99)
        self.assertGreater(float(np.dot(end_tangent, [0.0, 1.0])), 0.99)
        self.assertLessEqual(_maximum_discrete_curvature(connector),
                             1.0 / 0.3 + 0.08)

    def test_merged_curve_and_tangent_extension_keep_moving_at_joint(self):
        """验证弧线接直线后共用速度规划，不在接口人为停车。"""
        curve = generate_curve_action([0, 0, 0], 0.0, "tangent_curve", 2.0,
                                      math.radians(50), 0.025)
        tangent = _path_endpoint_tangent(curve, at_end=True)
        extension = np.repeat(curve[-1][None, :], 21, axis=0)
        extension[:, 0] += np.linspace(0.0, tangent[0] * 5.0, len(extension))
        extension[:, 2] += np.linspace(0.0, tangent[1] * 5.0, len(extension))
        event_distance = _path_length(curve)
        operation = _merge_forward_operations(
            {"points": curve, "reverse": False,
             "events": [{"type": "tangent_curve", "start_distance_m": 0.0,
                         "end_distance_m": event_distance}]},
            extension)
        config = {"motion": {
            "model": "differential_drive", "minimum_turn_radius_m": 0.3,
            "linear_speed_mps": 3.0, "angular_speed_radps": 1.2,
            "max_lateral_acceleration_mps2": 1.0,
            "max_acceleration_mps2": 1.0, "max_deceleration_mps2": 1.5,
            "curve_sample_spacing_m": 0.05, "fps": 10,
        }}

        _, frames, events = _build_operation_frames([operation], config)
        self.assertEqual(len(events), 1)
        positions = np.asarray([frame[0] for frame in frames])
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 10.0
        event_end = events[0]["end_frame"]
        self.assertGreater(float(speeds[max(0, event_end - 1)]), 0.05)
        self.assertGreater(float(speeds[min(len(speeds) - 1, event_end)]), 0.05)

    def test_segment_connection_uses_tangent_merge_when_available(self):
        """验证规划器采纳可行切向连接并合并两侧动作事件。"""
        previous_points = np.asarray([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        following_points = np.asarray([[10.0, 0.0, 10.0],
                                       [10.0, 0.0, 15.0]])
        start_stub = np.asarray([0.75, 0.0, 0.0])
        end_stub = np.asarray([10.0, 0.0, 9.25])
        config = {"motion": {
            "model": "differential_drive", "minimum_turn_radius_m": 0.3,
            "linear_speed_mps": 3.0, "angular_speed_radps": 1.2,
            "max_lateral_acceleration_mps2": 1.0,
            "max_acceleration_mps2": 1.0, "max_deceleration_mps2": 1.5,
            "curve_sample_spacing_m": 0.05, "fps": 10,
            "min_generated_route_length_m": 1.0,
            "max_generated_route_length_m": 100.0,
            "actions": {"tangent_connector_probability": 1.0,
                        "route_extension_attempts": 1}},
            "navigation": {"camera_trajectory_buffer_m": 0.0}}
        groups = [[{"points": previous_points, "reverse": False,
                    "events": [{"type": "first", "start_distance_m": 0.0,
                                "end_distance_m": 5.0}]}],
                  [{"points": following_points, "reverse": False,
                    "events": [{"type": "second", "start_distance_m": 0.0,
                                "end_distance_m": 5.0}]}]]
        with (patch("src.planning._shortest_path",
                    return_value=np.asarray([start_stub, end_stub])),
              patch("src.planning.load_camera_trajectory", return_value=None)):
            operations = _connect_segments(
                self.OpenNavMesh(), groups, np.random.default_rng(1), 0, config)

        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0]["tangent_connector_count"], 1)
        self.assertEqual([event["type"] for event in operations[0]["events"]],
                         ["first", "second"])
        _, frames, events = _build_operation_frames(operations, config)
        self.assertEqual([event["type"] for event in events], ["first", "second"])
        positions = np.asarray([frame[0] for frame in frames])
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 10.0
        first_end = events[0]["end_frame"]
        self.assertGreater(float(speeds[first_end]), 0.05)

    def test_tangent_connector_rejects_non_navigable_candidate(self):
        """验证 NavMesh 不可行时切向连接返回失败以供调用方回退。"""
        class BlockedNavMesh(self.OpenNavMesh):
            """拒绝所有候选点，模拟狭窄或被障碍截断的区域。"""

            def is_navigable(self, point, margin):
                """声明候选切线引导段不在可行区域内。"""
                return False

        result = _try_tangent_connector(
            BlockedNavMesh(), np.array([0.0, 0.0, 0.0]),
            np.array([4.0, 0.0, 4.0]), np.array([1.0, 0.0]),
            np.array([0.0, 1.0]), 0, None,
            {"motion": {"curve_sample_spacing_m": 0.05,
                        "minimum_turn_radius_m": 0.3},
             "navigation": {"camera_trajectory_buffer_m": 0.0}})
        self.assertIsNone(result)

    def test_ackermann_rejects_curve_tighter_than_minimum_radius(self):
        """验证 Ackermann 底盘拒绝小于配置最小转弯半径的轨迹。"""
        points = generate_curve_action([0, 0, 0], 0.0, "tangent_curve", 0.7,
                                       math.radians(50), 0.02)
        config = {"motion": {
            "model": "ackermann", "minimum_turn_radius_m": 1.0,
            "linear_speed_mps": 3.0, "angular_speed_radps": 1.2,
            "max_lateral_acceleration_mps2": 1.0,
            "max_acceleration_mps2": 1.0, "max_deceleration_mps2": 1.5,
            "curve_sample_spacing_m": 0.05, "fps": 10,
        }}
        with self.assertRaisesRegex(RuntimeError, "Ackermann"):
            time_parameterize_path(points, config)

    def test_time_parameterization_checks_discrete_angular_speed(self):
        """验证固定 FPS 输出帧的角速度不会因采样误差超过配置上限。"""
        points = generate_curve_action([0, 0, 0], 0.0, "tangent_curve", 0.3,
                                       math.radians(70), 0.02)
        config = {"motion": {
            "model": "differential_drive", "minimum_turn_radius_m": 0.3,
            "linear_speed_mps": 3.0, "angular_speed_radps": 1.2,
            "max_lateral_acceleration_mps2": 1.0,
            "max_acceleration_mps2": 1.0, "max_deceleration_mps2": 1.5,
            "curve_sample_spacing_m": 0.05, "fps": 10,
        }}
        frames = time_parameterize_path(points, config)
        yaws = np.unwrap(np.asarray([frame[1] for frame in frames]))
        speeds = np.linalg.norm(np.diff(
            np.asarray([frame[0] for frame in frames]), axis=0), axis=1) / 0.1
        self.assertLessEqual(float(np.abs(np.diff(yaws)).max() / 0.1), 1.2 + 1e-8)
        self.assertLessEqual(float(speeds.max()), 3.0 + 1e-8)

    def test_trajectory_tags_follow_existing_angle_thresholds(self):
        """验证轨迹级弧线标签按净航向角阈值归类且不创造新标签。"""
        frames = [(np.array([float(index), 0.0, 0.0]),
                   math.radians(60.0) * index / 10.0, index * 0.1)
                  for index in range(11)]
        tags = classify_trajectory_tags(
            frames, [{"type": "tangent_curve", "start_frame": 0, "end_frame": 10}])
        self.assertEqual(tags, ["curved_trajectory_moderate"])
        s_curve = generate_curve_action([0, 0, 0], 0.0, "s_curve", 4.0,
                                        math.radians(55), 0.1)
        delta = np.diff(s_curve[:, [0, 2]], axis=0)
        course = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        s_yaws = np.arctan2(-np.cos(course), -np.sin(course))
        s_frames = [(point, float(yaw), index * 0.1)
                    for index, (point, yaw) in enumerate(zip(s_curve[:-1], s_yaws))]
        s_frames.append((s_curve[-1], float(s_yaws[-1]), len(s_frames) * 0.1))
        s_tags = classify_trajectory_tags(
            s_frames, [{"type": "s_curve", "start_frame": 0,
                        "end_frame": len(s_frames) - 1}])
        self.assertFalse(any(tag.startswith("curved_trajectory_") for tag in s_tags))


if __name__ == "__main__":
    unittest.main()
