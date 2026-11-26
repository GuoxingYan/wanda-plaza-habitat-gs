"""中文文件头：覆盖动态行人代理胶囊碰撞检测的几何与输入边界。"""

import unittest

import numpy as np

from src.dynamic_pedestrians import minimum_capsule_clearance


class CapsuleClearanceTest(unittest.TestCase):
    """验证相交、间隔、端点投影、退化线段及逐帧检测。"""

    def test_crossing_capsules_are_reported_as_collision(self):
        """轴线相交且半径和为正时，剩余净距应为负值。"""
        first = np.array([[[-1, 0, 0, 1, 0, 0, 0.2]]], dtype=np.float32)
        second = np.array([[[0, 0, -1, 0, 0, 1, 0.2]]], dtype=np.float32)
        self.assertLess(minimum_capsule_clearance(first, second), 0.0)

    def test_separated_capsules_preserve_expected_clearance(self):
        """平行线段距离减去半径后应得到准确表面净距。"""
        first = np.array([[[0, 0, 0, 1, 0, 0, 0.2]]], dtype=np.float32)
        second = np.array([[[0, 1, 0, 1, 1, 0, 0.3]]], dtype=np.float32)
        self.assertAlmostEqual(minimum_capsule_clearance(first, second), 0.5, places=5)

    def test_checks_every_synchronized_frame(self):
        """只有末帧相交时也必须报告碰撞。"""
        first = np.repeat(np.array([[[0, 0, 0, 1, 0, 0, 0.1]]], dtype=np.float32), 2, axis=0)
        second = np.array([
            [[0, 1, 0, 1, 1, 0, 0.1]],
            [[0, 0, -1, 0, 0, 1, 0.1]],
        ], dtype=np.float32)
        self.assertLess(minimum_capsule_clearance(first, second), 0.0)

    def test_endpoint_projection_finds_closest_points(self):
        """最近点落在端点时仍应计算线段间的准确距离。"""
        first = np.array([[[0, 0, 0, 1, 0, 0, 0.1]]], dtype=np.float32)
        second = np.array([[[2, 0.5, 0, 3, 0.5, 0, 0.1]]], dtype=np.float32)
        self.assertAlmostEqual(
            minimum_capsule_clearance(first, second), np.sqrt(1.25) - 0.2, places=5,
        )

    def test_zero_length_capsules_are_treated_as_spheres(self):
        """零长度胶囊应退化为球体且不产生除零错误。"""
        first = np.array([[[0, 0, 0, 0, 0, 0, 0.25]]], dtype=np.float32)
        second = np.array([[[0, 0, 1, 0, 0, 1, 0.25]]], dtype=np.float32)
        self.assertAlmostEqual(minimum_capsule_clearance(first, second), 0.5, places=5)

    def test_rejects_mismatched_frame_counts(self):
        """帧数不一致时应拒绝错误的同步碰撞比较。"""
        first = np.zeros((1, 1, 7), dtype=np.float32)
        second = np.zeros((2, 1, 7), dtype=np.float32)
        with self.assertRaises(ValueError):
            minimum_capsule_clearance(first, second)


if __name__ == "__main__":
    unittest.main()
