"""采集相机轨迹坐标变换和缓冲距离判定测试。"""
import unittest

import numpy as np
from scipy.spatial import cKDTree

from wanda_nav.planning import route_within_camera_buffer


class CameraTrajectoryPriorTest(unittest.TestCase):
    def setUp(self):
        positions = np.array([[x, 0.0, 0.0] for x in np.arange(0.0, 10.1, 0.5)])
        self.prior = {"tree": cKDTree(positions[:, [0, 2]])}

    def test_route_inside_buffer_is_accepted(self):
        points = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]])
        self.assertTrue(route_within_camera_buffer(points, self.prior, 2.0))

    def test_route_leaving_buffer_is_rejected(self):
        points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 5.0], [10.0, 0.0, 0.0]])
        self.assertFalse(route_within_camera_buffer(points, self.prior, 2.0))

    def test_disabled_prior_accepts_any_route(self):
        points = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 100.0]])
        self.assertTrue(route_within_camera_buffer(points, None, 1.0))


if __name__ == "__main__":
    unittest.main()
