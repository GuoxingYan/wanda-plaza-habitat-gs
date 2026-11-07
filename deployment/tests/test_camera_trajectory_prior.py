"""采集相机轨迹坐标变换和缓冲距离判定测试。"""
import unittest

import numpy as np
from scipy.spatial import cKDTree

from wanda_nav.planning import (_SNAPPED_CANDIDATE_CACHE, _snapped_camera_candidates,
                                route_within_camera_buffer)


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

    def test_snapped_candidates_are_cached_by_input_files(self):
        import tempfile
        from pathlib import Path

        class FakePathfinder:
            def __init__(self):
                self.calls = 0

            def snap_point(self, position):
                self.calls += 1
                return position

            def get_island(self, position):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            navmesh = Path(directory) / "wanda.navmesh"
            trajectory_file = Path(directory) / "cameras.json"
            navmesh.write_bytes(b"navmesh")
            trajectory_file.write_text("[]", encoding="utf-8")
            prior = {"source_file": str(trajectory_file),
                     "positions": np.array([[0., 0., 0.], [1., 0., 0.]])}
            _SNAPPED_CANDIDATE_CACHE.clear()
            first = FakePathfinder()
            points, islands = _snapped_camera_candidates(first, prior, navmesh, 2.0)
            self.assertEqual(first.calls, 2)

            second = FakePathfinder()
            cached_points, cached_islands = _snapped_camera_candidates(second, prior, navmesh, 2.0)
            self.assertEqual(second.calls, 0)
            np.testing.assert_array_equal(points, cached_points)
            np.testing.assert_array_equal(islands, cached_islands)


if __name__ == "__main__":
    unittest.main()
