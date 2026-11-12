"""采集相机轨迹坐标变换和缓冲距离判定测试。"""
import unittest

import numpy as np
from scipy.spatial import cKDTree

from src.planning import (_SNAPPED_CANDIDATE_CACHE, _snapped_camera_candidates,
                                _load_camera_trajectory_file, classify_camera_trajectory_islands,
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

    def test_density_filter_requires_points_and_area_density(self):
        keep, counts, density = classify_camera_trajectory_islands(
            np.array([100.0, 1000.0, 10.0]),
            np.array([0] * 30 + [1] * 2 + [2] * 1),
            min_points=5, min_points_per_100m2=1.0)
        self.assertTrue(keep[0])
        self.assertFalse(keep[1])
        self.assertFalse(keep[2])
        np.testing.assert_array_equal(counts, [30, 2, 1])
        self.assertAlmostEqual(density[0], 30.0)

    def test_capture_tracks_merge_cameras_per_frame_and_split_video_sequences(self):
        import json
        import tempfile
        from pathlib import Path

        cameras = [
            {"img_name": "VIDEO_A_000001_0", "position": [0.0, 0.0, 1.0]},
            {"img_name": "VIDEO_A_000001_1", "position": [0.0, 0.0, 1.0]},
            {"img_name": "VIDEO_A_000002_0", "position": [1.0, 0.0, 1.0]},
            {"img_name": "VIDEO_B_000001_0", "position": [10.0, 0.0, 1.0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(json.dumps(cameras), encoding="utf-8")
            prior = _load_camera_trajectory_file(str(path), path.stat().st_size,
                                                 path.stat().st_mtime_ns)
        self.assertEqual(len(prior["capture_segments"]), 2)
        self.assertEqual([len(segment) for segment in prior["capture_segments"]], [2, 1])
        np.testing.assert_allclose(prior["capture_segments"][0][:, 0], [0.0, 1.0])
        np.testing.assert_allclose(prior["capture_segments"][1][:, 0], [10.0])


if __name__ == "__main__":
    unittest.main()
