"""用非对称四角颜色检测俯视底图的翻转和边界丢点。"""
import unittest
import numpy as np
from wanda_nav.visualization import (_load_cached_navmesh, _pixel_colors,
                                     _static_visualization_key)
from wanda_nav.visualization import project_navmesh_to_topdown_pixels

class TopdownCoordinatesTest(unittest.TestCase):
    def test_navmesh_triangle_cache_avoids_rebuilding_islands(self):
        import tempfile
        from pathlib import Path

        class FakePathfinder:
            num_islands = 1

            def __init__(self):
                self.calls = 0

            def build_navmesh_vertices(self, island_id):
                self.calls += 1
                return np.array([[0, 0, 0], [2, 0, 0], [0, 0, 2]], dtype=float)

        with tempfile.TemporaryDirectory() as directory:
            navmesh = Path(directory) / "wanda.navmesh"
            navmesh.write_bytes(b"navmesh-v1")
            first = FakePathfinder()
            triangles, bounds = _load_cached_navmesh(first, navmesh)
            self.assertEqual(first.calls, 1)
            self.assertEqual(bounds, (0.0, 2.0, 0.0, 2.0))

            second = FakePathfinder()
            cached, cached_bounds = _load_cached_navmesh(second, navmesh)
            self.assertEqual(second.calls, 0)
            self.assertEqual(cached_bounds, bounds)
            np.testing.assert_array_equal(cached[0][1], triangles[0][1])

    def test_static_visualization_cache_key_tracks_inputs_and_render_settings(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            navmesh, mesh = root / "wanda.navmesh", root / "mesh.ply"
            navmesh.write_bytes(b"nav")
            mesh.write_bytes(b"mesh")
            config = {"navigation": {"seed": 1}, "output": {"topdown_max_points": None},
                      "visualization": {"topdown_pixel_size_m": 0.1,
                                        "topdown_smoothing_px": 1,
                                        "image_width_px": 200, "image_height_px": 100,
                                        "dpi": 100, "navmesh_color": "#00aa00",
                                        "mesh_overlay_alpha": 0.3,
                                        "detail_regions_xz": []}}
            first = _static_visualization_key(navmesh, mesh, config)["key"]
            config["visualization"]["topdown_pixel_size_m"] = 0.2
            second = _static_visualization_key(navmesh, mesh, config)["key"]
            self.assertNotEqual(first, second)

    def test_world_corners_match_image_rows(self):
        points = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=float)
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
        image, occupied, extent = _pixel_colors(points, colors, ((0, 2), (0, 2)), 1, smoothing_px=0)
        # origin=upper、extent=(xmin,xmax,zmin,zmax)：第一行代表 Z 最大一侧。
        np.testing.assert_allclose(image, [[colors[1], colors[3]], [colors[0], colors[2]]])
        self.assertTrue(occupied.all())
        self.assertEqual(extent, (0, 2, 0, 2))

    def test_topdown_camera_projection_uses_pixel_xy(self):
        # 俯视相机向下看，世界 +Z 投影到图像上方，X 增大向右。
        height, width, focal = 200, 300, 100.0
        center_x, center_z, camera_y = 0.0, 0.0, 10.0
        points_xyz = np.array([
            [-1.0, 0.0, -1.0], [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0], [1.0, 0.0, 1.0],
        ])
        projected = project_navmesh_to_topdown_pixels(
            points_xyz.reshape(1, 4, 3)[:, :3], camera_y, focal,
            center_x, center_z, width, height)[0]
        self.assertLess(projected[1, 1], projected[0, 1])
        self.assertLess(projected[0, 0], projected[2, 0])

    def test_projection_rejects_vertices_above_camera(self):
        triangle = np.array([[[0, 11, 0], [1, 0, 0], [0, 0, 1]]])
        with self.assertRaises(ValueError):
            project_navmesh_to_topdown_pixels(triangle, 10, 100, 0, 0, 300, 200)

if __name__ == '__main__':
    unittest.main()
