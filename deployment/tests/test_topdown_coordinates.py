"""用非对称四角颜色检测俯视底图的翻转和边界丢点。"""
import unittest
import numpy as np
from wanda_nav.visualization import _pixel_colors

class TopdownCoordinatesTest(unittest.TestCase):
    def test_world_corners_match_image_rows(self):
        points = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=float)
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
        image, occupied, extent = _pixel_colors(points, colors, ((0, 2), (0, 2)), 1, smoothing_px=0)
        # origin=upper、extent=(xmin,xmax,zmin,zmax)：第一行代表 Z 最大一侧。
        np.testing.assert_allclose(image, [[colors[1], colors[3]], [colors[0], colors[2]]])
        self.assertTrue(occupied.all())
        self.assertEqual(extent, (0, 2, 0, 2))

if __name__ == '__main__':
    unittest.main()
