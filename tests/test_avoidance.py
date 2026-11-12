"""临时障碍物配置、净距与绕行折线回归测试。"""
import unittest

import numpy as np

from src.avoidance import composite_obstacle_rgb, plan_around_synthetic_obstacle


class AvoidanceTest(unittest.TestCase):
    def test_projected_obstacle_changes_rgb_frame(self):
        rgb = np.zeros((180, 320, 3), dtype=np.uint8)
        obstacle = {"triggered": True, "center": [0.0, 0.5, -3.0],
                    "radius_m": 0.35, "height_m": 1.0}
        rendered = composite_obstacle_rgb(
            rgb, obstacle, [0.0, 1.5, 0.0], 0.0,
            {"width": 320, "height": 180, "hfov_deg": 69.0})
        self.assertGreater(np.count_nonzero(rendered), 50)

    def test_obstacle_on_original_route_triggers_clear_detour(self):
        class FakePathfinder:
            def get_bounds(self):
                return np.array([[-5, 0, -5], [15, 2, 5]], dtype=float)

            def snap_point(self, point):
                return point

            def is_navigable(self, point, margin):
                return True

        route = np.array([[0., 0., 0.], [10., 0., 0.]])
        config = {
            "avoidance": {"enabled": True, "placement_fraction": 0.5,
                          "obstacle_radius_m": 0.35, "obstacle_height_m": 1.0,
                          "trigger_distance_m": 2.0, "safety_margin_m": 0.25,
                          "grid_resolution_m": 0.2, "max_search_cells": 250000,
                          "force": False},
            "navmesh": {"agent_width_m": 0.75},
        }
        detour, obstacle = plan_around_synthetic_obstacle(FakePathfinder(), route, config)
        self.assertTrue(obstacle["triggered"])
        self.assertGreater(obstacle["height_m"], 0.5)
        self.assertGreater(len(detour), 2)
        clearance = obstacle["planning_clearance_radius_m"]
        for first, second in zip(detour, detour[1:]):
            # A* route points must lie outside the obstacle footprint plus robot margin.
            for point in (first, second):
                self.assertGreaterEqual(np.linalg.norm(point[[0, 2]] - np.array([5., 0.])),
                                        clearance - 0.4)


if __name__ == "__main__":
    unittest.main()
