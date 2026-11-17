"""NavMesh 代理宽度与额外障碍余量的配置验证。"""
import unittest

from src.planning import navmesh_agent_radius


class NavmeshClearanceTest(unittest.TestCase):
    def test_default_fullres_configuration_adds_clearance_to_half_width(self):
        self.assertAlmostEqual(navmesh_agent_radius({
            "agent_width_m": 0.75,
            "obstacle_clearance_m": 0.3,
        }), 0.525)

    def test_clearance_must_be_non_negative_and_finite(self):
        for value in (-0.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                navmesh_agent_radius({"agent_width_m": 0.75,
                                      "obstacle_clearance_m": value})


if __name__ == "__main__":
    unittest.main()
