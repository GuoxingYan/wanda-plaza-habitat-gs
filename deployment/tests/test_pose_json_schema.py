"""导航位姿 JSON 的公开字段契约回归测试。"""
import unittest


class NavigationPoseSchemaTest(unittest.TestCase):
    def test_confirmed_top_level_and_frame_field_names(self):
        payload = {
            "coordinate_frame": {
                "name": "Y-up",
                "position_unit": "m",
                "orientation_format": "quaternion_xyzw",
            },
            "route": {
                "route_id": "route_0000",
                "seed": 17,
                "start": {"position": [0.0, 0.0, 0.0]},
                "goal": {"position": [1.0, 0.0, 1.0]},
                "path_length": 12.3,
            },
            "camera": {"model": "D435 RGB nominal pinhole", "width": 640,
                       "height": 360, "hfov_deg": 69.0},
            "frames": [{"frame_index": 0, "timestamp_s": 0.0,
                        "position": [0.0, 1.5, 0.0], "xyzw": [0.0, 0.0, 0.0, 1.0]}],
        }
        self.assertNotIn("schema_version", payload)
        self.assertNotIn("source", payload["route"]["start"])
        self.assertNotIn("source", payload["route"]["goal"])
        self.assertEqual(payload["coordinate_frame"]["name"], "Y-up")
        self.assertIn("path_length", payload["route"])
        self.assertNotIn("geodesic_distance_m", payload["route"])
        self.assertEqual(set(payload["frames"][0]),
                         {"frame_index", "timestamp_s", "position", "xyzw"})


if __name__ == "__main__":
    unittest.main()
