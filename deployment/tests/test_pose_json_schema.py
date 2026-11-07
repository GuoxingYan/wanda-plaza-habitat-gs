"""导航位姿 JSON 的公开字段契约回归测试。"""
import unittest


class NavigationPoseSchemaTest(unittest.TestCase):
    def test_confirmed_top_level_and_frame_field_names(self):
        payload = {
            "coordinate_frame": {
                "name": "Y-up",
                "position_unit": "m",
                "orientation_format": "xyzw",
            },
            "route": {
                "route_id": "route_0000",
                "seed": 17,
                "start": {"agent_position": [0.0, 0.0, 0.0],
                          "camera_position": [0.0, 1.5, 0.0]},
                "goal": {"agent_position": [1.0, 0.0, 1.0],
                         "camera_position": [1.0, 1.5, 1.0]},
                "path_length": 12.3,
            },
            "camera": {"model": "D435 RGB nominal pinhole", "width": 640,
                       "height": 360, "hfov_deg": 69.0},
            "motion": {"model": "differential_drive", "frame_rate_fps": 10,
                       "max_linear_speed_mps": 3.0, "max_acceleration_mps2": 1.0,
                       "max_deceleration_mps2": 1.5,
                       "max_angular_speed_radps": 1.2},
            "frames": [{"frame_index": 0, "timestamp_s": 0.0,
                        "agent_position": [0.0, 0.0, 0.0],
                        "position": [0.0, 1.5, 0.0], "xyzw": [0.0, 0.0, 0.0, 1.0]}],
        }
        self.assertNotIn("schema_version", payload)
        self.assertNotIn("source", payload["route"]["start"])
        self.assertNotIn("source", payload["route"]["goal"])
        self.assertEqual(payload["coordinate_frame"]["name"], "Y-up")
        self.assertEqual(payload["coordinate_frame"]["orientation_format"], "xyzw")
        self.assertIn("path_length", payload["route"])
        self.assertNotIn("geodesic_distance_m", payload["route"])
        self.assertEqual(set(payload["frames"][0]),
                         {"frame_index", "timestamp_s", "agent_position", "position", "xyzw"})
        self.assertNotEqual(payload["route"]["start"]["agent_position"],
                            payload["route"]["start"]["camera_position"])
        self.assertEqual(payload["motion"]["model"], "differential_drive")


if __name__ == "__main__":
    unittest.main()
