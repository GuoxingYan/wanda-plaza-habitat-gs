"""批量续跑所需文件集合的回归测试。"""
import unittest
import tempfile
from pathlib import Path

from wanda_nav.cli import is_route_complete, required_route_outputs


class BatchResumeTest(unittest.TestCase):
    def test_old_pose_schema_is_not_considered_complete(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"output": {"video": False, "poses_json": True,
                                  "trajectory_topdown": False,
                                  "mesh_navmesh_overlay": False,
                                  "gaussian_navmesh_overlay": False}}
            report = {"config": config}
            (root / "validation.json").touch()
            (root / "navigation_poses.json").write_text(json.dumps({
                "route": {"start": {"agent_position": [0, 0, 0]},
                          "goal": {"agent_position": [1, 0, 0]}},
                "frames": [{"agent_position": [0, 0, 0], "position": [0, 1, 0]}],
            }), encoding="utf-8")
            self.assertFalse(is_route_complete(root, report, config))

    def test_video_disabled_does_not_require_mp4(self):
        files = required_route_outputs({"video": False, "poses_json": True,
                                        "trajectory_topdown": True,
                                        "mesh_navmesh_overlay": True,
                                        "gaussian_navmesh_overlay": False})
        self.assertNotIn("navigation_rgb.mp4", files)
        self.assertIn("navigation_poses.json", files)

    def test_optional_outputs_are_not_required(self):
        files = required_route_outputs({"video": False, "poses_json": False,
                                        "trajectory_topdown": False,
                                        "mesh_navmesh_overlay": False,
                                        "gaussian_navmesh_overlay": False})
        self.assertEqual(files, ["validation.json"])

    def test_gaussian_overlay_requires_both_files(self):
        files = required_route_outputs({"video": False, "poses_json": False,
                                        "trajectory_topdown": False,
                                        "mesh_navmesh_overlay": False,
                                        "gaussian_navmesh_overlay": True})
        self.assertEqual(files, ["validation.json", "gaussian_topdown_raw.png",
                                 "gaussian_navmesh_topdown.png"])

    def test_video_disabled_can_resume_without_rgb_frame_count(self):
        import tempfile
        from pathlib import Path

        config = {"output": {"video": False, "poses_json": False,
                             "trajectory_topdown": False,
                             "mesh_navmesh_overlay": False,
                             "gaussian_navmesh_overlay": False}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "validation.json").touch()
            report = {"config": config, "rgb_frames": 0}
            self.assertTrue(is_route_complete(root, report, config))

    def test_video_enabled_requires_rgb_frames(self):
        import tempfile
        from pathlib import Path

        config = {"output": {"video": True, "poses_json": False,
                             "trajectory_topdown": False,
                             "mesh_navmesh_overlay": False,
                             "gaussian_navmesh_overlay": False}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "validation.json").touch()
            report = {"config": config, "rgb_frames": 0}
            self.assertFalse(is_route_complete(root, report, config))


if __name__ == "__main__":
    unittest.main()
