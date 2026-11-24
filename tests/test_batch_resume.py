"""批量续跑所需文件集合的回归测试。"""
import unittest
import tempfile
from pathlib import Path

from src.cli import effective_config_text, is_route_complete, required_route_outputs
from src.output_manifest import build_route_manifest, config_digest, write_batch_manifest, write_json_atomic


class BatchResumeTest(unittest.TestCase):
    def test_mcap_resume_checks_recorded_configuration(self):
        """已有配置快照不能让不同编码的旧 MCAP 被当成本次产物。"""
        with tempfile.TemporaryDirectory() as directory:
            route = Path(directory)
            config = {"output": {"mcap": True, "video": False, "poses_json": False,
                                 "validation_json": False},
                      "camera": {"mcap": {"rgb": {"encoding": "h264"}}}}
            name = f"trajectory_{route.name}.mcap"
            (route / name).write_bytes(b"completed")
            (route / "effective_config.yaml").touch()
            write_json_atomic(route / "manifest.json", {
                "schema_version": 1, "config_sha256": config_digest(config), "mcap": name,
                "artifacts": [{"path": name, "size_bytes": 9}],
            })
            self.assertTrue(is_route_complete(route, {}, config))
            changed = {**config, "camera": {"mcap": {"rgb": {"encoding": "jpeg"}}}}
            self.assertFalse(is_route_complete(route, {}, changed))
            (route / name).write_bytes(b"short")
            self.assertFalse(is_route_complete(route, {}, config))

    def test_manifest_indexes_route_without_opening_mcap(self):
        """验证 manifest 索引编码和文件大小且不依赖 MCAP 读取。"""
        with tempfile.TemporaryDirectory() as directory:
            route = Path(directory) / "route_0000"
            route.mkdir()
            (route / "trajectory_route_0000.mcap").write_bytes(b"mcap-data")
            config = {"navigation": {"seed": 7},
                      "camera": {"width": 640, "height": 360,
                                 "mcap": {"rgb": {"encoding": "h264"},
                                          "depth": {"encoding": "16UC1", "scale_m": 0.01}}},
                      "motion": {"fps": 10}}
            route_manifest = build_route_manifest(
                route, config, {"tags": ["turn"], "rgb_frames": 12})
            self.assertEqual(route_manifest["mcap"], "trajectory_route_0000.mcap")
            self.assertEqual(route_manifest["encoding"]["rgb"]["codec"], "h264")
            self.assertEqual(route_manifest["encoding"]["depth"]["format"], "16UC1")
            self.assertAlmostEqual(route_manifest["duration_s"], 1.1)
            write_json_atomic(route / "manifest.json", route_manifest)
            batch = write_batch_manifest(directory)
            self.assertEqual(batch[0]["frame_count"], 12)
            self.assertEqual(batch[0]["mcap"], "route_0000/trajectory_route_0000.mcap")

    def test_effective_config_preserves_chinese_comments_and_route_seed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "# 全局说明\nscene:\n  directory: scene\n\n"
                "navigation:\n  # 固定随机种子，用于复现路线\n  seed: 17\n"
                "  min_route_length_m: 15\n",
                encoding="utf-8")
            output = effective_config_text(path, {"navigation": {"seed": 42}})
        self.assertIn("# 固定随机种子，用于复现路线", output)
        self.assertIn("  seed: 42", output)
        self.assertIn("  min_route_length_m: 15", output)

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

    def test_density_comparison_not_required_without_camera_prior(self):
        files = required_route_outputs(
            {"video": False, "poses_json": False, "trajectory_topdown": False,
             "mesh_navmesh_overlay": False, "gaussian_navmesh_overlay": False,
             "camera_trajectory_density_comparison": True},
            {"use_camera_trajectory_prior": False})
        self.assertNotIn("camera_trajectory_density_comparison.png", files)

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
