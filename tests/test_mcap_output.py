"""MCAP 轨迹归档的通道与同步消息回归测试。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages
from google.protobuf.descriptor_pb2 import FileDescriptorSet

from src.mcap_output import (NavigationMcapWriter, _compressed_depth, _compressed_image,
                             _quantize_depth, mcap_camera_options, sanitize_depth)


class McapOutputTest(unittest.TestCase):
    def test_config_registers_preview_and_lossless_depth_channels(self):
        """验证深度预览和可选无损深度通道按配置注册。"""
        with tempfile.NamedTemporaryFile(suffix=".mcap") as stream:
            writer = NavigationMcapWriter(
                stream.name, "route_configured",
                {"width": 2, "height": 2, "fx": 1.0, "fy": 1.0,
                 "cx": 0.5, "cy": 0.5},
                metadata={"rgb": {"encoding": "jpeg", "quality": 88},
                          "depth": {"encoding": "32FC1"},
                          "depth_preview": {"enabled": True, "min_m": 0.0,
                                            "max_m": 8.0, "encoding": "mono8"},
                          "depth_compression": {"enabled": True}})
            self.assertIn("/camera/depth_preview", writer.channels)
            self.assertIn("/camera/depth_compressed", writer.channels)
            writer.add_frame(writer.timestamp_ns(0), np.zeros((2, 2, 3), dtype=np.uint8),
                             np.full((2, 2), 2.0, dtype=np.float32), [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0], 0)
            writer.finish()
            with open(stream.name, "rb") as saved:
                topics = {channel.topic for channel in make_reader(saved).get_summary().channels.values()}
            self.assertIn("/camera/depth_preview", topics)
            self.assertIn("/camera/depth_compressed", topics)
            with open(stream.name, "rb") as saved:
                messages = [(channel, message) for _, channel, message
                            in make_reader(saved).iter_messages()
                            if channel.topic == "/camera/depth_compressed"]
            self.assertEqual(len(messages), 1)
            self.assertTrue(messages[0][1].data.startswith(b"D32Z"))
            import struct
            import zlib
            payload = messages[0][1].data
            magic, height, width, stride = struct.unpack("<4sIII", payload[:16])
            self.assertEqual((magic, height, width, stride), (b"D32Z", 2, 2, 8))
            restored = np.frombuffer(zlib.decompress(payload[16:]), dtype="<f4").reshape(2, 2)
            np.testing.assert_array_equal(restored, np.full((2, 2), 2.0, dtype=np.float32))

    def test_h264_writer_emits_one_decodable_video_packet_per_frame(self):
        """验证 H.264 配置在 MCAP 中产生逐帧 Foxglove 视频消息。"""
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is None:
            self.skipTest("系统未安装 ffmpeg")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.mcap"
            writer = NavigationMcapWriter(
                path, "route_h264",
                {"width": 32, "height": 24, "fx": 16.0, "fy": 16.0,
                 "cx": 15.5, "cy": 11.5},
                metadata={"rgb": {"encoding": "h264", "crf": 23, "fps": 10.0},
                          "depth": {"encoding": "16UC1", "scale_m": 0.01}})
            for index in range(3):
                rgb = np.full((24, 32, 3), index * 60, dtype=np.uint8)
                writer.add_frame(writer.timestamp_ns(index / 10), rgb,
                                 np.full((24, 32), 2.0, dtype=np.float32),
                                 [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], index)
            writer.finish()

            with path.open("rb") as stream:
                reader = make_reader(stream)
                messages = [(channel, message) for _, channel, message
                            in reader.iter_messages() if channel.topic == "/camera/rgb"]
                summary = reader.get_summary()
            self.assertEqual(len(messages), 3)
            self.assertEqual(messages[0][0].schema_id,
                             next(channel.schema_id for channel in summary.channels.values()
                                  if channel.topic == "/camera/rgb"))
            packets = [_extract_video_payload(message.data) for _, message in messages]
            encoded = b"".join(packets)
            decoded = subprocess.run(
                [shutil.which("ffmpeg"), "-v", "error", "-f", "h264", "-i", "pipe:0",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                input=encoded, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(len(decoded.stdout), 3 * 24 * 32 * 3)

    def test_h264_abort_terminates_encoder_process(self):
        """验证中止时编码器子进程退出且其临时文件被清理。"""
        import os
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.mcap"
            writer = NavigationMcapWriter(
                path, "route_abort_process",
                {"width": 32, "height": 24, "fx": 16.0, "fy": 16.0,
                 "cx": 15.5, "cy": 11.5},
                metadata={"rgb": {"encoding": "h264"}})
            writer.add_frame(writer.timestamp_ns(0), np.zeros((24, 32, 3), dtype=np.uint8),
                             np.ones((24, 32), dtype=np.float32), [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0], 0)
            process = writer._video_process
            temporary_path = writer._rgb_temp.name
            writer.abort()
            self.assertIsNotNone(process.poll())
            self.assertFalse(os.path.exists(temporary_path))

    def test_abort_closes_mcap_and_removes_rgb_encoder_tempfile(self):
        """验证录制失败中止时关闭 MCAP 并清理 H.264 临时输入。"""
        import os
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.mcap"
            writer = NavigationMcapWriter(
                path, "route_abort",
                {"width": 2, "height": 2, "fx": 1.0, "fy": 1.0,
                 "cx": 0.5, "cy": 0.5},
                metadata={"rgb": {"encoding": "h264"}})
            writer.add_frame(writer.timestamp_ns(0), np.zeros((2, 2, 3), dtype=np.uint8),
                             np.ones((2, 2), dtype=np.float32), [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0], 0)
            temporary_path = writer._rgb_temp.name
            writer.abort()
            self.assertTrue(writer.stream.closed)
            self.assertFalse(os.path.exists(temporary_path))

    def test_abort_without_frames_closes_mcap_file(self):
        """验证首帧写入前异常时同样可以关闭 MCAP 文件。"""
        with tempfile.TemporaryDirectory() as directory:
            writer = NavigationMcapWriter(
                Path(directory) / "empty.mcap", "route_empty",
                {"width": 2, "height": 2, "fx": 1.0, "fy": 1.0,
                 "cx": 0.5, "cy": 0.5})
            writer.abort()
            self.assertTrue(writer.stream.closed)
            self.assertFalse((Path(directory) / "empty.mcap").exists())

    def test_finish_failure_aborts_and_removes_partial_mcap(self):
        """验证 FFmpeg 输出异常时 finish 自动清理半成品与子进程。"""
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.mcap"
            writer = NavigationMcapWriter(
                path, "route_failed",
                {"width": 32, "height": 24, "fx": 16.0, "fy": 16.0,
                 "cx": 15.5, "cy": 11.5},
                metadata={"rgb": {"encoding": "h264"}})
            writer.add_frame(writer.timestamp_ns(0), np.zeros((24, 32, 3), dtype=np.uint8),
                             np.ones((24, 32), dtype=np.float32), [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0], 0)
            with patch("src.mcap_output._split_annexb_access_units",
                       side_effect=RuntimeError("decode split failed")):
                with self.assertRaisesRegex(RuntimeError, "decode split failed"):
                    writer.finish()
            self.assertTrue(writer.stream.closed)
            self.assertFalse(path.exists())
            self.assertIsNotNone(writer._video_process.poll())

    def test_abort_preserves_preexisting_mcap(self):
        """验证失败重录不删除原先已完整写入的目标 MCAP。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.mcap"
            path.write_bytes(b"previous-valid-file")
            writer = NavigationMcapWriter(
                path, "route_retry",
                {"width": 2, "height": 2, "fx": 1.0, "fy": 1.0,
                 "cx": 0.5, "cy": 0.5})
            writer.abort()
            self.assertEqual(path.read_bytes(), b"previous-valid-file")

    def test_depth_quantization_uses_configured_centimeter_scale(self):
        """验证厘米量化使用 uint16 且误差不超过半个量化步长。"""
        values = np.array([[1.234, 2.345], [0.0, 500.0]], dtype=np.float32)
        quantized = _quantize_depth(values, scale_m=0.01, invalid_value=0)
        np.testing.assert_array_equal(quantized, [[123, 235], [0, 50000]])
        restored = quantized.astype(np.float32) * 0.01
        self.assertLessEqual(float(np.max(np.abs(restored - values))), 0.005)

    def test_depth_quantization_rejects_overflow_and_reserved_value(self):
        """验证深度超出编码量程时失败且有效值不占用无效码。"""
        with self.assertRaises(ValueError):
            _quantize_depth(np.array([[700.0]], dtype=np.float32), 0.01, 0)
        quantized = _quantize_depth(np.array([[1.0]], dtype=np.float32), 0.01, 100)
        self.assertNotEqual(int(quantized[0, 0]), 100)
        quantized = _quantize_depth(np.array([[0.5, 5.0]], dtype=np.float32), 0.01, 0, 2.0)
        np.testing.assert_array_equal(quantized, [[50, 0]])

    def test_pipeline_options_forward_camera_settings(self):
        """验证 YAML 相机配置和运动帧率完整传入 MCAP 写入器。"""
        camera = {"far_m": 120.0, "jpeg_quality": 87,
                  "depth_preview": {"enabled": False},
                  "depth_compression": {"enabled": True},
                  "mcap": {"rgb": {"encoding": "h264", "crf": 21},
                           "depth": {"encoding": "16UC1", "scale_m": 0.01}}}
        options = mcap_camera_options(camera, 12)
        self.assertEqual(options["rgb"],
                         {"encoding": "h264", "quality": 87, "crf": 21, "fps": 12.0})
        self.assertEqual(options["depth"], {"encoding": "16UC1", "scale_m": 0.01})
        self.assertEqual(options["depth_preview"], {"enabled": False})
        self.assertEqual(options["depth_compression"], {"enabled": True})
        self.assertEqual(options["depth_far_clip_m"], 120.0)

    def test_depth_holes_and_out_of_range_values_use_far_clip(self):
        """验证无效深度、非正深度及超远值都被替换为远裁剪距离。"""
        values = np.array([[1.25, np.nan, np.inf], [0.0, -1.0, 600.0]], dtype=np.float32)
        result = sanitize_depth(values, 500.0)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, [[1.25, 500.0, 500.0], [500.0, 500.0, 500.0]])

    def test_writes_rgb_depth_pose_camera_info_and_trajectory_tags(self):
        with tempfile.NamedTemporaryFile(suffix=".mcap") as stream:
            writer = NavigationMcapWriter(
                stream.name, "route_0000",
                {"width": 4, "height": 3, "fx": 2.0, "fy": 2.0,
                 "cx": 1.5, "cy": 1.0},
                metadata={"rgb": {"encoding": "jpeg", "quality": 100},
                          "depth": {"encoding": "16UC1", "scale_m": 0.01},
                          "depth_preview": {"enabled": False}},
                tags=["straight_backward", "curved_trajectory_major"])
            stamp = writer.timestamp_ns(0)
            rgb = np.zeros((3, 4, 3), dtype=np.uint8)
            depth = np.full((3, 4), 2.5, dtype=np.float32)
            writer.add_frame(stamp, rgb, depth, [1.0, 2.0, 3.0],
                             [0.0, 0.0, 0.0, 1.0], 0)
            writer.finish()

            with open(stream.name, "rb") as saved:
                reader = make_reader(saved)
                summary = reader.get_summary()
                self.assertEqual(reader.get_header().profile, "ros2")
                self.assertTrue(summary.chunk_indexes)
                self.assertTrue(all(chunk.compression == "zstd" for chunk in summary.chunk_indexes))
                schemas = {schema.id: schema for schema in summary.schemas.values()}
                self.assertIn(b"MSG: geometry_msgs/Pose\n", schemas[
                    next(channel.schema_id for channel in summary.channels.values()
                         if channel.topic == "/trajectory/pose")].data)
                pose_schema = schemas[next(channel.schema_id for channel in summary.channels.values()
                                           if channel.topic == "/trajectory/pose")]
                self.assertIn(b"MSG: geometry_msgs/Point\n", pose_schema.data)
                self.assertIn(b"MSG: geometry_msgs/Quaternion\n", pose_schema.data)
                self.assertNotIn(b"geometry_msgs/msg/", pose_schema.data)
                tag_channel = next(channel for channel in summary.channels.values()
                                   if channel.topic == "/trajectory/tags")
                self.assertEqual(schemas[tag_channel.schema_id].name, "wanda_nav.TrajectoryTags")
                self.assertNotIn("ros2_type", tag_channel.metadata)
                messages = list(reader.iter_messages())
                descriptor = next(schema.data for schema in summary.schemas.values()
                                  if schema.encoding == "protobuf")
            topics = [channel.topic for _, channel, _ in messages]
            self.assertEqual(set(topics), {
                "/trajectory/metadata", "/camera/rgb", "/camera/depth",
                "/camera/info", "/trajectory/pose", "/trajectory/tags"})
            self.assertEqual(topics.count("/camera/rgb"), 1)
            self.assertEqual(topics.count("/camera/depth"), 1)
            tag = next(message.data for _, channel, message in messages
                       if channel.topic == "/trajectory/tags")
            self.assertIn(b"straight_backward", tag)
            self.assertIn(b"curved_trajectory_major", tag)
            self.assertIn(b"route_0000", tag)
            image = next(message.data for _, channel, message in messages
                         if channel.topic == "/camera/rgb")
            self.assertTrue(image.startswith(b"\x00\x01\x00\x00"))
            self.assertIn(b"jpeg\x00", image)
            self.assertTrue(image.endswith(b"\xff\xd9"))
            decoded = list(read_ros2_messages(stream.name, topics=[
                "/camera/depth", "/camera/info", "/trajectory/pose"]))
            self.assertEqual(len(decoded), 3)
            decoded_by_topic = {message.channel.topic: message.ros_msg for message in decoded}
            self.assertEqual(decoded_by_topic["/camera/depth"].encoding, "16UC1")
            np.testing.assert_array_equal(
                np.frombuffer(decoded_by_topic["/camera/depth"].data, dtype=np.uint16),
                np.full(12, 250, dtype=np.uint16))
            self.assertEqual(decoded_by_topic["/camera/info"].width, 4)
            self.assertEqual(decoded_by_topic["/trajectory/pose"].pose.position.x, 1.0)
            self.assertEqual(list(read_ros2_messages(
                stream.name, topics=["/camera/depth_preview"])), [])
            self.assertIn(b"MSG: geometry_msgs/Point\n", pose_schema.data)
            parsed = FileDescriptorSet.FromString(descriptor)
            self.assertEqual(parsed.file[0].message_type[0].name, "TrajectoryTags")
            for _, channel, message in messages:
                if channel.topic in ("/camera/rgb", "/camera/depth", "/trajectory/pose"):
                    self.assertGreater(message.log_time, 0)

    def test_compressed_rgb_is_decodable_jpeg_at_requested_quality(self):
        """验证压缩 RGB CDR 消息声明 jpeg 且包含有效 JPEG 图像。"""
        from io import BytesIO
        from PIL import Image

        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        payload = _compressed_image(1_000_000_000, "camera_color_optical_frame", rgb, 90)
        marker = payload.index(b"jpeg\x00") + len(b"jpeg\x00")
        marker += (-marker) % 4
        jpeg_size = int.from_bytes(payload[marker:marker + 4], "little")
        jpeg = payload[marker + 4:marker + 4 + jpeg_size]
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        with Image.open(BytesIO(jpeg)) as image:
            self.assertEqual(image.size, (12, 8))
            self.assertEqual(image.format, "JPEG")


def _extract_video_payload(data):
    """按 Foxglove CompressedVideo 的 CDR 布局提取 Annex-B 帧载荷。"""
    import struct
    offset = 4
    offset += 8  # timestamp
    frame_id_size = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + frame_id_size
    offset += (-(offset - 4)) % 4
    payload_size = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    payload = data[offset:offset + payload_size]
    offset += payload_size
    offset += (-(offset - 4)) % 4
    format_size = struct.unpack_from("<I", data, offset)[0]
    format_value = data[offset + 4:offset + 4 + format_size]
    if format_value != b"h264\x00":
        raise AssertionError(f"unexpected video format: {format_value!r}")
    return payload


if __name__ == "__main__":
    unittest.main()
