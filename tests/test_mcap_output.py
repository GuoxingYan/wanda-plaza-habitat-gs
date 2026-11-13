"""MCAP 轨迹归档的通道与同步消息回归测试。"""
import tempfile
import unittest

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages
from google.protobuf.descriptor_pb2 import FileDescriptorSet

from src.mcap_output import NavigationMcapWriter, depth_preview, sanitize_depth


class McapOutputTest(unittest.TestCase):
    def test_depth_holes_and_out_of_range_values_use_far_clip(self):
        """验证无效深度、非正深度及超远值都被替换为远裁剪距离。"""
        values = np.array([[1.25, np.nan, np.inf], [0.0, -1.0, 600.0]], dtype=np.float32)
        result = sanitize_depth(values, 500.0)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, [[1.25, 500.0, 500.0], [500.0, 500.0, 500.0]])

    def test_depth_preview_maps_configured_metric_range(self):
        """验证深度预览按配置米制范围映射为 8 位灰度且不处理原数组。"""
        values = np.array([[0.0, 10.0, 20.0, 30.0]], dtype=np.float32)
        preview = depth_preview(values, 0.0, 20.0)
        np.testing.assert_array_equal(preview, [[0, 128, 255, 255]])
        np.testing.assert_array_equal(values, [[0.0, 10.0, 20.0, 30.0]])

    def test_writes_rgb_depth_pose_camera_info_and_default_tag(self):
        with tempfile.NamedTemporaryFile(suffix=".mcap") as stream:
            writer = NavigationMcapWriter(
                stream.name, "route_0000",
                {"width": 4, "height": 3, "fx": 2.0, "fy": 2.0,
                 "cx": 1.5, "cy": 1.0},
                metadata={"depth_preview": {"enabled": True, "min_m": 0.0,
                                             "max_m": 20.0, "encoding": "mono8"}})
            stamp = writer.timestamp_ns(0)
            rgb = np.zeros((3, 4, 3), dtype=np.uint8)
            depth = np.full((3, 4), 2.5, dtype=np.float32)
            writer.add_frame(stamp, rgb, depth, [1.0, 2.0, 3.0],
                             [0.0, 0.0, 0.0, 1.0], 0)
            writer.finish()

            stream.seek(0)
            reader = make_reader(stream)
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
            topics = [channel.topic for _, channel, _ in messages]
            self.assertEqual(set(topics), {
                "/trajectory/metadata", "/camera/rgb", "/camera/depth",
                "/camera/depth_preview", "/camera/info", "/trajectory/pose",
                "/trajectory/tags"})
            self.assertEqual(topics.count("/camera/rgb"), 1)
            self.assertEqual(topics.count("/camera/depth"), 1)
            self.assertEqual(topics.count("/camera/depth_preview"), 1)
            tag = next(message.data for _, channel, message in messages
                       if channel.topic == "/trajectory/tags")
            self.assertIn(b"straight_forward", tag)
            self.assertIn(b"route_0000", tag)
            image = next(message.data for _, channel, message in messages
                         if channel.topic == "/camera/rgb")
            self.assertTrue(image.startswith(b"\x00\x01\x00\x00"))
            self.assertIn(b"rgb8\x00", image)
            self.assertIn(np.zeros((3, 4, 3), dtype=np.uint8).tobytes(), image)
            decoded = list(read_ros2_messages(stream.name, topics=[
                "/camera/rgb", "/camera/depth", "/camera/depth_preview",
                "/camera/info", "/trajectory/pose"]))
            self.assertEqual(len(decoded), 5)
            decoded_by_topic = {message.channel.topic: message.ros_msg for message in decoded}
            self.assertEqual(decoded_by_topic["/camera/rgb"].encoding, "rgb8")
            self.assertEqual(decoded_by_topic["/camera/depth"].encoding, "32FC1")
            self.assertEqual(decoded_by_topic["/camera/depth_preview"].encoding, "mono8")
            np.testing.assert_array_equal(
                np.frombuffer(decoded_by_topic["/camera/depth_preview"].data, dtype=np.uint8),
                np.full(12, 32, dtype=np.uint8))
            self.assertEqual(decoded_by_topic["/camera/info"].width, 4)
            self.assertEqual(decoded_by_topic["/trajectory/pose"].pose.position.x, 1.0)
            stream.seek(0)
            descriptor = next(schema.data for schema in make_reader(stream).get_summary().schemas.values()
                              if schema.encoding == "protobuf")
            parsed = FileDescriptorSet.FromString(descriptor)
            self.assertEqual(parsed.file[0].message_type[0].name, "TrajectoryTags")
            for _, channel, message in messages:
                if channel.topic in ("/camera/rgb", "/camera/depth", "/trajectory/pose"):
                    self.assertGreater(message.log_time, 0)


if __name__ == "__main__":
    unittest.main()
