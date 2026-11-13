#!/usr/bin/env python3
# 流式修复导航 MCAP 的类型定义、深度空洞和时间戳，同时保留有效传感器数据。
import argparse
import json
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

from src.mcap_output import (NavigationMcapWriter, _image, _trajectory_tags_descriptor,
                             depth_preview, sanitize_depth)


def repair(source: Path, destination: Path, preview_min_m: float = 0.0,
           preview_max_m: float = 20.0) -> None:
    """流式重建 schema 和 channel，并填补深度空洞后写入新的 MCAP。"""
    with source.open("rb") as input_stream:
        reader = make_reader(input_stream)
        summary = reader.get_summary()
        if summary is None:
            raise ValueError(f"MCAP 缺少可读索引: {source}")

        source_schemas = summary.schemas
        source_channels = summary.channels
        far_depth_m = 500.0
        for _, channel, message in reader.iter_messages(topics=["/trajectory/metadata"]):
            try:
                metadata = json.loads(message.data.decode("utf-8"))
                far_depth_m = float(metadata.get("depth_far_clip_m", far_depth_m))
                break
            except (ValueError, TypeError, UnicodeDecodeError):
                continue
        image_channel_ids = {channel.id for channel in source_channels.values()
                             if channel.topic == "/camera/rgb"}
        first_image_time = None
        for _, channel, message in reader.iter_messages(topics=["/camera/rgb"]):
            if channel.id in image_channel_ids:
                first_image_time = message.log_time
                break
        if first_image_time is None or first_image_time <= 0:
            raise ValueError("找不到有效的 RGB 起始时间戳")

        source_topics = {channel.topic for channel in source_channels.values()}
        add_preview = "/camera/depth_preview" not in source_topics

        input_stream.seek(0)
        reader = make_reader(input_stream)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output_stream:
            writer = Writer(output_stream, compression=CompressionType.ZSTD)
            writer.start(profile="ros2", library="wanda-nav mcap schema repair")

            schema_ids = {}
            for schema in source_schemas.values():
                schema_aliases = {
                    "Image": "sensor_msgs/Image",
                    "CameraInfo": "sensor_msgs/CameraInfo",
                    "PoseStamped": "geometry_msgs/PoseStamped",
                    "TrajectoryTags": "wanda_nav.TrajectoryTags",
                    "wanda_nav/msg/TrajectoryTags": "wanda_nav.TrajectoryTags",
                }
                name = schema_aliases.get(schema.name, schema.name)
                if name in NavigationMcapWriter.SCHEMAS:
                    if name == "wanda_nav.TrajectoryTags":
                        encoding, data = "protobuf", _trajectory_tags_descriptor()
                    else:
                        encoding = "ros2msg"
                        data = NavigationMcapWriter.SCHEMAS[name].encode("utf-8")
                else:
                    encoding, data = schema.encoding, schema.data
                schema_ids[schema.id] = writer.register_schema(name, encoding, data)

            channel_ids = {}
            for channel in source_channels.values():
                metadata = channel.metadata
                if channel.message_encoding == "protobuf":
                    metadata = {key: value for key, value in metadata.items()
                                if key != "ros2_type"}
                channel_ids[channel.id] = writer.register_channel(
                    channel.topic,
                    channel.message_encoding,
                    schema_ids.get(channel.schema_id, 0),
                    metadata,
                )

            preview_channel = None
            if add_preview:
                image_schema = next(schema_ids[schema.id] for schema in source_schemas.values()
                                    if schema.name in ("Image", "sensor_msgs/Image"))
                preview_channel = writer.register_channel(
                    "/camera/depth_preview", "cdr", image_schema,
                    {"ros2_type": "sensor_msgs/msg/Image"},
                )

            for schema, channel, message in reader.iter_messages(log_time_order=True):
                log_time = message.log_time
                publish_time = message.publish_time
                data = message.data
                if channel.topic == "/trajectory/metadata":
                    metadata = json.loads(data.decode("utf-8"))
                    metadata["timestamp_source"] = "UTC route start + simulation time"
                    metadata["depth_encoding"] = "32FC1"
                    metadata["depth_unit"] = "m"
                    metadata["depth_invalid_value"] = far_depth_m
                    metadata["depth_far_clip_m"] = far_depth_m
                    metadata["depth_preview"] = {
                        "enabled": True,
                        "min_m": float(preview_min_m),
                        "max_m": float(preview_max_m),
                        "encoding": "mono8",
                        "note": "仅用于可视化，不是可测量的原始深度",
                    }
                    data = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
                    if log_time <= 0:
                        log_time = first_image_time
                    if publish_time <= 0:
                        publish_time = first_image_time
                elif channel.topic == "/trajectory/tags":
                    # 轨迹级标签与起始帧关联，避免标签记录排在时间序列起点之前。
                    log_time = publish_time = first_image_time
                elif channel.topic == "/camera/depth":
                    data = _sanitize_depth_image_message(data, far_depth_m)
                    if preview_channel is not None:
                        height, width, frame_id, depth = _decode_depth_image_message(data)
                        preview = depth_preview(depth, preview_min_m, preview_max_m)
                        preview_data = _image(message.log_time, frame_id, preview, "mono8")
                        writer.add_message(preview_channel, log_time, preview_data,
                                           publish_time, message.sequence)
                writer.add_message(
                    channel_ids[channel.id], log_time, data, publish_time,
                    message.sequence,
                )
            writer.finish()


def _sanitize_depth_image_message(data: bytes, far_depth_m: float) -> bytes:
    """替换 ROS2 32FC1 深度消息中的无效像素，并保留其余 CDR 字段。"""
    import numpy as np

    payload = bytearray(data)
    if len(payload) < 4 or payload[:4] != b"\x00\x01\x00\x00":
        raise ValueError("深度消息不是受支持的小端 CDR 格式")
    offset = 4

    def align(size: int) -> None:
        """按 CDR 对齐规则推进当前消息游标。"""
        nonlocal offset
        offset += (-(offset - 4)) % size

    def read_u32() -> int:
        """读取一个按 CDR 对齐的 uint32 并推进游标。"""
        import struct

        nonlocal offset
        align(4)
        value = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        return value

    align(4)
    offset += 8  # Header 中的秒和纳秒
    align(4)
    frame_id_length = read_u32()
    offset += frame_id_length
    height = read_u32()
    width = read_u32()
    encoding_length = read_u32()
    encoding = bytes(payload[offset:offset + encoding_length - 1]).decode("ascii")
    offset += encoding_length
    offset += 1  # is_bigendian
    step = read_u32()
    data_length = read_u32()
    if encoding != "32FC1" or step != width * 4 or data_length != height * step:
        raise ValueError("深度消息字段与 32FC1 图像尺寸不一致")
    if offset + data_length != len(payload):
        raise ValueError("深度消息数据长度与 CDR payload 长度不一致")
    depth = np.frombuffer(payload, dtype="<f4", count=height * width, offset=offset)
    depth[:] = sanitize_depth(depth, far_depth_m)
    return bytes(payload)


def _decode_depth_image_message(data: bytes):
    """从已校验的 ROS2 32FC1 CDR 消息中读取尺寸、坐标系和深度数组。"""
    import numpy as np
    import struct

    payload = memoryview(data)
    offset = 4

    def align(size: int) -> None:
        """按 CDR 对齐规则推进当前消息游标。"""
        nonlocal offset
        offset += (-(offset - 4)) % size

    def read_u32() -> int:
        """读取一个按 CDR 对齐的 uint32 并推进游标。"""
        nonlocal offset
        align(4)
        value = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        return value

    align(4)
    offset += 8
    frame_id_length = read_u32()
    frame_id = bytes(payload[offset:offset + frame_id_length - 1]).decode("utf-8")
    offset += frame_id_length
    height, width = read_u32(), read_u32()
    encoding_length = read_u32()
    encoding = bytes(payload[offset:offset + encoding_length - 1]).decode("ascii")
    offset += encoding_length + 1
    step, data_length = read_u32(), read_u32()
    if encoding != "32FC1" or step != width * 4 or data_length != height * step:
        raise ValueError("深度消息字段与 32FC1 图像尺寸不一致")
    depth = np.frombuffer(payload, dtype="<f4", count=height * width,
                          offset=offset).reshape(height, width)
    return height, width, frame_id, depth


def main() -> None:
    """解析命令行参数并执行 MCAP 的流式修复。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--preview-min-m", type=float, default=0.0,
                        help="深度预览映射的最小距离（米）")
    parser.add_argument("--preview-max-m", type=float, default=20.0,
                        help="深度预览映射的最大距离（米）")
    args = parser.parse_args()
    repair(args.source, args.destination, args.preview_min_m, args.preview_max_m)
    print(f"已修复 {args.destination}")


if __name__ == "__main__":
    main()
