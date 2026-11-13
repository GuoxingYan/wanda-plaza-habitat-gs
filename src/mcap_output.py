"""将同步 RGB、Depth、相机位姿和轨迹标签写入 ROS 2 风格 MCAP。"""
import json
from datetime import datetime, timezone
import struct
import time

from mcap.writer import CompressionType, Writer


def _align(buffer, alignment):
    padding = (-(len(buffer) - 4)) % alignment
    if padding:
        buffer.extend(b"\x00" * padding)


def _pack(buffer, fmt, *values, alignment=None):
    if alignment is None:
        alignment = min(struct.calcsize(fmt), 8)
    _align(buffer, alignment)
    buffer.extend(struct.pack("<" + fmt, *values))


def _string(buffer, value):
    encoded = value.encode("utf-8") + b"\x00"
    _pack(buffer, "I", len(encoded))
    buffer.extend(encoded)


def _sequence(buffer, values, fmt):
    _pack(buffer, "I", len(values))
    if fmt == "B" and isinstance(values, (bytes, bytearray, memoryview)):
        buffer.extend(values)
        return
    for value in values:
        _pack(buffer, fmt, value)


def _cdr(serialize):
    buffer = bytearray(b"\x00\x01\x00\x00")
    serialize(buffer)
    return bytes(buffer)


def _header(buffer, stamp_ns, frame_id):
    seconds, nanoseconds = divmod(int(stamp_ns), 1_000_000_000)
    _pack(buffer, "iI", seconds, nanoseconds, alignment=4)
    _string(buffer, frame_id)


def _image(stamp_ns, frame_id, rgb_or_depth, encoding, is_bigendian=0):
    import numpy as np

    image = np.ascontiguousarray(rgb_or_depth)
    height, width = image.shape[:2]
    if encoding == "rgb8":
        step = width * 3
    elif encoding == "mono8":
        step = width
    elif encoding == "32FC1":
        step = width * 4
    else:
        raise ValueError(f"unsupported image encoding: {encoding}")
    data = image.tobytes()

    def serialize(buffer):
        _header(buffer, stamp_ns, frame_id)
        _pack(buffer, "II", height, width, alignment=4)
        _string(buffer, encoding)
        _pack(buffer, "B", is_bigendian)
        _pack(buffer, "I", step)
        _sequence(buffer, data, "B")

    return _cdr(serialize)


def _pose(stamp_ns, frame_id, position, xyzw):
    def serialize(buffer):
        _header(buffer, stamp_ns, frame_id)
        for value in position:
            _pack(buffer, "d", float(value))
        for value in xyzw:
            _pack(buffer, "d", float(value))

    return _cdr(serialize)


def _camera_info(stamp_ns, frame_id, camera):
    width, height = int(camera["width"]), int(camera["height"])
    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx, cy = float(camera["cx"]), float(camera["cy"])

    def serialize(buffer):
        _header(buffer, stamp_ns, frame_id)
        _pack(buffer, "II", height, width, alignment=4)
        _string(buffer, "plumb_bob")
        _sequence(buffer, [0.0] * 5, "d")
        for value in [fx, 0, cx, 0, fy, cy, 0, 0, 1]:
            _pack(buffer, "d", value)
        for value in [1, 0, 0, 0, 1, 0, 0, 0, 1]:
            _pack(buffer, "d", value)
        for value in [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]:
            _pack(buffer, "d", value)
        _pack(buffer, "II", 0, 0, alignment=4)
        _pack(buffer, "IIII", 0, 0, 0, 0, alignment=4)
        _pack(buffer, "B", 0)

    return _cdr(serialize)


def _protobuf_string(field_number, value):
    payload = value.encode("utf-8")
    tag = (field_number << 3) | 2
    length = len(payload)

    def varint(number):
        out = bytearray()
        while number > 0x7F:
            out.append((number & 0x7F) | 0x80)
            number >>= 7
        out.append(number)
        return bytes(out)

    return varint(tag) + varint(length) + payload


def _protobuf_varint(number):
    """将非负整数编码为 Protobuf 变长整数。"""
    out = bytearray()
    while number > 0x7F:
        out.append((number & 0x7F) | 0x80)
        number >>= 7
    out.append(number)
    return bytes(out)


def _protobuf_bytes(field_number, payload):
    """编码 Protobuf 的长度限定字段及其字节内容。"""
    return (_protobuf_varint((field_number << 3) | 2) +
            _protobuf_varint(len(payload)) + payload)


def _protobuf_integer(field_number, value):
    """编码 Protobuf 的 varint 整数字段。"""
    return (_protobuf_varint(field_number << 3) +
            _protobuf_varint(value))


def _protobuf_field_descriptor(name, number, label):
    """构造字符串字段的 Protobuf 描述符字段。"""
    return (_protobuf_bytes(1, name.encode("utf-8")) +
            _protobuf_integer(3, number) +
            _protobuf_integer(4, label) +
            _protobuf_integer(5, 9))  # TYPE_STRING


def _trajectory_tags_descriptor():
    """构造 Foxglove 可解析的 protobuf FileDescriptorSet（二进制）。"""
    message = (_protobuf_bytes(1, b"TrajectoryTags") +
               _protobuf_bytes(2, _protobuf_field_descriptor("trajectory_id", 1, 1)) +
               _protobuf_bytes(2, _protobuf_field_descriptor("tags", 2, 3)))
    file_descriptor = (_protobuf_bytes(1, b"trajectory_tags.proto") +
                       _protobuf_bytes(2, b"wanda_nav") +
                       _protobuf_bytes(4, message) +
                       _protobuf_bytes(12, b"proto3"))
    return _protobuf_bytes(1, file_descriptor)


def sanitize_depth(depth, far_m):
    """将深度无效值和超过远裁剪面的值统一替换为远裁剪距离。"""
    import numpy as np

    far_value = float(far_m)
    if not np.isfinite(far_value) or far_value <= 0:
        raise ValueError("深度远裁剪距离必须为有限正数")
    depth = np.asarray(depth, dtype=np.float32)
    return np.where(np.isfinite(depth) & (depth > 0) & (depth <= far_value),
                    depth, np.float32(far_value)).astype(np.float32, copy=False)


def depth_preview(depth, min_m, max_m, encoding="mono8"):
    """把米制深度映射为便于 Foxglove 直接查看的 8 位预览图。

    该图仅用于可视化，不替代 ``/camera/depth`` 的精确深度。距离越近像素越暗，
    超出配置范围的值会被截断到范围端点，从而避免 Foxglove 按 0～1 的默认范围
    显示 ``32FC1`` 时整幅图看起来没有内容。
    """
    import numpy as np

    if encoding != "mono8":
        raise ValueError(f"暂不支持的深度预览编码: {encoding}")
    lower, upper = float(min_m), float(max_m)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("深度预览范围必须是有限值，且 max_m 大于 min_m")
    values = np.asarray(depth, dtype=np.float32)
    values = np.nan_to_num(values, nan=upper, posinf=upper, neginf=lower)
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


class NavigationMcapWriter:
    """以 ROS 2 CDR 图像/位姿消息和 Protobuf 标签写出一条路线。"""

    SCHEMAS = {
        "sensor_msgs/Image": """std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
""",
        "geometry_msgs/PoseStamped": """std_msgs/Header header
geometry_msgs/Pose pose
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
""",
        "sensor_msgs/CameraInfo": """std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: sensor_msgs/RegionOfInterest
uint32 x_offset
uint32 y_offset
uint32 height
uint32 width
bool do_rectify
""",
        "wanda_nav.TrajectoryTags": "binary FileDescriptorSet",
    }

    def __init__(self, path, trajectory_id, camera, metadata=None, start_time_ns=None):
        """初始化带 ROS2 标准消息、有效时间戳和 ZSTD 压缩的 MCAP 写入器。"""
        self.path = path
        self.trajectory_id = trajectory_id
        self.camera = camera
        self.depth_far_clip_m = float((metadata or {}).get("depth_far_clip_m", 500.0))
        self.start_time_ns = (int(start_time_ns) if start_time_ns is not None
                              else time.time_ns())
        self.started_utc = (datetime.fromtimestamp(self.start_time_ns // 1_000_000_000, timezone.utc)
                            if start_time_ns is not None
                            else datetime.now(timezone.utc))
        self.stream = open(path, "wb")
        self.writer = Writer(self.stream, compression=CompressionType.ZSTD)
        self.writer.start(profile="ros2", library="wanda-nav")
        self.channels = {}
        self.schemas = {}
        for name, text in self.SCHEMAS.items():
            encoding = "ros2msg" if name != "wanda_nav.TrajectoryTags" else "protobuf"
            data = (_trajectory_tags_descriptor() if encoding == "protobuf"
                    else text.encode("utf-8"))
            self.schemas[name] = self.writer.register_schema(name, encoding, data)
        for topic, name, encoding, ros2_type in (
            ("/camera/rgb", "sensor_msgs/Image", "cdr", "sensor_msgs/msg/Image"),
            ("/camera/depth", "sensor_msgs/Image", "cdr", "sensor_msgs/msg/Image"),
            ("/camera/info", "sensor_msgs/CameraInfo", "cdr", "sensor_msgs/msg/CameraInfo"),
            ("/trajectory/pose", "geometry_msgs/PoseStamped", "cdr", "geometry_msgs/msg/PoseStamped"),
            ("/trajectory/tags", "wanda_nav.TrajectoryTags", "protobuf", "wanda_nav.TrajectoryTags"),
        ):
            self.channels[topic] = self.writer.register_channel(
                topic, encoding, self.schemas[name],
                ({"ros2_type": ros2_type} if encoding == "cdr" else {}),
            )
        self.depth_preview_config = (metadata or {}).get("depth_preview", {})
        self.depth_preview_enabled = bool(self.depth_preview_config.get("enabled", True))
        self.depth_preview_min_m = float(self.depth_preview_config.get("min_m", 0.0))
        self.depth_preview_max_m = float(self.depth_preview_config.get("max_m", 20.0))
        self.depth_preview_encoding = str(self.depth_preview_config.get("encoding", "mono8"))
        if self.depth_preview_enabled:
            self.channels["/camera/depth_preview"] = self.writer.register_channel(
                "/camera/depth_preview", "cdr", self.schemas["sensor_msgs/Image"],
                {"ros2_type": "sensor_msgs/msg/Image"},
            )
        self.sequence = 0
        self._metadata_channel = None
        metadata_schema = self.writer.register_schema("NavigationMetadata", "jsonschema",
            json.dumps({"type": "object"}, separators=(",", ":")).encode("utf-8"))
        self._metadata_channel = self.writer.register_channel(
            "/trajectory/metadata", "json", metadata_schema)
        self.add_json(self.timestamp_ns(0), {"trajectory_id": trajectory_id,
                          "coordinate_frame": "Y-up", "position_unit": "m",
                          "orientation_format": "xyzw",
                          "timestamp_source": "UTC generation start + simulation time",
                          "generation_start_utc": self.started_utc.isoformat(),
                          "depth_encoding": "32FC1", "depth_unit": "m",
                          "depth_invalid_value": float((metadata or {}).get("depth_far_clip_m", 500.0)),
                          "depth_preview": {
                              "enabled": self.depth_preview_enabled,
                              "min_m": self.depth_preview_min_m,
                              "max_m": self.depth_preview_max_m,
                              "encoding": self.depth_preview_encoding,
                              "note": "仅用于可视化，不是可测量的原始深度",
                          },
                          **(metadata or {})})

    def timestamp_ns(self, simulation_seconds):
        """把相对模拟时间转换为路线开始时刻起算的 Unix 纳秒时间戳。"""
        return self.start_time_ns + int(round(float(simulation_seconds) * 1_000_000_000))

    def add_json(self, timestamp_ns, value):
        """写入路线级元数据，并使用与传感器消息相同的时间基准。"""
        self.writer.add_message(self._metadata_channel, timestamp_ns,
                                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                                timestamp_ns, self.sequence)
        self.sequence += 1

    def add_frame(self, timestamp_ns, rgb, depth, position, xyzw, index):
        """按同一时间戳写入 RGB、有限值深度和相机位姿。"""
        depth = sanitize_depth(depth, self.depth_far_clip_m)
        messages = (
            ("/camera/rgb", _image(timestamp_ns, "camera_color_optical_frame", rgb, "rgb8")),
            ("/camera/depth", _image(timestamp_ns, "camera_depth_optical_frame", depth, "32FC1")),
            ("/trajectory/pose", _pose(timestamp_ns, "world", position, xyzw)),
        )
        if self.depth_preview_enabled:
            preview = depth_preview(depth, self.depth_preview_min_m,
                                    self.depth_preview_max_m,
                                    self.depth_preview_encoding)
            messages += (("/camera/depth_preview",
                          _image(timestamp_ns, "camera_depth_optical_frame", preview,
                                 self.depth_preview_encoding)),)
        for topic, data in messages:
            self.writer.add_message(self.channels[topic], timestamp_ns, data,
                                    timestamp_ns, self.sequence)
            self.sequence += 1
        if index == 0:
            self.writer.add_message(self.channels["/camera/info"], timestamp_ns,
                _camera_info(timestamp_ns, "camera_color_optical_frame", self.camera),
                timestamp_ns, self.sequence)
            self.sequence += 1

    def finish(self):
        """追加默认路线标签并结束 MCAP 文件。"""
        tags = (_protobuf_string(1, self.trajectory_id) +
                _protobuf_string(2, "straight_forward"))
        timestamp_ns = self.timestamp_ns(0)
        self.writer.add_message(self.channels["/trajectory/tags"], timestamp_ns,
                                tags, timestamp_ns, self.sequence)
        self.writer.finish()
        self.stream.close()
