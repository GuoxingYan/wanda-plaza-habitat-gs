"""将同步 RGB、Depth、相机位姿和轨迹标签写入 ROS 2 风格 MCAP。"""
import json
from datetime import datetime, timezone
import os
import shutil
import struct
import subprocess
import tempfile
import time
import zlib
from io import BytesIO
from pathlib import Path

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
    elif encoding in ("32FC1", "16UC1"):
        step = width * (4 if encoding == "32FC1" else 2)
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


def _compressed_image(stamp_ns, frame_id, rgb, quality=90):
    """将 RGB 帧编码为 JPEG，并封装为 ROS 2 CompressedImage CDR 消息。"""
    import numpy as np
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB")
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=int(quality), optimize=True)
    payload = stream.getvalue()

    def serialize(buffer):
        _header(buffer, stamp_ns, frame_id)
        _string(buffer, "jpeg")
        _sequence(buffer, payload, "B")

    return _cdr(serialize)


def _split_annexb_access_units(data):
    """按 H.264 AUD NAL 单元切分 FFmpeg 输出的 Annex-B 视频帧。"""
    starts = []
    index = 0
    while index < len(data) - 4:
        if data[index:index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
        elif data[index:index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 1
        else:
            index += 1
    aud = [pos for pos, size in starts if (data[pos + size] & 0x1F) == 9]
    if not aud:
        raise RuntimeError("FFmpeg H.264 输出缺少 AUD，无法可靠映射逐帧时间戳")
    return [data[start:aud[index + 1] if index + 1 < len(aud) else len(data)]
            for index, start in enumerate(aud)]


def _foxglove_video_packet(timestamp_ns, data):
    """按 Foxglove ROS 2 消息定义封装一帧 Annex-B 码流。"""
    sec, nsec = divmod(int(timestamp_ns), 1_000_000_000)

    def serialize(buffer):
        """关键帧由 NAL 类型识别，不额外写私有字段。"""
        _pack(buffer, "iI", sec, nsec, alignment=4)
        _string(buffer, "camera_color_optical_frame")
        _pack(buffer, "I", len(data), alignment=4)
        buffer.extend(data)
        _string(buffer, "h264")

    return _cdr(serialize)


def _rgb_image(stamp_ns, frame_id, rgb):
    """将 RGB 帧编码为 ROS 2 sensor_msgs/Image 的 rgb8 消息。"""
    return _image(stamp_ns, frame_id, rgb, "rgb8")


def _quantize_depth(depth, scale_m=0.01, invalid_value=0, far_clip_m=None):
    """按米/整数比例量化深度，0 等无效值保留为配置的无效码。"""
    import numpy as np

    scale = float(scale_m)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth.scale_m 必须为有限正数")
    if not 0 <= int(invalid_value) <= 65535:
        raise ValueError("depth.invalid_value 必须位于 uint16 范围内")
    values = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0)
    if far_clip_m is not None:
        valid &= values <= float(far_clip_m)
    encoded = np.full(values.shape, np.uint16(invalid_value), dtype=np.uint16)
    scaled = np.rint(values[valid] / scale)
    max_value = 65535 if invalid_value != 65535 else 65534
    min_value = 1 if invalid_value == 0 else 0
    if np.any(scaled < min_value) or np.any(scaled > max_value):
        raise ValueError("深度值超出 uint16 量化范围")
    codes = scaled.astype(np.uint16)
    # 避开内部保留码时，选离原始值最近的相邻码，不能跳到量程端点。
    collision = codes == invalid_value
    prefer_lower = (invalid_value > 1) & (values[valid][collision] / scale < invalid_value)
    codes[collision] = np.where(prefer_lower, max(1, invalid_value - 1),
                                min(65535, invalid_value + 1)).astype(np.uint16)
    encoded[valid] = codes
    return encoded


def _compressed_depth(depth):
    """使用 D32Z 头和 zlib 无损编码 little-endian 米制 float32 深度。"""
    import numpy as np

    values = np.ascontiguousarray(depth, dtype="<f4")
    height, width = values.shape[:2]
    return struct.pack("<4sIII", b"D32Z", height, width, width * 4) + zlib.compress(
        values.tobytes(), level=6)


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
    """将米制深度映射为灰度预览；无效值显示为最远，不用于测距。"""
    import numpy as np

    lower, upper = float(min_m), float(max_m)
    if encoding != "mono8" or not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("深度预览只支持 mono8，且 max_m 必须大于有限的 min_m")
    values = sanitize_depth(depth, upper)
    return np.rint(np.clip((values - lower) / (upper - lower), 0, 1) * 255).astype(np.uint8)


def mcap_camera_options(camera, fps):
    """集中解析相机录制配置，同时接受已有的顶层 JPEG、预览和压缩选项。"""
    options = camera.get("mcap", {})
    rgb_options = options.get("rgb", {})
    return {
        "depth_far_clip_m": float(camera["far_m"]),
        "jpeg_quality": rgb_options.get("quality", camera.get("jpeg_quality", 90)),
        "rgb": {"encoding": rgb_options.get("encoding", "jpeg"),
                "quality": rgb_options.get("quality", camera.get("jpeg_quality", 90)),
                **rgb_options, "fps": float(fps)},
        "depth": options.get("depth", {}),
        "depth_preview": {**camera.get("depth_preview", {}), **options.get("depth_preview", {})},
        "depth_compression": {**camera.get("depth_compression", {}),
                              **options.get("depth_compression", {})},
    }


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
        "sensor_msgs/CompressedImage": """std_msgs/Header header
string format
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
        "foxglove_msgs/msg/CompressedVideo": """builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
""",
    }

    def __init__(self, path, trajectory_id, camera, metadata=None, start_time_ns=None,
                 tags=None):
        """初始化带 ROS2 标准消息、有效时间戳和 ZSTD 压缩的 MCAP 写入器。"""
        import numpy as np

        self.path = Path(path)
        self.trajectory_id = trajectory_id
        self.tags = list(tags or [])
        self.camera = camera
        options = metadata or {}
        self.rgb_config = options.get("rgb", {})
        self.depth_config = options.get("depth", {})
        self.rgb_encoding = str(self.rgb_config.get("encoding", "jpeg")).lower()
        if self.rgb_encoding not in ("jpeg", "h264"):
            raise ValueError("rgb.encoding 只支持 jpeg 或 h264")
        self.jpeg_quality = int(self.rgb_config.get("quality", options.get("jpeg_quality", 90)))
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality 必须位于 1 到 100 之间")
        self.depth_encoding = str(self.depth_config.get("encoding", "32FC1")).upper()
        if self.depth_encoding not in ("32FC1", "16UC1"):
            raise ValueError("depth.encoding 只支持 32FC1 或 16UC1")
        self.depth_scale_m = float(self.depth_config.get("scale_m", 0.01))
        self.depth_invalid_value = int(self.depth_config.get("invalid_value", 0))
        if self.depth_encoding == "16UC1":
            _quantize_depth(np.zeros((1, 1), dtype=np.float32), self.depth_scale_m,
                            self.depth_invalid_value)
        self.depth_far_clip_m = float(options.get("depth_far_clip_m", 500.0))
        self.video_crf = int(self.rgb_config.get("crf", 23))
        self.video_preset = str(self.rgb_config.get("preset", "fast"))
        self.video_gop = int(self.rgb_config.get("keyframe_interval", 30))
        self.video_fps = float(self.rgb_config.get("fps", 10.0))
        if not 0 <= self.video_crf <= 51:
            raise ValueError("rgb.crf 必须位于 0 到 51 之间")
        if self.video_gop <= 0 or not np.isfinite(self.video_fps) or self.video_fps <= 0:
            raise ValueError("rgb.keyframe_interval 和 rgb.fps 必须为正数")
        self.start_time_ns = (int(start_time_ns) if start_time_ns is not None
                              else time.time_ns())
        self.started_utc = (datetime.fromtimestamp(self.start_time_ns // 1_000_000_000, timezone.utc)
                            if start_time_ns is not None
                            else datetime.now(timezone.utc))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, self._partial_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".partial", dir=self.path.parent)
        self.stream = os.fdopen(fd, "wb")
        self._closed = False
        self.writer = Writer(self.stream, compression=CompressionType.ZSTD)
        self.writer.start(profile="ros2", library="wanda-nav")
        self.channels = {}
        self.schemas = {}
        for name, text in self.SCHEMAS.items():
            encoding = "ros2msg" if name != "wanda_nav.TrajectoryTags" else "protobuf"
            data = (_trajectory_tags_descriptor() if encoding == "protobuf"
                    else text.encode("utf-8"))
            self.schemas[name] = self.writer.register_schema(name, encoding, data)
        rgb_name = ("sensor_msgs/CompressedImage" if self.rgb_encoding == "jpeg"
                    else "foxglove_msgs/msg/CompressedVideo")
        for topic, name, encoding, ros2_type in (
            ("/camera/rgb", rgb_name, "cdr", "sensor_msgs/msg/CompressedImage" if self.rgb_encoding == "jpeg" else "foxglove_msgs/msg/CompressedVideo"),
            ("/camera/depth", "sensor_msgs/Image", "cdr", "sensor_msgs/msg/Image"),
            ("/camera/info", "sensor_msgs/CameraInfo", "cdr", "sensor_msgs/msg/CameraInfo"),
            ("/trajectory/pose", "geometry_msgs/PoseStamped", "cdr", "geometry_msgs/msg/PoseStamped"),
            ("/trajectory/tags", "wanda_nav.TrajectoryTags", "protobuf", "wanda_nav.TrajectoryTags"),
        ):
            self.channels[topic] = self.writer.register_channel(
                topic, encoding, self.schemas[name],
                ({"ros2_type": ros2_type} if encoding == "cdr" else {}),
            )
        self.depth_preview_config = options.get("depth_preview", {})
        self.depth_preview_enabled = bool(self.depth_preview_config.get("enabled", True))
        self.depth_preview_min_m = float(self.depth_preview_config.get("min_m", 0.0))
        self.depth_preview_max_m = float(self.depth_preview_config.get("max_m", 20.0))
        self.depth_preview_encoding = str(self.depth_preview_config.get("encoding", "mono8"))
        self.depth_compression_enabled = bool(options.get("depth_compression", {}).get("enabled", False))
        if self.depth_preview_enabled:
            self.channels["/camera/depth_preview"] = self.writer.register_channel(
                "/camera/depth_preview", "cdr", self.schemas["sensor_msgs/Image"],
                {"ros2_type": "sensor_msgs/msg/Image"})
        if self.depth_compression_enabled:
            schema_id = self.writer.register_schema(
                "wanda_nav/CompressedDepth", "binary",
                b"D32Z: magic[4], height:uint32, width:uint32, stride:uint32, zlib(float32_le[H,W])")
            self.channels["/camera/depth_compressed"] = self.writer.register_channel(
                "/camera/depth_compressed", "bytes", schema_id,
                {"format": "D32Z", "encoding": "32FC1", "compression": "zlib"})
        self.sequence = 0
        self._metadata_channel = None
        metadata_schema = self.writer.register_schema("NavigationMetadata", "jsonschema",
            json.dumps({"type": "object"}, separators=(",", ":")).encode("utf-8"))
        self._metadata_channel = self.writer.register_channel(
            "/trajectory/metadata", "json", metadata_schema)
        self.add_json(self.timestamp_ns(0), {**(metadata or {}),
                          "trajectory_id": trajectory_id,
                          "coordinate_frame": "Y-up", "position_unit": "m",
                          "orientation_format": "xyzw",
                          "timestamp_source": "UTC generation start + simulation time",
                          "generation_start_utc": self.started_utc.isoformat(),
                          "rgb": {"encoding": self.rgb_encoding,
                                  "codec": "h264" if self.rgb_encoding == "h264" else "jpeg",
                                  "crf": self.video_crf if self.rgb_encoding == "h264" else None,
                                  "fps": self.video_fps if self.rgb_encoding == "h264" else None},
                          "depth": {"encoding": self.depth_encoding,
                                    "scale_m": self.depth_scale_m if self.depth_encoding == "16UC1" else 1.0,
                                    "unit": "m",
                                    "invalid_value": self.depth_invalid_value if self.depth_encoding == "16UC1" else None,
                                    "depth_compression": self.depth_compression_enabled},
                          "depth_preview": {"enabled": self.depth_preview_enabled,
                                            "min_m": self.depth_preview_min_m,
                                            "max_m": self.depth_preview_max_m,
                                            "encoding": self.depth_preview_encoding},
                          })

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
        import numpy as np

        depth = np.asarray(depth, dtype=np.float32)
        if self.depth_encoding == "16UC1":
            depth_data = _quantize_depth(depth, self.depth_scale_m, self.depth_invalid_value,
                                         self.depth_far_clip_m)
        else:
            depth_data = sanitize_depth(depth, self.depth_far_clip_m)
        if self.rgb_encoding == "h264":
            if not hasattr(self, "_rgb_temp"):
                self._rgb_temp = tempfile.NamedTemporaryFile(prefix="wanda-mcap-rgb-", suffix=".rgb")
                self._rgb_shape = tuple(rgb.shape)
                height, width = self._rgb_shape[:2]
                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg is None:
                    raise RuntimeError("RGB H.264 编码需要可执行的 ffmpeg")
                self._video_process = subprocess.Popen(
                    [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-video_size", f"{width}x{height}", "-framerate", str(self.video_fps),
                     "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", self.video_preset,
                     "-crf", str(self.video_crf), "-g", str(self.video_gop), "-bf", "0",
                     "-x264-params", "aud=1:repeat-headers=1", "-pix_fmt", "yuv420p", "-f", "h264",
                     "pipe:1"], stdin=subprocess.PIPE, stdout=self._rgb_temp,
                    stderr=subprocess.PIPE)
                self._rgb_timestamps = []
            if tuple(rgb.shape) != self._rgb_shape:
                raise ValueError("H.264 编码期间 RGB 帧尺寸不可变化")
            self._video_process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
            self._rgb_timestamps.append(int(timestamp_ns))
        else:
            message = (_compressed_image(timestamp_ns, "camera_color_optical_frame", rgb,
                                         self.jpeg_quality)
                       if self.rgb_encoding == "jpeg"
                       else _rgb_image(timestamp_ns, "camera_color_optical_frame", rgb))
            self.writer.add_message(self.channels["/camera/rgb"], timestamp_ns,
                                    message, timestamp_ns, self.sequence)
            self.sequence += 1
        depth_message = _image(timestamp_ns, "camera_depth_optical_frame", depth_data,
                                self.depth_encoding)
        messages = (("/camera/depth", depth_message),
                    ("/trajectory/pose", _pose(timestamp_ns, "world", position, xyzw)))
        if self.depth_preview_enabled:
            preview = depth_preview(depth, self.depth_preview_min_m,
                                    self.depth_preview_max_m, self.depth_preview_encoding)
            messages += (("/camera/depth_preview",
                          _image(timestamp_ns, "camera_depth_optical_frame", preview,
                                 self.depth_preview_encoding)),)
        if self.depth_compression_enabled:
            messages += (("/camera/depth_compressed", _compressed_depth(depth)),)
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
        """追加调用方提供的路线标签并结束 MCAP 文件。"""
        if self._closed:
            return
        try:
            if self.rgb_encoding == "h264" and hasattr(self, "_rgb_temp"):
                self._video_process.stdin.close()
                stderr = self._video_process.stderr.read()
                if self._video_process.wait() != 0:
                    raise RuntimeError(f"FFmpeg H.264 编码失败：{stderr.decode(errors='replace')}")
                self._video_process.stderr.close()
                self._rgb_temp.seek(0)
                packets = _split_annexb_access_units(self._rgb_temp.read())
                if len(packets) != len(self._rgb_timestamps):
                    raise RuntimeError(f"H.264 帧数不匹配：{len(packets)} != {len(self._rgb_timestamps)}")
                for timestamp_ns, packet in zip(self._rgb_timestamps, packets):
                    message = _foxglove_video_packet(timestamp_ns, packet)
                    self.writer.add_message(self.channels["/camera/rgb"], timestamp_ns,
                                            message, timestamp_ns, self.sequence)
                    self.sequence += 1
                self._rgb_temp.close()
                self._rgb_temp = None
            tags = _protobuf_string(1, self.trajectory_id)
            for tag in self.tags:
                tags += _protobuf_string(2, tag)
            timestamp_ns = self.timestamp_ns(0)
            self.writer.add_message(self.channels["/trajectory/tags"], timestamp_ns,
                                    tags, timestamp_ns, self.sequence)
            self.writer.finish()
            self.stream.close()
            os.replace(self._partial_path, self.path)
            self._closed = True
        except Exception:
            self.abort()
            raise

    def abort(self):
        """中止录制并可靠回收 FFmpeg、临时帧文件及 MCAP 文件。"""
        if getattr(self, "_closed", False):
            return
        process = getattr(self, "_video_process", None)
        if process is not None and process.poll() is None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.stderr and not process.stderr.closed:
                process.stderr.close()
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
        rgb_temp = getattr(self, "_rgb_temp", None)
        if rgb_temp is not None:
            if not rgb_temp.closed:
                rgb_temp.close()
            os.unlink(rgb_temp.name) if os.path.exists(rgb_temp.name) else None
        if not self.stream.closed:
            self.stream.close()
        if os.path.exists(self._partial_path):
            os.unlink(self._partial_path)
        self._closed = True
