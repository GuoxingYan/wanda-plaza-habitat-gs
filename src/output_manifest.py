"""生成路线和批次产物索引，供汇总与检索时避免逐个读取 MCAP。"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.mcap_output import mcap_camera_options


def config_digest(config):
    """对生效配置计算稳定指纹，供续跑核对录制参数。"""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_atomic(path, value):
    """先写同目录临时文件再替换，避免中断留下半份 JSON。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, target)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def build_route_manifest(route_dir, config, report=None):
    """从已生成文件和配置快照提取路线索引，不扫描 MCAP 内容。"""
    route_dir = Path(route_dir)
    camera = config.get("camera", {})
    report = report or {}
    fps = config.get("motion", {}).get("fps")
    mcap_name = f"trajectory_{route_dir.name}.mcap"
    has_mcap = config.get("output", {}).get("mcap", True) and (route_dir / mcap_name).is_file()
    artifacts = [{"path": path.name, "size_bytes": path.stat().st_size}
                 for path in sorted(route_dir.iterdir())
                 if path.is_file() and not path.name.startswith(".")
                 and path.name not in {"manifest.json", "failure.json"}
                 and (path.suffix != ".mcap" or (has_mcap and path.name == mcap_name))]
    encoding = None
    if has_mcap:
        options = mcap_camera_options({"far_m": 500.0, **camera}, fps or 10)
        rgb, depth = options["rgb"], options["depth"]
        codec = str(rgb["encoding"]).lower()
        depth_format = str(depth.get("encoding", "32FC1")).upper()
        quantized = depth_format == "16UC1"
        encoding = {
            "rgb": {"codec": codec, "quality": int(rgb["quality"]) if codec == "jpeg" else None,
                    "crf": int(rgb.get("crf", 23)) if codec == "h264" else None},
            "depth": {"format": depth_format, "unit": "m",
                      "scale_m": float(depth.get("scale_m", 0.01)) if quantized else 1.0,
                      "invalid_value": int(depth.get("invalid_value", 0)) if quantized else options["depth_far_clip_m"]},
            "depth_preview": bool(options["depth_preview"].get("enabled", True)),
            "depth_compression": bool(options["depth_compression"].get("enabled", False)),
        }
    return {
        "schema_version": 1,
        "config_sha256": config_digest(config),
        "route_id": route_dir.name,
        "seed": config.get("navigation", {}).get("seed"),
        "tags": report.get("tags", []),
        "frame_count": report.get("rgb_frames"),
        "duration_s": ((max(0, int(report["rgb_frames"]) - 1) / float(fps))
                       if report.get("rgb_frames") is not None and fps else None),
        "camera": {"width": camera.get("width"), "height": camera.get("height"),
                   "fps": fps},
        "encoding": encoding,
        "mcap": mcap_name if has_mcap else None,
        "artifacts": artifacts,
    }


def write_batch_manifest(output_dir, route_dirs=None):
    """汇总现有路线 manifest；跳过损坏项并报告可用路线。"""
    output_dir = Path(output_dir)
    routes = []
    route_dirs = sorted(output_dir.glob("route_*")) if route_dirs is None else route_dirs
    for route_dir in route_dirs:
        if (route_dir / "failure.json").exists():
            continue
        try:
            route = json.loads((route_dir / "manifest.json").read_text(encoding="utf-8"))
            if route["schema_version"] != 1 or route["route_id"] != route_dir.name:
                continue
            prefix = route_dir.relative_to(output_dir)
            if route["mcap"] is not None:
                route["mcap"] = (prefix / route["mcap"]).as_posix()
            for artifact in route["artifacts"]:
                artifact["path"] = (prefix / artifact["path"]).as_posix()
            routes.append(route)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    write_json_atomic(output_dir / "manifest.json", {
        "schema_version": 1,
        "route_count": len(routes),
        "routes": routes,
    })
    return routes
