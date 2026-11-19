#!/usr/bin/env python3
# 中文说明：按 XY 递归切分 SanyaPo 的 mesh、Gaussian 和相机轨迹，并缓存切分结果。
"""按 XY 递归切分 SanyaPo 的 mesh、Gaussian 和相机轨迹并落盘缓存。"""
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def read_header(path):
    """读取二进制 PLY 头、数据偏移、元素数量和属性定义。"""
    lines = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline().decode("ascii")
            lines.append(line)
            if line.strip() == "end_header":
                return lines, stream.tell()


def scan_binary_faces(path, offset, vertex_count, bounds, vertices_xy, output_path=None):
    """分块扫描三角面并按质心归属 tile，保留跨 tile 面所需的全部顶点。"""
    output_path = output_path or Path("/dev/null")
    total = 0
    write_file = output_path.open("wb")
    try:
        with path.open("rb") as source:
            source.seek(offset)
            buffer = bytearray()
            cursor = 0
            while True:
                block = source.read(16 * 1024 * 1024)
                if not block:
                    break
                buffer.extend(block)
                stop = len(buffer) - (len(buffer) % 13)
                if stop == 0:
                    continue
                records = np.frombuffer(bytes(buffer[:stop]),
                                        dtype=np.dtype([("n", "u1"), ("i", "<i4", 3)]))
                indices = records["i"]
                valid = ((records["n"] == 3) & (indices.min(axis=1) >= 0) &
                         (indices.max(axis=1) < vertex_count))
                # 三角面按质心唯一归属 tile，避免重叠边界重复输出。
                centroid = vertices_xy[indices.clip(0, vertex_count - 1)].mean(axis=1)
                xmin, xmax, ymin, ymax = bounds
                valid &= ((centroid[:, 0] >= xmin) & (centroid[:, 0] <= xmax) &
                          (centroid[:, 1] >= ymin) & (centroid[:, 1] <= ymax))
                if valid.any():
                    kept = np.empty(int(valid.sum()), dtype=records.dtype)
                    kept["n"] = 3
                    kept["i"] = indices[valid]
                    kept.tofile(write_file)
                    total += len(kept)
                consumed_records = stop // 13
                cursor += consumed_records
                del records
                del buffer[:stop]
            if buffer:
                raise ValueError("mesh face 数据长度不是 13 字节定长三角形格式")
    finally:
        write_file.close()
    return total


def scalar_dtype(name):
    """将 PLY 标量类型转换为 little-endian NumPy 类型。"""
    return {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
            "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
            "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}[name]


def parse_elements(lines):
    """解析 PLY element/property 结构，支持固定标量属性和三角形列表面。"""
    elements = []
    current = None
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif len(parts) >= 3 and parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["properties"].append(("list", parts[2], parts[3], parts[4]))
            else:
                current["properties"].append(("scalar", parts[1], parts[2]))
    return elements


def replace_count(lines, element, count):
    """替换 PLY 头中指定 element 的数量。"""
    return [f"element {element} {count}\n" if line.startswith(f"element {element} ") else line
            for line in lines]


def bounds_for(points):
    """返回点集 XY 包围盒。"""
    return [float(points[:, 0].min()), float(points[:, 0].max()),
            float(points[:, 1].min()), float(points[:, 1].max())]


def split_boxes(points, bounds, limit, minimum_size):
    """按点云实际密度递归二分，直到每块高斯数不超限。"""
    inside = in_box(points[:, :2], bounds)
    count = int(inside.sum())
    xmin, xmax, ymin, ymax = bounds
    if count <= limit:
        return [bounds]
    if max(xmax - xmin, ymax - ymin) <= minimum_size:
        raise ValueError(f"空间块已小于 --min-size，但仍有 {count:,} 个 Gaussian 超过上限 {limit:,}")
    axis = 0 if (xmax - xmin) >= (ymax - ymin) else 1
    low, high = (xmin, xmax) if axis == 0 else (ymin, ymax)
    # 以高斯坐标中位数切分，保证密度偏斜时两侧仍有相近点数。
    values = points[inside, axis]
    pivot = float(np.partition(values, count // 2)[count // 2])
    if not low < pivot < high:
        pivot = (low + high) / 2.0
    if axis == 0:
        first, second = [xmin, pivot, ymin, ymax], [pivot, xmax, ymin, ymax]
    else:
        first, second = [xmin, xmax, ymin, pivot], [xmin, xmax, pivot, ymax]
    if pivot <= low or pivot >= high:
        raise ValueError("无法进一步切分 Gaussian 空间块")
    return (split_boxes(points, first, limit, minimum_size) +
            split_boxes(points, second, limit, minimum_size))


def in_box(points, box):
    """返回点是否位于空间块内，切分面使用左闭右开规则。"""
    xmin, xmax, ymin, ymax = box
    return ((points[:, 0] >= xmin) & (points[:, 0] < xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] < ymax))


def write_ply(path, lines, elements, arrays):
    """写出保留原始属性和顺序的二进制 little-endian PLY。"""
    counts = {name: len(arrays[name]) for name in arrays}
    output_lines = []
    for line in lines:
        if line.startswith("element "):
            name = line.split()[1]
            output_lines.append(f"element {name} {counts.get(name, next(e['count'] for e in elements if e['name'] == name))}\n")
        else:
            output_lines.append(line)
    with path.open("wb") as stream:
        stream.write("".join(output_lines).encode("ascii"))
        for element in elements:
            value = arrays.get(element["name"])
            if value is None:
                continue
            value.tofile(stream)


def main():
    """切分 SanyaPo 资产并写出可直接用于场景准备的缓存块。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/sanyapo"))
    parser.add_argument("--output", type=Path, default=Path("data/sanyapo/tiles"))
    parser.add_argument("--max-gaussians", type=int, default=25_000_000)
    parser.add_argument("--min-size", type=float, default=1.0)
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = Path(cached.get("source", ""))
        expected = {"mesh.ply": (args.input / "mesh.ply").stat().st_size,
                    "point_cloud.ply": (args.input / "point_cloud.ply").stat().st_size,
                    "cameras.json": (args.input / "cameras.json").stat().st_size}
        if source == args.input.resolve() and cached.get("source_sizes") == expected and all(
                (args.output / tile["directory"] / name).is_file()
                for tile in cached.get("tiles", [])
                for name in ("mesh.ply", "point_cloud.ply", "cameras.json", "bounds.json")):
            print(f"复用已有切分缓存: {args.output}", flush=True)
            return
    args.output.mkdir(parents=True, exist_ok=True)
    mesh_lines, mesh_offset = read_header(args.input / "mesh.ply")
    mesh_elements = parse_elements(mesh_lines)
    mesh_v = next(e for e in mesh_elements if e["name"] == "vertex")
    mesh_f = next(e for e in mesh_elements if e["name"] == "face")
    if mesh_f["properties"] != [("list", "uchar", "int", "vertex_indices")]:
        raise ValueError("mesh.ply 需要标准三角形 list uchar int vertex_indices 面格式")
    vertex_dtype = np.dtype([("xyz", "<f4", 3), ("rgba", "u1", 4)])
    vertices = np.memmap(args.input / "mesh.ply", mode="r", offset=mesh_offset,
                         dtype=vertex_dtype, shape=(mesh_v["count"],))
    point_lines, point_offset = read_header(args.input / "point_cloud.ply")
    point_elements = parse_elements(point_lines)
    point_v = next(e for e in point_elements if e["name"] == "vertex")
    point_props = point_v["properties"]
    if any(p[0] != "scalar" or scalar_dtype(p[1]) != "<f4" for p in point_props):
        raise ValueError("point_cloud.ply 当前要求所有 Gaussian 属性为 float32 标量")
    point_dtype = np.dtype([(p[2], "<f4") for p in point_props])
    points = np.memmap(args.input / "point_cloud.ply", mode="r", offset=point_offset,
                       dtype=point_dtype, shape=(point_v["count"],))
    source_bounds = bounds_for(vertices["xyz"])
    cameras = json.loads((args.input / "cameras.json").read_text(encoding="utf-8"))
    camera_points = np.asarray([camera["position"] for camera in cameras], dtype=np.float64)
    camera_bounds = bounds_for(camera_points)
    common_bounds = [min(source_bounds[0], camera_bounds[0]),
                     max(source_bounds[1], camera_bounds[1]),
                     min(source_bounds[2], camera_bounds[2]),
                     max(source_bounds[3], camera_bounds[3])]
    gaussian_xy = np.column_stack((points["x"], points["y"])).astype(np.float32, copy=False)
    gaussian_bounds = bounds_for(gaussian_xy)
    # 以 Gaussian 实际范围切分；mesh 和相机按 tile 范围同步落盘。
    boxes = split_boxes(gaussian_xy, gaussian_bounds, args.max_gaussians, args.min_size)
    manifest = {"source": str(args.input.resolve()), "max_gaussians": args.max_gaussians,
                "source_sizes": {name: (args.input / name).stat().st_size for name in
                                 ("mesh.ply", "point_cloud.ply", "cameras.json")},
                "source_bounds_xy": source_bounds, "camera_bounds_xy": camera_bounds,
                "gaussian_bounds_xy": gaussian_bounds, "tiles": []}
    for index, box in enumerate(boxes):
        p_mask = in_box(gaussian_xy, box)
        tile = args.output / f"tile_{index:03d}"
        tile.mkdir(exist_ok=True)
        face_path = tile / "faces.bin"
        face_count = scan_binary_faces(args.input / "mesh.ply",
                                       mesh_offset + mesh_v["count"] * vertex_dtype.itemsize,
                                       mesh_v["count"], box,
                                       vertices["xyz"][:, :2], face_path)
        face_dtype = np.dtype([("n", "u1"), ("i", "<i4", 3)])
        face_array = np.memmap(face_path, mode="r+", dtype=face_dtype, shape=(face_count,))
        selected_vertices = np.unique(face_array["i"].reshape(-1)) if face_count else np.array([], dtype=np.int64)
        remap = np.full(mesh_v["count"], -1, dtype=np.int32)
        remap[selected_vertices] = np.arange(len(selected_vertices), dtype=np.int32)
        face_array["i"] = remap[face_array["i"]]
        write_ply(tile / "mesh.ply", mesh_lines, mesh_elements,
                  {"vertex": np.asarray(vertices[selected_vertices]), "face": face_array})
        face_array._mmap.close()
        face_path.unlink()
        write_ply(tile / "point_cloud.ply", point_lines, point_elements,
                  {"vertex": np.asarray(points[p_mask])})
        tile_cameras = [c for c in cameras if in_box(np.asarray([c["position"][:2]], dtype=np.float64), box)[0]]
        (tile / "cameras.json").write_text(json.dumps(tile_cameras, ensure_ascii=False), encoding="utf-8")
        (tile / "bounds.json").write_text(json.dumps({"xy": box, "gaussians": int(p_mask.sum()),
                                                        "mesh_vertices": len(selected_vertices),
                                                        "mesh_faces": face_count, "cameras": len(tile_cameras)}, indent=2), encoding="utf-8")
        manifest["tiles"].append({"directory": tile.name, "xy": box, "gaussians": int(p_mask.sum()),
                                  "mesh_vertices": len(selected_vertices), "mesh_faces": face_count,
                                  "cameras": len(tile_cameras)})
        print(tile, manifest["tiles"][-1], flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
