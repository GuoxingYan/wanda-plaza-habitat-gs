"""按 NavMesh 高度生成仅用于俯视诊断的 Gaussian 渲染副本。

原始全密度 PLY 不会被修改；该副本保留每个 Gaussian 的全部属性，只按中心
Y 坐标过滤高处点。导航 RGB 视频继续使用原始全密度场景。
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read_header(path):
    lines = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            lines.append(line)
            if line.strip() == b"end_header":
                return lines, stream.tell()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-y", type=float, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    args = parser.parse_args()

    lines, offset = read_header(args.input)
    vertex_line = next(line for line in lines if line.startswith(b"element vertex "))
    count = int(vertex_line.split()[2])
    properties = [line.split()[-1].decode("ascii") for line in lines
                  if line.startswith(b"property ")]
    if len(properties) != 62:
        raise ValueError(f"Gaussian PLY 属性数异常: {len(properties)}")
    data = np.memmap(args.input, mode="r", offset=offset, dtype="<f4",
                     shape=(count, len(properties)))
    keep_count = int(np.count_nonzero(data[:, 1] <= args.threshold_y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    replacement = f"element vertex {keep_count}\n".encode("ascii")
    header = b"".join(replacement if line.startswith(b"element vertex ") else line
                      for line in lines)
    written = 0
    with args.output.open("wb") as stream:
        stream.write(header)
        for start in range(0, count, 200_000):
            chunk = np.asarray(data[start:start + 200_000])
            chunk = chunk[chunk[:, 1] <= args.threshold_y]
            chunk.tofile(stream)
            written += len(chunk)
    if written != keep_count:
        raise RuntimeError(f"写入数量不一致: {written} != {keep_count}")

    scene_dir = args.scene_dir
    stage = json.loads((scene_dir / "wanda.stage_config.json").read_text(encoding="utf-8"))
    stage["render_asset"] = str(args.output.resolve())
    (args.output.parent / "wanda.stage_config.json").write_text(
        json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output.parent / "wanda.navmesh").write_bytes(
        (scene_dir / "wanda.navmesh").read_bytes())
    metadata = {
        "source_render_asset": str(args.input.resolve()),
        "output_render_asset": str(args.output.resolve()),
        "filter": "gaussian_center_y <= navmesh_walkable_max_y + margin",
        "threshold_y_m": args.threshold_y,
        "input_gaussians": count,
        "output_gaussians": keep_count,
        "removed_gaussians": count - keep_count,
        "removed_fraction": (count - keep_count) / count,
        "diagnostic_only": True,
    }
    (args.output.parent / "topdown_crop.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
