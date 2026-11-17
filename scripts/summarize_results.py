#!/usr/bin/env python3
"""将批量导航结果汇总为一个可直接打开的中文 HTML 报告。

图片以内嵌 data URI 保存，HTML 复制到别处仍可查看；视频保留为结果目录中的相对链接，
避免把 50 条 MP4 再复制一份导致报告异常膨胀。
"""
import argparse
import base64
import html
import json
import colorsys
import shutil
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def combined_route_map(routes: list[tuple[Path, dict]], output_path: Path) -> Path | None:
    """将所有路线以不同颜色叠加到同一张彩色 mesh 俯视底图。"""
    from PIL import Image, ImageDraw
    import numpy as np

    if not routes:
        return None
    first_dir = routes[0][0]
    base_path = first_dir / "mesh_topdown_base.png"
    coordinates_path = first_dir / "mesh_topdown_coordinates.json"
    if base_path.is_file() and coordinates_path.is_file():
        image = Image.open(base_path).convert("RGB")
        coordinates = json.loads(coordinates_path.read_text(encoding="utf-8"))
        extent = coordinates["extent_xmin_xmax_zmin_zmax"]
    else:
        config_path = first_dir / "effective_config.yaml"
        if not config_path.is_file():
            return None
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        stage_path = Path(config["scene"]["directory"]) / "wanda.stage_config.json"
        if not stage_path.is_absolute():
            stage_path = Path(__file__).resolve().parents[1] / stage_path
        if not stage_path.is_file():
            return None
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        mesh_path = Path(stage["collision_asset"])
        if not mesh_path.is_absolute():
            mesh_path = stage_path.parent / mesh_path
        from src.visualization import _load_static_base
        base, extent, _ = _load_static_base(
            mesh_path, config, Path(config["scene"]["directory"]))
        image = Image.fromarray(np.uint8(np.clip(base, 0, 1) * 255), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    xmin, xmax, zmin, zmax = extent
    width, height = image.size
    palette = [tuple(int(channel * 255) for channel in
                     colorsys.hsv_to_rgb(i / max(1, len(routes)), 0.82, 0.92))
               for i in range(len(routes))]

    def pixel(point):
        col = (float(point[0]) - xmin) / (xmax - xmin) * (width - 1)
        row = (zmax - float(point[2])) / (zmax - zmin) * (height - 1)
        return int(round(col)), int(round(row))

    for index, (directory, _) in enumerate(routes):
        trajectory_path = directory / "trajectory.npy"
        pose_payload = None
        if trajectory_path.is_file():
            trajectory = np.load(trajectory_path, mmap_mode="r")
        else:
            poses_path = directory / "navigation_poses.json"
            if not poses_path.is_file():
                continue
            pose_payload = json.loads(poses_path.read_text(encoding="utf-8"))
            trajectory = np.asarray([frame["position"] for frame in pose_payload["frames"]])
        if len(trajectory) < 2:
            continue
        color = palette[index]
        # 用白色描边提高不同路线在纹理底图上的辨识度。
        points = [pixel(point) for point in trajectory]
        draw.line(points, fill=(255, 255, 255, 225), width=9, joint="curve")
        draw.line(points, fill=(*color, 235), width=5, joint="curve")
        if pose_payload is None:
            poses_path = directory / "navigation_poses.json"
            if poses_path.is_file():
                pose_payload = json.loads(poses_path.read_text(encoding="utf-8"))
        if pose_payload and pose_payload.get("frames"):
            frames = pose_payload["frames"]
            frame_positions = np.asarray([frame["position"] for frame in frames],
                                         dtype=np.float64)
            cumulative = np.concatenate((
                [0.0], np.cumsum(np.linalg.norm(
                    np.diff(frame_positions[:, [0, 2]], axis=0), axis=1))))
            route_distance = float(cumulative[-1])
            sample_count = min(9, max(2, int(route_distance // 8.0)))
            targets = np.linspace(route_distance * 0.08,
                                  route_distance * 0.92, sample_count)
            sample_indices = np.unique(np.searchsorted(cumulative, targets).clip(
                0, len(frames) - 1))
            # 在航向持续变化的区段内补充转弯中段和结束处的箭头。
            quaternions = np.asarray([frame["xyzw"] for frame in frames],
                                     dtype=np.float64)
            xq, yq, zq, wq = quaternions.T
            yaw = np.unwrap(np.arctan2(2 * (wq * yq + xq * zq),
                                      1 - 2 * (yq * yq + zq * zq)))
            window = max(2, int(round(0.5 * 10)))
            turn_mask = np.zeros(len(frames), dtype=bool)
            for frame_index in range(window, len(frames) - window):
                delta_yaw = float(yaw[frame_index + window]
                                  - yaw[frame_index - window])
                turn_mask[frame_index] = abs(delta_yaw) >= np.deg2rad(20)
            turn_indices = np.flatnonzero(turn_mask)
            turn_runs = []
            if len(turn_indices):
                run_start = previous = int(turn_indices[0])
                for frame_index in turn_indices[1:]:
                    frame_index = int(frame_index)
                    if frame_index - previous > 15:
                        turn_runs.append((run_start, previous))
                        run_start = frame_index
                    previous = frame_index
                turn_runs.append((run_start, previous))
            turn_sample_indices = set()
            for run_start, run_end in turn_runs:
                net_turn = abs(float(yaw[run_end + window]
                                    - yaw[run_start - window]))
                if net_turn < np.deg2rad(25):
                    continue
                turn_sample_indices.add((run_start + run_end) // 2)
            sample_indices = np.asarray(sorted(set(sample_indices.tolist())
                                               | turn_sample_indices), dtype=int)
            regular_arrow_length = max(28, min(42, round(min(width, height) * 0.006)))
            for frame_index in sample_indices:
                frame = frames[int(frame_index)]
                x, y, z, w = frame["xyzw"]
                # Habitat 代理局部前向为 -Z；四元数旋转后投影到世界 X/Z 平面。
                forward_x = -2.0 * (x * z + w * y)
                forward_z = -(1.0 - 2.0 * (x * x + y * y))
                norm = float(np.hypot(forward_x, forward_z))
                if norm < 1e-8:
                    continue
                px, py = points[int(frame_index)]
                arrow_length = (44 if int(frame_index) in turn_sample_indices
                                else regular_arrow_length)
                unit = np.array([forward_x, -forward_z], dtype=np.float64) / norm
                side = np.array([-unit[1], unit[0]])

                def arrow_polygon(length, head_half_width, shaft_half_width):
                    """按统一箭杆与三角箭头头样式生成可缩放轮廓。"""
                    origin = np.array([px, py], dtype=np.float64)
                    tip = origin + unit * length
                    neck = origin + unit * (length * 0.48)
                    shoulder = neck + side * head_half_width
                    other_shoulder = neck - side * head_half_width
                    shaft_left = origin + side * shaft_half_width
                    shaft_right = origin - side * shaft_half_width
                    return [tuple(np.rint(point).astype(int)) for point in
                            (tip, shoulder, neck + side * shaft_half_width,
                             shaft_left, shaft_right, neck - side * shaft_half_width,
                             other_shoulder)]

                draw.polygon(arrow_polygon(arrow_length, arrow_length * 0.28,
                                           arrow_length * 0.13),
                             fill=(18, 24, 22, 245))
                draw.polygon(arrow_polygon(arrow_length * 0.88,
                                           arrow_length * 0.24,
                                           arrow_length * 0.085),
                             fill=(255, 255, 255, 255))
                draw.polygon(arrow_polygon(arrow_length * 0.76,
                                           arrow_length * 0.20,
                                           arrow_length * 0.05),
                             fill=(*color, 255))
        sx, sy = points[0]
        gx, gy = points[-1]
        draw.ellipse((sx - 8, sy - 8, sx + 8, sy + 8), fill=(22, 116, 209, 255),
                     outline=(255, 255, 255, 255), width=2)
        draw.polygon([(gx, gy - 10), (gx + 9, gy + 8), (gx - 9, gy + 8)],
                     fill=(255, 140, 0, 255), outline=(255, 255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def esc(value) -> str:
    return html.escape(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combined-map", type=Path,
                        help="可选的 50 条路线汇总图输出路径；默认写到 HTML 同目录")
    args = parser.parse_args()
    routes = []
    for directory in sorted(args.results.glob("route_*")):
        report_file = directory / "validation.json"
        report = (json.loads(report_file.read_text(encoding="utf-8"))
                  if report_file.is_file() else {})
        if not report:
            # 新存储规范不单独生成 validation.json；从逐帧位姿恢复页面需要的基本统计。
            poses_file = directory / "navigation_poses.json"
            if poses_file.is_file():
                poses = json.loads(poses_file.read_text(encoding="utf-8"))
                report = {
                    "geodesic_distance_m": poses.get("route", {}).get("path_length", 0),
                    "rgb_frames": len(poses.get("frames", [])),
                    "all_steps_navigable": None,
                    "route_island": "-",
                    "goal_error_m": 0,
                    "camera_trajectory_prior": {"route_within_buffer": None},
                }
        if report:
            routes.append((directory, report))
    if not routes:
        raise SystemExit(f"未找到有效路线: {args.results}")

    distances = [float(r.get("geodesic_distance_m", 0)) for _, r in routes]
    frames = [int(r.get("rgb_frames", 0)) for _, r in routes]
    valid = sum(r.get("all_steps_navigable") is True for _, r in routes)
    valid_unknown = sum(r.get("all_steps_navigable") is None for _, r in routes)
    prior_ok = sum(r.get("camera_trajectory_prior", {}).get("route_within_buffer") is True
                   for _, r in routes)
    prior_unknown = sum(r.get("camera_trajectory_prior", {}).get("route_within_buffer") is None
                        for _, r in routes)
    first = routes[0][1]
    cfg = first.get("config", {})
    motion = cfg.get("motion", {})
    prior = cfg.get("navigation", {})
    combined_path = args.combined_map or args.output.parent / "all_routes_mesh_overlay.png"
    combined_map = combined_route_map(routes, combined_path)
    route_legend = "".join(
        f'<span><i style="background:rgb({int(colorsys.hsv_to_rgb(i / max(1, len(routes)), 0.82, 0.92)[0] * 255)},'
        f'{int(colorsys.hsv_to_rgb(i / max(1, len(routes)), 0.82, 0.92)[1] * 255)},'
        f'{int(colorsys.hsv_to_rgb(i / max(1, len(routes)), 0.82, 0.92)[2] * 255)})"></i>{esc(directory.name)}</span>'
        for i, (directory, _) in enumerate(routes))
    combined_section = (f'<section class="combined"><h2>全部路线叠加总览</h2>'
                        f'<p>彩色 mesh 纹理底图上每种颜色对应一条路线；小箭头为沿路程抽样的车头朝向，大箭头标出拐点前后的朝向，不叠加 NavMesh；蓝色圆点为起点，橙色三角为终点。</p>'
                        f'<a href="{html.escape(combined_map.name)}"><img class="combined-map" '
                        f'src="{html.escape(combined_map.name)}" alt="全部路线叠加在 mesh 底图"></a>'
                        f'<div class="route-legend">{route_legend}</div></section>'
                        if combined_map else "")
    cards = []
    for directory, report in routes:
        route = directory.name
        image = directory / "trajectory_topdown.png"
        keyframes = sorted(directory.glob("rgb_*.png"))
        thumbs = "".join(
            f'<img class="thumb" loading="lazy" src="{data_uri(p)}" alt="{esc(route)} {esc(p.name)}">'
            for p in keyframes
        )
        video = directory / "navigation_rgb.mp4"
        # HTML 通常写在结果根目录；视频链接相对于 HTML 所在目录计算。
        video_link = html.escape(video.relative_to(args.output.parent).as_posix()) if video.is_file() else ""
        navigability = ("通过" if report.get("all_steps_navigable") is True else
                        "未通过" if report.get("all_steps_navigable") is False else "已在生成流程验证")
        buffer_status = ("通过" if report.get("camera_trajectory_prior", {}).get("route_within_buffer") is True else
                         "未通过" if report.get("camera_trajectory_prior", {}).get("route_within_buffer") is False else "未启用")
        map_html = (f'<img class="map" loading="lazy" src="{data_uri(image)}" alt="{esc(route)} NavMesh 与轨迹">'
                    if image.is_file() else "")
        mcap_file = next(iter(directory.glob("*.mcap")), None)
        mcap_link = (f'<a class="video" href="{html.escape(mcap_file.relative_to(args.output.parent).as_posix())}">下载 MCAP（含 RGB、Depth、Pose）</a>'
                     if mcap_file else "")
        cards.append(f"""
        <article class="route">
          <div class="route-head"><h2>{esc(route)}</h2>
            <span class="badge {'ok' if report.get('all_steps_navigable') is not False else 'bad'}">
              {navigability}
            </span></div>
          <div class="route-grid">
            {map_html}
            <div class="facts">
              <p>测地距离 <b>{report.get('geodesic_distance_m', 0):.2f} m</b></p>
              <p>RGB 帧数 <b>{report.get('rgb_frames', 0)}</b></p>
              <p>运行岛 <b>{report.get('route_island', '-')}</b></p>
              <p>目标误差 <b>{report.get('goal_error_m', 0):.3f} m</b></p>
              <p>轨迹缓冲区 <b>{buffer_status}</b></p>
              {f'<a class="video" href="{video_link}">打开导航 RGB 视频</a>' if video_link else ''}
              {mcap_link}
              <a class="video" href="{html.escape((directory / 'navigation_poses.json').relative_to(args.output.parent).as_posix())}">打开逐帧位姿 JSON</a>
            </div>
          </div>
          <div class="thumbs">{thumbs}</div>
        </article>""")

    title = "Wanda Plaza Habitat-GS 导航数据汇总"
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;margin:0;background:#f4f6f5;color:#17221c}}
header{{background:#173b2b;color:#fff;padding:28px 5vw}} h1{{margin:0 0 8px;font-size:28px}} header p{{margin:4px 0;color:#d6e8dc}}
main{{max-width:1500px;margin:24px auto;padding:0 24px}} .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px}}
.stat,.route{{background:#fff;border:1px solid #dce5df;border-radius:8px;box-shadow:0 2px 9px #1b432010}} .stat{{padding:14px 16px}} .stat b{{display:block;font-size:25px;color:#17653c}} .stat span{{font-size:13px;color:#627269}}
.route{{padding:18px;margin:18px 0}} .route-head{{display:flex;align-items:center;gap:10px}} h2{{margin:0;font-size:20px}} .badge{{font-size:12px;padding:3px 9px;border-radius:12px}} .ok{{background:#d8f2e0;color:#146331}} .bad{{background:#ffe0df;color:#a22b27}}
.combined{{background:#fff;border:1px solid #dce5df;border-radius:8px;padding:18px;margin-bottom:22px}} .route-grid{{display:grid;grid-template-columns:minmax(0,3fr) minmax(180px,1fr);gap:16px;margin-top:12px}} .map{{width:100%;height:360px;object-fit:contain;background:#fafafa;border:1px solid #e2e8e3}} .facts{{font-size:14px;line-height:1.5}} .facts p{{margin:7px 0}} .video{{display:inline-block;margin-top:8px;color:#17653c}}
.combined-map{{display:block;width:100%;max-height:900px;object-fit:contain;background:#fafafa;border:1px solid #e2e8e3}}
.route-legend{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:5px;margin-top:12px;font-size:12px}} .route-legend span{{display:flex;align-items:center;gap:6px}} .route-legend i{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.route-grid{{display:grid;grid-template-columns:minmax(0,3fr) minmax(180px,1fr);gap:16px;margin-top:12px}} .map{{width:100%;height:360px;object-fit:contain;background:#fafafa;border:1px solid #e2e8e3}} .facts{{font-size:14px;line-height:1.5}} .facts p{{margin:7px 0}} .video{{display:inline-block;margin-top:8px;color:#17653c}}
.thumbs{{display:flex;gap:8px;overflow:auto;margin-top:12px;padding-bottom:4px}} .thumb{{width:180px;height:102px;object-fit:cover;border:1px solid #dce5df}} footer{{padding:30px 5vw;color:#68776d;font-size:13px}}
@media(max-width:700px){{.route-grid{{grid-template-columns:1fr}} .map{{height:260px}}}}
</style></head><body><header><h1>{title}</h1>
<p>批量生产配置：线速度 {motion.get('linear_speed_mps', '-') } m/s，角速度 {motion.get('angular_speed_radps', '-')} rad/s，路线先验缓冲区 {prior.get('camera_trajectory_buffer_m', '-')} m。</p>
<p>结果目录：{esc(args.results)}；生成时间由本地文件系统记录。</p></header><main>
<section class="stats"><div class="stat"><b>{len(routes)}</b><span>路线总数</span></div><div class="stat"><b>{valid}/{len(routes)}</b><span>NavMesh 可行走验证{f'（另 {valid_unknown} 条由生成流程验证）' if valid_unknown else ''}</span></div><div class="stat"><b>{prior_ok}/{len(routes)}</b><span>采集轨迹缓冲区通过{f'（另 {prior_unknown} 条未启用）' if prior_unknown else ''}</span></div><div class="stat"><b>{sum(distances)/len(distances):.1f} m</b><span>平均测地距离</span></div><div class="stat"><b>{sum(frames)}</b><span>RGB 总帧数</span></div></section>
<p>顶部总览为彩色 mesh 纹理底图上的全部路线叠加，不绘制 NavMesh；下方为各路线统计和关键 RGB 帧。</p>
{combined_section}
{''.join(cards)}</main><footer>由 Habitat-GS Wanda Plaza 导航验证流水线生成。HTML 内嵌 PNG，视频仍链接到同目录下的 MP4。</footer></body></html>"""
    args.output.write_text(doc, encoding="utf-8")
    print(f"已生成 {args.output}，包含 {len(routes)} 条路线")


if __name__ == "__main__":
    main()
