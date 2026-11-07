#!/usr/bin/env python3
"""将批量导航结果汇总为一个可直接打开的中文 HTML 报告。

图片以内嵌 data URI 保存，HTML 复制到别处仍可查看；视频保留为结果目录中的相对链接，
避免把 50 条 MP4 再复制一份导致报告异常膨胀。
"""
import argparse
import base64
import html
import json
from pathlib import Path


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def esc(value) -> str:
    return html.escape(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    routes = []
    for directory in sorted(args.results.glob("route_*")):
        report_file = directory / "validation.json"
        if not report_file.is_file():
            continue
        report = json.loads(report_file.read_text(encoding="utf-8"))
        routes.append((directory, report))
    if not routes:
        raise SystemExit(f"未找到有效路线: {args.results}")

    distances = [float(r.get("geodesic_distance_m", 0)) for _, r in routes]
    frames = [int(r.get("rgb_frames", 0)) for _, r in routes]
    valid = sum(bool(r.get("all_steps_navigable")) for _, r in routes)
    prior_ok = sum(bool(r.get("camera_trajectory_prior", {}).get("route_within_buffer"))
                   for _, r in routes)
    first = routes[0][1]
    cfg = first.get("config", {})
    motion = cfg.get("motion", {})
    prior = cfg.get("navigation", {})
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
        cards.append(f"""
        <article class="route">
          <div class="route-head"><h2>{esc(route)}</h2>
            <span class="badge {'ok' if report.get('all_steps_navigable') else 'bad'}">
              {'通过' if report.get('all_steps_navigable') else '失败'}
            </span></div>
          <div class="route-grid">
            <img class="map" loading="lazy" src="{data_uri(image)}" alt="{esc(route)} NavMesh 与轨迹">
            <div class="facts">
              <p>测地距离 <b>{report.get('geodesic_distance_m', 0):.2f} m</b></p>
              <p>RGB 帧数 <b>{report.get('rgb_frames', 0)}</b></p>
              <p>运行岛 <b>{report.get('route_island', '-')}</b></p>
              <p>目标误差 <b>{report.get('goal_error_m', 0):.3f} m</b></p>
              <p>轨迹缓冲区 <b>{'通过' if report.get('camera_trajectory_prior', {}).get('route_within_buffer') else '未通过'}</b></p>
              {f'<a class="video" href="{video_link}">打开导航 RGB 视频</a>' if video_link else ''}
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
.route-grid{{display:grid;grid-template-columns:minmax(0,3fr) minmax(180px,1fr);gap:16px;margin-top:12px}} .map{{width:100%;height:360px;object-fit:contain;background:#fafafa;border:1px solid #e2e8e3}} .facts{{font-size:14px;line-height:1.5}} .facts p{{margin:7px 0}} .video{{display:inline-block;margin-top:8px;color:#17653c}}
.thumbs{{display:flex;gap:8px;overflow:auto;margin-top:12px;padding-bottom:4px}} .thumb{{width:180px;height:102px;object-fit:cover;border:1px solid #dce5df}} footer{{padding:30px 5vw;color:#68776d;font-size:13px}}
@media(max-width:700px){{.route-grid{{grid-template-columns:1fr}} .map{{height:260px}}}}
</style></head><body><header><h1>{title}</h1>
<p>批量生产配置：线速度 {motion.get('linear_speed_mps', '-') } m/s，角速度 {motion.get('angular_speed_radps', '-')} rad/s，路线先验缓冲区 {prior.get('camera_trajectory_buffer_m', '-')} m。</p>
<p>结果目录：{esc(args.results)}；生成时间由本地文件系统记录。</p></header><main>
<section class="stats"><div class="stat"><b>{len(routes)}</b><span>路线总数</span></div><div class="stat"><b>{valid}/{len(routes)}</b><span>NavMesh 可行走验证</span></div><div class="stat"><b>{prior_ok}/{len(routes)}</b><span>采集轨迹缓冲区通过</span></div><div class="stat"><b>{sum(distances)/len(distances):.1f} m</b><span>平均测地距离</span></div><div class="stat"><b>{sum(frames)}</b><span>RGB 总帧数</span></div></section>
<p>每张大图显示全 NavMesh、当前路线起终点与实际运行轨迹；下方为关键 RGB 帧。绿色为 NavMesh 可行走区域，红线为路线，蓝点为起点，红点为终点。</p>
{''.join(cards)}</main><footer>由 Habitat-GS Wanda Plaza 导航验证流水线生成。HTML 内嵌 PNG，视频仍链接到同目录下的 MP4。</footer></body></html>"""
    args.output.write_text(doc, encoding="utf-8")
    print(f"已生成 {args.output}，包含 {len(routes)} 条路线")


if __name__ == "__main__":
    main()
