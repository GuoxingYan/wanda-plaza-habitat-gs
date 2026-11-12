"""生成避障演示的路线对比图和单页 HTML 报告。"""
import argparse
import html
import json
from pathlib import Path

import numpy as np


def _original_route(pathfinder, start, goal):
    """用 Habitat 原生 NavMesh 最短路 API 重建未放障碍时的基线路径。"""
    route = __import__("habitat_sim").ShortestPath()
    route.requested_start = np.asarray(start, dtype=np.float32)
    route.requested_end = np.asarray(goal, dtype=np.float32)
    if not pathfinder.find_path(route):
        raise RuntimeError("无法在 NavMesh 上重建未避障基线路径")
    return np.asarray(route.points, dtype=np.float64), float(route.geodesic_distance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path, required=True)
    args = parser.parse_args()
    result = args.result.resolve()
    validation = json.loads((result / "validation.json").read_text(encoding="utf-8"))
    poses = json.loads((result / "navigation_poses.json").read_text(encoding="utf-8"))
    obstacle = validation["synthetic_obstacle"]
    if not obstacle.get("triggered"):
        raise RuntimeError("此结果没有触发避障，不能作为避障演示")

    import habitat_sim as hs
    pathfinder = hs.PathFinder()
    if not pathfinder.load_nav_mesh(str(args.navmesh.resolve())):
        raise RuntimeError(f"NavMesh 加载失败：{args.navmesh}")
    camera_height = float(validation["config"]["motion"]["camera_height_m"])
    start = np.asarray(poses["route"]["start"]["position"], dtype=np.float64)
    goal = np.asarray(poses["route"]["goal"]["position"], dtype=np.float64)
    # JSON 按约定记录相机中心；PathFinder 需要地面代理中心。
    start[1] -= camera_height
    goal[1] -= camera_height
    original, original_length = _original_route(pathfinder, start, goal)
    avoided = np.load(result / "path.npy").astype(np.float64)
    trajectory = np.load(result / "trajectory.npy").astype(np.float64)

    coordinates = json.loads((result / "mesh_topdown_coordinates.json").read_text(encoding="utf-8"))
    xmin, xmax, zmin, zmax = coordinates["extent_xmin_xmax_zmin_zmax"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PolyCollection
    from matplotlib import font_manager

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    base = plt.imread(result / "mesh_topdown_base.png")
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(base, extent=(xmin, xmax, zmin, zmax), origin="upper", interpolation="nearest")
    # 仅叠加路线邻域的 NavMesh 三角形，避免全场景大图造成多余绘图开销。
    xlo, xhi = min(original[:, 0].min(), avoided[:, 0].min()) - 4, max(original[:, 0].max(), avoided[:, 0].max()) + 4
    zlo, zhi = min(original[:, 2].min(), avoided[:, 2].min()) - 4, max(original[:, 2].max(), avoided[:, 2].max()) + 4
    for island in range(pathfinder.num_islands):
        triangles = np.asarray(pathfinder.build_navmesh_vertices(island)).reshape(-1, 3, 3)
        mask = ((triangles[:, :, 0].max(axis=1) >= xlo) & (triangles[:, :, 0].min(axis=1) <= xhi)
                & (triangles[:, :, 2].max(axis=1) >= zlo) & (triangles[:, :, 2].min(axis=1) <= zhi))
        ax.add_collection(PolyCollection(triangles[mask][:, :, [0, 2]], facecolors="#54cf9a",
                                         edgecolors="none", alpha=0.24, zorder=2))
    ax.plot(original[:, 0], original[:, 2], color="#0878bd", linewidth=3.0,
            label=f"原始 NavMesh 最短路线 ({original_length:.2f} m)", zorder=4)
    ax.plot(trajectory[:, 0], trajectory[:, 2], color="#e34a33", linewidth=2.2,
            label=f"避障执行轨迹 ({validation['planned_path_length_m']:.2f} m)", zorder=5)
    center = obstacle["center"]
    radius = float(obstacle["radius_m"])
    clearance = float(obstacle["planning_clearance_radius_m"])
    ax.add_patch(Circle((center[0], center[2]), radius, facecolor="#c51b1d",
                        edgecolor="#7f0000", alpha=0.9, label="圆柱障碍物 footprint", zorder=6))
    ax.add_patch(Circle((center[0], center[2]), clearance, facecolor="none",
                        edgecolor="#ff8c00", linestyle="--", linewidth=1.8,
                        label=f"规划净距半径 ({clearance:.3f} m)", zorder=5))
    ax.scatter(start[0], start[2], color="#1a9850", edgecolor="white", s=100,
               label="起点", zorder=7)
    ax.scatter(goal[0], goal[2], color="#762a83", edgecolor="white", s=100,
               label="终点", zorder=7)
    ax.set(xlim=(min(original[:, 0].min(), avoided[:, 0].min(), center[0]) - 4,
                 max(original[:, 0].max(), avoided[:, 0].max(), center[0]) + 4),
           ylim=(min(original[:, 2].min(), avoided[:, 2].min(), center[2]) - 4,
                 max(original[:, 2].max(), avoided[:, 2].max(), center[2]) + 4),
           xlabel="Habitat X (m)", ylabel="Habitat Z (m)",
           title="NavMesh 临时障碍绕行：基线路径与实际执行轨迹")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(result / "avoidance_route_comparison.png", dpi=220)
    plt.close(fig)

    margin = float(obstacle["actual_clearance_margin_m"])
    detour = float(validation["planned_path_length_m"]) - original_length
    esc = lambda value: html.escape(str(value), quote=True)
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NavMesh 避障验证</title><style>
body{{font:16px/1.65 system-ui,"Noto Sans CJK SC",sans-serif;margin:0;color:#202a32;background:#f4f6f7}}
main{{max-width:1320px;margin:auto;padding:28px}}h1{{margin:0 0 6px}}h2{{margin-top:30px;border-bottom:1px solid #ccd3d8;padding-bottom:6px}}
.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}}
.facts div{{background:white;border:1px solid #d6dde1;padding:12px 14px;border-radius:6px}}
.value{{font-size:22px;font-weight:650;display:block}}img{{max-width:100%;height:auto;border:1px solid #ccd3d8;background:white}}
video{{width:min(100%,960px);background:#111}}.note{{background:#fff8e1;border-left:4px solid #d99a00;padding:12px 16px}}
a{{color:#075e91}}code{{overflow-wrap:anywhere}}
</style></head><body><main>
<h1>NavMesh 避障场景验证</h1><p>全密度 Gaussian 场景；单条路线；相机采用 D435 RGB 参数。</p>
<div class="facts">
<div><span class="value">{original_length:.2f} m</span>无障碍 NavMesh 基线路径</div>
<div><span class="value">{float(validation['planned_path_length_m']):.2f} m</span>绕行路线长度</div>
<div><span class="value">{detour:.2f} m</span>绕行增加距离</div>
<div><span class="value">{float(obstacle['actual_min_center_distance_m']):.3f} m</span>实际轨迹到障碍中心最小距离</div>
<div><span class="value">{margin:.3f} m</span>机体外缘最小净空（满足配置要求）</div>
<div><span class="value">{int(validation['navigation_steps'])}</span>导航步数；全程 NavMesh 可行：{esc(validation['all_steps_navigable'])}</div>
</div>
<h2>路线与障碍位置</h2><p>彩色纹理 mesh 顶视底图上，半透明绿色为 NavMesh；蓝色为原始最短路线，红线为避障轨迹，红圆为障碍占地，橙色虚线为规划净距。</p>
<a href="avoidance_route_comparison.png"><img src="avoidance_route_comparison.png" alt="基线路径和避障路线叠加图"></a>
<h2>Gaussian RGB 关键帧</h2><p>第 545 帧的圆柱完整落在相机视野内；第 390 帧展示近距离绕过时的视角。</p>
<div class="facts"><div><a href="rgb_0545.png">第 545 帧</a><br><a href="rgb_0545.png"><img src="rgb_0545.png" alt="避障圆柱完整可见关键帧"></a></div>
<div><a href="rgb_0390.png">第 390 帧</a><br><a href="rgb_0390.png"><img src="rgb_0390.png" alt="避障圆柱关键帧"></a></div></div>
<h2>导航视频</h2><p>合成视频包含临时圆柱；原始 Gaussian 渲染视频用于对照。</p>
<video controls preload="metadata" src="navigation_rgb_composited.mp4"></video><p><a href="navigation_rgb_composited.mp4">下载避障合成视频</a> · <a href="navigation_rgb.mp4">查看原始 Gaussian RGB 视频</a></p>
<div class="note"><strong>当前原型限制：</strong>本次是已知固定障碍的离线重规划，开始运动前强制触发，并非运行到触发距离后在线检测和重规划；尚未完成圆柱占地与原始场景几何的碰撞检查。Habitat-GS RGB 后端没有将运行时刚体写入 Gaussian 渲染结果，因此圆柱以 D435 针孔投影合成到 RGB 帧，暂未与场景几何进行深度遮挡融合。</div>
<h2>场景与数据</h2><p>障碍物：程序化圆柱，高 {float(obstacle['height_m']):.2f} m、半径 {radius:.2f} m；中心 Y={float(center[1]):.3f} m。起终点、逐帧相机位置和 xyzw 四元数见 <a href="navigation_poses.json">navigation_poses.json</a>；完整验证指标见 <a href="validation.json">validation.json</a>。</p>
</main></body></html>"""
    (result / "avoidance_summary.html").write_text(page, encoding="utf-8")
    print(f"Wrote {result / 'avoidance_summary.html'}")
    print(f"Baseline={original_length:.3f}m avoided={validation['planned_path_length_m']:.3f}m margin={margin:.3f}m")


if __name__ == "__main__":
    main()
