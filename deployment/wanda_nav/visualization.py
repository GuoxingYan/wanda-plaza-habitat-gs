"""彩色模型与可行走区域的统一坐标系俯视诊断图。"""
import numpy as np
def topdown_plots(pathfinder, points, frames, scene_dir, output_dir, config, island):
    """输出路径俯视图及彩色 mesh 与 NavMesh 的俯视叠加图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from matplotlib.collections import PolyCollection
    from plyfile import PlyData

    tris = np.asarray(pathfinder.build_navmesh_vertices(island)).reshape(-1, 3, 3)
    trajectory = np.asarray([frame[0] for frame in frames])
    if config["output"]["trajectory_topdown"]:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.add_collection(PolyCollection(tris[:, :, [0, 2]], facecolors="#d8e8dc", edgecolors="none"))
        ax.plot(trajectory[:, 0], trajectory[:, 2], color=config["visualization"]["trajectory_color"], linewidth=config["visualization"]["trajectory_linewidth"], label="轨迹")
        ax.scatter(*trajectory[0, [0, 2]], color="#1674d1", s=45, label="起点", zorder=4)
        ax.scatter(*trajectory[-1, [0, 2]], color="#e33c32", s=45, label="终点", zorder=4)
        ax.set_aspect("equal"); ax.set_xlabel("Habitat X (m)"); ax.set_ylabel("Habitat Z (m)")
        ax.legend(); ax.autoscale(); fig.tight_layout()
        fig.savefig(output_dir / "trajectory_topdown.png", dpi=config["visualization"]["dpi"]); plt.close(fig)
    if config["output"]["mesh_navmesh_overlay"]:
        mesh = PlyData.read(str(scene_dir / "wanda_crop_mesh_yup.ply"))["vertex"].data
        coords = np.column_stack([mesh["x"], mesh["z"]])
        colors = np.column_stack([mesh["red"], mesh["green"], mesh["blue"]]).astype(np.float32) / 255.0
        limit = config["output"]["mesh_max_points"]
        if len(coords) > limit:
            selected = np.random.default_rng(config["navigation"]["seed"]).choice(len(coords), limit, replace=False)
            coords, colors = coords[selected], colors[selected]
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=0.18, linewidths=0, rasterized=True)
        ax.add_collection(PolyCollection(tris[:, :, [0, 2]], facecolors=config["visualization"]["navmesh_color"], edgecolors="none", alpha=config["visualization"]["navmesh_alpha"]))
        ax.set_aspect("equal"); ax.set_xlabel("Habitat X (m)"); ax.set_ylabel("Habitat Z (m)")
        ax.set_title("彩色 Mesh 顶视图 + NavMesh（绿色半透明）")
        ax.autoscale(); fig.tight_layout()
        fig.savefig(output_dir / "mesh_navmesh_topdown.png", dpi=config["visualization"]["dpi"]); plt.close(fig)


