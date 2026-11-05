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

    # 俯视诊断必须显示所有连通岛；之前只绘制路线所在岛，会把右侧/边缘区域误判为“没有 NavMesh”。
    island_triangles = []
    for island_id in range(pathfinder.num_islands):
        vertices = np.asarray(pathfinder.build_navmesh_vertices(island_id)).reshape(-1, 3, 3)
        if len(vertices):
            island_triangles.append((island_id, vertices))
    tris = np.concatenate([vertices for _, vertices in island_triangles], axis=0)
    trajectory = np.asarray([frame[0] for frame in frames])
    if config["output"]["trajectory_topdown"]:
        fig, ax = plt.subplots(figsize=(10, 10))
        for island_id, island_tri in island_triangles:
            color = "#9bd3b0" if island_id == island else "#d8e8dc"
            alpha = 0.55 if island_id == island else 0.28
            ax.add_collection(PolyCollection(island_tri[:, :, [0, 2]], facecolors=color,
                                             edgecolors="none", alpha=alpha))
        ax.plot(trajectory[:, 0], trajectory[:, 2], color=config["visualization"]["trajectory_color"], linewidth=config["visualization"]["trajectory_linewidth"], label="轨迹")
        ax.scatter(*trajectory[0, [0, 2]], color="#1674d1", s=45, label="起点", zorder=4)
        ax.scatter(*trajectory[-1, [0, 2]], color="#e33c32", s=45, label="终点", zorder=4)
        ax.set_aspect("equal"); ax.set_xlabel("Habitat X (m)"); ax.set_ylabel("Habitat Z (m)")
        ax.legend(); ax.autoscale(); fig.tight_layout()
        fig.savefig(output_dir / "trajectory_topdown.png", dpi=config["visualization"]["dpi"]); plt.close(fig)
    if config["output"]["mesh_navmesh_overlay"]:
        import json
        from pathlib import Path
        stage = json.loads((scene_dir / "wanda.stage_config.json").read_text(encoding="utf-8"))
        mesh_path = Path(stage["collision_asset"])
        if not mesh_path.is_absolute():
            mesh_path = scene_dir / mesh_path
        mesh = PlyData.read(str(mesh_path))["vertex"].data
        coords = np.column_stack([mesh["x"], mesh["z"]])
        colors = np.column_stack([mesh["red"], mesh["green"], mesh["blue"]]).astype(np.float32) / 255.0
        limit = config["output"]["mesh_max_points"]
        if len(coords) > limit:
            selected = np.random.default_rng(config["navigation"]["seed"]).choice(len(coords), limit, replace=False)
            coords, colors = coords[selected], colors[selected]
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=0.18, linewidths=0, rasterized=True)
        for island_id, island_tri in island_triangles:
            alpha = config["visualization"]["navmesh_alpha"] if island_id == island else 0.12
            ax.add_collection(PolyCollection(island_tri[:, :, [0, 2]],
                                             facecolors=config["visualization"]["navmesh_color"],
                                             edgecolors="none", alpha=alpha))
        ax.set_aspect("equal"); ax.set_xlabel("Habitat X (m)"); ax.set_ylabel("Habitat Z (m)")
        ax.set_title("彩色 Mesh 顶视图 + NavMesh（绿色半透明）")
        ax.autoscale(); fig.tight_layout()
        fig.savefig(output_dir / "mesh_navmesh_topdown.png", dpi=config["visualization"]["dpi"]); plt.close(fig)


