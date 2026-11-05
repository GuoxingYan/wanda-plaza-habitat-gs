"""裁剪并转换 Wanda Plaza 资产。

mesh 和 Gaussian 使用同一个源坐标包围盒；mesh 三角面要求三个顶点都在范围内，
高斯中心使用相同范围，避免两种资产的可见区域不一致。
"""
import argparse, json
from pathlib import Path
import numpy as np

R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

def header(path):
    with open(path, "rb") as f:
        lines = []
        while True:
            line = f.readline().decode("ascii"); lines.append(line)
            if line.strip() == "end_header": return lines, f.tell()

def replace_element(lines, element, count):
    return "".join(f"element {element} {count}\n" if line.startswith(f"element {element} ") else line for line in lines)

def basis(d):
    x, y, z = d.T; xx, yy, zz = x*x, y*y, z*z
    return np.column_stack([-.4886025119029199*y, .4886025119029199*z, -.4886025119029199*x,
        1.0925484305920792*x*y, -1.0925484305920792*y*z, .31539156525252005*(2*zz-xx-yy),
        -1.0925484305920792*x*z, .5462742152960396*(xx-yy), -.5900435899266435*y*(3*xx-yy),
        2.890611442640554*x*y*z, -.4570457994644658*y*(4*zz-xx-yy),
        .3731763325901154*z*(2*zz-3*xx-3*yy), -.4570457994644658*x*(4*zz-xx-yy),
        1.445305721320277*z*(xx-yy), -.5900435899266435*x*(xx-3*yy)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/data/projects/CloudRobo/Wanda Plaza"))
    ap.add_argument("--out", type=Path, default=Path("/data/projects/CloudRobo/deployment/scene_crop"))
    ap.add_argument("--crop", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"), default=[-50, 10, -40, 20])
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    xmin, xmax, ymin, ymax = args.crop

    lines, off = header(args.input / "mesh.ply")
    nv = int(next(x.split()[2] for x in lines if x.startswith("element vertex")))
    nf = int(next(x.split()[2] for x in lines if x.startswith("element face")))
    vd = np.dtype([("p", "<f8", (3,)), ("c", "u1", (3,))]); fd = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    v = np.memmap(args.input / "mesh.ply", mode="r", offset=off, dtype=vd, shape=(nv,))
    faces = np.memmap(args.input / "mesh.ply", mode="r", offset=off + nv*vd.itemsize, dtype=fd, shape=(nf,))
    assert np.all(faces["n"] == 3)
    # 只有三个顶点都在包围盒内的三角面才会保留。
    in_box = ((v["p"][:, 0] >= xmin) & (v["p"][:, 0] <= xmax) & (v["p"][:, 1] >= ymin) & (v["p"][:, 1] <= ymax))
    kept_faces = faces[np.all(in_box[faces["i"]], axis=1)]
    used = np.unique(kept_faces["i"].reshape(-1)); remap = np.full(nv, -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    out_faces = np.empty(len(kept_faces), dtype=fd); out_faces["n"] = 3; out_faces["i"] = remap[kept_faces["i"]]
    mesh = args.out / "wanda_crop_mesh_yup.ply"
    mesh_header = replace_element(lines, "vertex", len(used))
    mesh_header = replace_element(mesh_header.splitlines(keepends=True), "face", len(out_faces)).replace("property double", "property float")
    with open(mesh, "wb") as f:
        f.write(mesh_header.encode())
        for start in range(0, len(used), 200000):
            chunk = v[used[start:start+200000]]; out = np.empty(len(chunk), dtype=[("p", "<f4", (3,)), ("c", "u1", (3,))])
            out["p"] = chunk["p"] @ R.T; out["c"] = chunk["c"]; out.tofile(f)
        out_faces.tofile(f)
    print(f"Mesh crop: {len(used):,} vertices, {len(out_faces):,} triangles", flush=True)

    lines, off = header(args.input / "point_cloud.ply")
    ng = int(next(x.split()[2] for x in lines if x.startswith("element vertex")))
    props = [x.split()[-1] for x in lines if x.startswith("property")]; assert len(props) == 62
    data = np.memmap(args.input / "point_cloud.ply", mode="r", offset=off, dtype="<f4", shape=(ng, len(props)))
    # Gaussian 中心复用完全相同的四个边界参数。
    gs_keep = ((data[:, 0] >= xmin) & (data[:, 0] <= xmax) & (data[:, 1] >= ymin) & (data[:, 1] <= ymax))
    count = int(np.count_nonzero(gs_keep)); dirs = np.random.default_rng(3).normal(size=(200, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    T = np.linalg.lstsq(basis(dirs @ R.T), basis(dirs), rcond=None)[0]
    sh_error = float(np.max(abs(basis(dirs @ R.T) @ T - basis(dirs)))); assert sh_error < 1e-12; T[np.abs(T) < 1e-12] = 0
    target = args.out / "wanda_crop.gs.ply"; written = 0
    with open(target, "wb") as f:
        f.write(replace_element(lines, "vertex", count).encode())
        for start in range(0, ng, 200000):
            source = data[start:start+200000]; mask = gs_keep[start:start+len(source)]
            if not mask.any(): continue
            a = np.array(source[mask])
            # 位置、法线和后面的四元数/SH 一起旋转，保证视角变化时颜色方向正确。
            a[:, :3] = a[:, :3] @ R.T; a[:, 3:6] = a[:, 3:6] @ R.T
            a[:, 9:54] = np.einsum("ij,ncj->nci", T.astype(np.float32), a[:, 9:54].reshape(-1, 3, 15)).reshape(-1, 45)
            w, x, y, z = a[:, 58:62].T.copy(); s = np.float32(2**-.5)
            a[:, 58], a[:, 59], a[:, 60], a[:, 61] = s*(w+x), s*(x-w), s*(y+z), s*(z-y)
            a.tofile(f); written += len(a); print(f"GS crop written: {written:,}/{count:,}", flush=True)
    assert written == count
    meta = {"source": str(args.input.resolve()), "crop_source_xyz": {"x": [xmin, xmax], "y": [ymin, ymax], "z": None},
        "crop_rule": "same XY bounds; mesh faces require all 3 vertices inside", "input_mesh_vertices": nv,
        "input_mesh_triangles": nf, "mesh_vertices": len(used), "mesh_triangles": len(out_faces),
        "input_gaussians": ng, "output_gaussians": count, "transform_source_to_habitat": R.tolist(), "sh_rotation_max_error": sh_error}
    (args.out / "preparation.json").write_text(json.dumps(meta, indent=2))
    stage = {"render_asset": str(target), "collision_asset": str(mesh), "up": [0, 1, 0], "front": [0, 0, -1], "units_to_meters": 1.0, "requires_lighting": False}
    (args.out / "wanda.stage_config.json").write_text(json.dumps(stage, indent=2)); print(json.dumps(meta, indent=2), flush=True)

if __name__ == "__main__": main()
