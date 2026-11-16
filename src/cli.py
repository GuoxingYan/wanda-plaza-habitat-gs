"""批量执行入口：一次烘焙，按独立种子生成多条轨迹及执行配置快照。"""
import argparse
import copy
import json
from pathlib import Path
import re
import traceback
import yaml
from src.planning import load_config, resolve_path
from src.pipeline import run_one
import habitat_sim as hs
import magnum as mn


def required_route_outputs(output_config, navigation_config=None):
    """根据本次启用的输出类型返回续跑时必须存在的文件。"""
    required = ["validation.json"] if output_config.get("validation_json", True) else []
    if output_config.get("poses_json", True):
        required.append("navigation_poses.json")
    if output_config.get("mcap", False):
        required.append("*.mcap")
    for enabled, filename in (
        ("video", "navigation_rgb.mp4"),
        ("poses_json", "navigation_poses.json"),
        ("trajectory_topdown", "trajectory_topdown.png"),
        ("mesh_navmesh_overlay", "mesh_navmesh_topdown.png"),
        ("gaussian_navmesh_overlay", "gaussian_topdown_raw.png"),
        ("gaussian_navmesh_overlay", "gaussian_navmesh_topdown.png"),
    ):
        if output_config.get(enabled, False):
            required.append(filename)
    if (output_config.get("camera_trajectory_density_comparison", False)
            and (navigation_config or {}).get("use_camera_trajectory_prior", True)):
        required.append("camera_trajectory_density_comparison.png")
    return required


def is_route_complete(route_dir, report, config):
    """判断路线是否可续跑跳过，避免关闭视频时仍要求 RGB 帧统计。"""
    output_config = config["output"]
    if report.get("config") != config and not (Path(route_dir) / "effective_config.yaml").is_file():
        return False
    if (output_config.get("validation_json", True)
            and output_config.get("video", True) and report.get("rgb_frames", 0) <= 0):
        return False
    if output_config.get("poses_json", True):
        try:
            import json
            pose_data = json.loads((Path(route_dir) / "navigation_poses.json").read_text(encoding="utf-8"))
            endpoints = (pose_data["route"]["start"], pose_data["route"]["goal"])
            if (any("agent_position" in endpoint or "camera_position" in endpoint
                    or set(endpoint) != {"position", "xyzw"}
                    or "position" not in endpoint or "xyzw" not in endpoint
                    for endpoint in endpoints)
                    or any("agent_position" in frame or "camera_position" in frame
                           or "motion" in pose_data
                           or "position" not in frame or "xyzw" not in frame
                           for frame in pose_data["frames"])):
                return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
    required = required_route_outputs(output_config, config.get("navigation"))
    if output_config.get("mcap", False):
        required = [name for name in required if name != "*.mcap"]
        required.append(f"trajectory_{Path(route_dir).name}.mcap")
    return all((Path(route_dir) / name).is_file() for name in required)


def effective_config_text(config_path, config):
    """保留源 YAML 的逐项中文注释，只替换该路线实际使用的随机种子。"""
    text = Path(config_path).read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    in_navigation = False
    replaced = False
    for index, line in enumerate(lines):
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            in_navigation = line.split(":", 1)[0].strip() == "navigation"
        if in_navigation and re.match(r"^  seed\s*:", line):
            prefix, remainder = line.split(":", 1)
            value, marker, comment = remainder.partition("#")
            suffix = f"#{comment}" if marker else ""
            newline = "\n" if line.endswith("\n") else ""
            spacing = " " if suffix else ""
            lines[index] = f"{prefix}: {config['navigation']['seed']}{spacing}{suffix}{newline}"
            replaced = True
            break
    if not replaced:
        raise ValueError(f"配置文件中未找到 navigation.seed: {config_path}")
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument('--config', type=Path, default=project_root / 'configs' / 'default.yaml')
    parser.add_argument('--scene-dir', type=Path)
    parser.add_argument('--output', type=Path, default=project_root / 'outputs' / 'default')
    parser.add_argument('--skip-bake', action='store_true')
    parser.add_argument('--bake-only', action='store_true')
    parser.add_argument('--force', action='store_true', help='重建已完成的路线目录')
    parser.add_argument('--count', type=int, help='覆盖 YAML 中的轨迹条数')
    parser.add_argument('--single', action='store_true', help='只运行一条路线（用于完整场景预验证）')
    parser.add_argument('--visualize-only', action='store_true',
                        help='只刷新已有路线的俯视图，不重新生成视频或烘焙 NavMesh')
    parser.add_argument('--route', help='配合 --visualize-only 时只刷新指定 route_*/ 目录')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.visualize_only:
        refresh_visualizations(args, config)
        return
    count = args.count if args.count is not None else config['generation']['trajectory_count']
    if args.single:
        count = 1
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('generation.trajectory_count 必须为正整数')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.bake_only:
        run_one(args, config)
        return
    failures = []
    scene_dir = args.scene_dir or resolve_path(config['scene']['directory'])
    sim_config = hs.SimulatorConfiguration()
    sim_config.scene_id = str((scene_dir / 'wanda.stage_config.json').resolve())
    sim_config.enable_physics = False
    sim_config.load_semantic_mesh = False
    sensor = hs.CameraSensorSpec()
    sensor.uuid = 'rgb'
    sensor.sensor_type = hs.SensorType.COLOR
    sensor.resolution = [config['camera']['height'], config['camera']['width']]
    sensor.hfov = config['camera']['hfov_deg']
    sensor.position = mn.Vector3(0, config['motion']['camera_height_m'], 0)
    sensor.near = config['camera']['near_m']
    sensor.far = config['camera']['far_m']
    agent = hs.agent.AgentConfiguration()
    sensors = [sensor]
    if config['output'].get('mcap', False):
        depth_sensor = hs.CameraSensorSpec()
        depth_sensor.uuid = 'depth'
        depth_sensor.sensor_type = hs.SensorType.DEPTH
        depth_sensor.resolution = [config['camera']['height'], config['camera']['width']]
        depth_sensor.hfov = config['camera']['hfov_deg']
        depth_sensor.position = mn.Vector3(0, config['motion']['camera_height_m'], 0)
        depth_sensor.near = config['camera']['near_m']
        depth_sensor.far = config['camera']['far_m']
        sensors.append(depth_sensor)
    if config['output'].get('gaussian_navmesh_overlay', False) and not args.bake_only:
        import numpy as np
        # 使用 NavMesh 世界范围和最高点设置整批共用的高空俯视传感器。
        scene_pathfinder = hs.PathFinder()
        if not scene_pathfinder.load_nav_mesh(str(scene_dir / 'wanda.navmesh')):
            raise RuntimeError(f"无法为俯视相机读取 NavMesh: {scene_dir / 'wanda.navmesh'}")
        bounds = np.asarray(scene_pathfinder.get_bounds(), dtype=np.float32)
        xmin, zmin = bounds[0, 0], bounds[0, 2]
        xmax, zmax = bounds[1, 0], bounds[1, 2]
        visual = config['visualization']
        tile_size = float(visual.get('gaussian_topdown_tile_size_m', 64.0))
        tile_pixels = int(visual.get('gaussian_topdown_tile_resolution_px', 768))
        camera_height = float(visual.get('gaussian_topdown_camera_height_m', 8.0))
        above = float(visual.get('gaussian_topdown_height_above_walkable_m', 3.0))
        below = float(visual.get('gaussian_topdown_depth_below_walkable_m', 0.5))
        topdown = hs.CameraSensorSpec()
        topdown.uuid = 'topdown_rgb'; topdown.sensor_type = hs.SensorType.COLOR
        # Gaussian 渲染器对针孔路径支持更成熟；使用低高度局部透视 tile，
        # 避免整场景高空透视的浮点，也避免正交路径的方向性拉丝。
        topdown.sensor_subtype = hs.SensorSubType.PINHOLE
        topdown.resolution = [tile_pixels, tile_pixels]; topdown.position = [0.0, 0.0, 0.0]
        topdown.orientation = [-float(np.pi/2), 0.0, 0.0]
        topdown.hfov = float(np.degrees(2 * np.arctan(tile_size / (2 * camera_height))))
        topdown.near = max(0.01, camera_height - above)
        topdown.far = camera_height + below
        sensors.append(topdown)
    agent.sensor_specifications = sensors
    pending = []
    for index in range(count):
        current = copy.deepcopy(config)
        current['navigation']['seed'] += index
        current['generation']['batch_seed'] = int(config['navigation']['seed'])
        current['generation']['batch_size'] = int(count)
        current['generation']['batch_index'] = int(index)
        run_args = copy.copy(args)
        run_args.output = args.output / f'route_{index:04d}'
        # 批次使用同一份已验证 NavMesh；外部预先烘焙可避免反复初始化模拟器。
        run_args.skip_bake = True
        run_args.output.mkdir(parents=True, exist_ok=True)
        done_file = run_args.output / 'validation.json'
        report = json.loads(done_file.read_text()) if done_file.exists() else {}
        required = required_route_outputs(current['output'], current.get('navigation'))
        if (not args.force and not (run_args.output / 'failure.json').exists()
                and is_route_complete(run_args.output, report, current)):
            print(f'[{index + 1}/{count}] skip completed {run_args.output}', flush=True)
            continue
        # 每条结果保存实际生效配置，便于复现实验及查看可视化参数。
        (run_args.output / 'effective_config.yaml').write_text(
            effective_config_text(args.config, current), encoding='utf-8')
        pending.append((index, run_args, current))
    if pending:
        with hs.Simulator(hs.Configuration(sim_config, [agent])) as sim:
            for index, run_args, current in pending:
                print(f'[{index + 1}/{count}] start {run_args.output}', flush=True)
                try:
                    run_one(run_args, current, render_sim=sim)
                    (run_args.output / 'failure.json').unlink(missing_ok=True)
                except Exception as error:
                    failure = {'trajectory_index': index, 'output': str(run_args.output),
                               'error': repr(error), 'traceback': traceback.format_exc()}
                    failures.append(failure)
                    (run_args.output / 'failure.json').write_text(
                        json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(f'[{index + 1}/{count}] FAILED: {error}', flush=True)
    if failures:
        (args.output / 'failures.json').write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
        raise RuntimeError(f'{len(failures)} / {count} 条路线失败，详见 {args.output / "failures.json"}')
    (args.output / 'failures.json').unlink(missing_ok=True)


def refresh_visualizations(args, config):
    """基于已有 route_*/path.npy 刷新高分辨率可视化。"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import habitat_sim as hs
    import magnum as mn
    from PIL import Image
    from .visualization import (topdown_plots, save_gaussian_navmesh_overlay,
                                project_navmesh_world_to_topdown_pixels,
                                render_ground_sliced_gaussian_topdown)

    scene_dir = args.scene_dir or resolve_path(config['scene']['directory'])
    stage = json.loads((scene_dir / 'wanda.stage_config.json').read_text(encoding='utf-8'))
    navmesh_path = scene_dir / 'wanda.navmesh'
    pathfinder = hs.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh_path)):
        raise RuntimeError(f'无法载入 NavMesh: {navmesh_path}')
    scene_bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float32)
    xmin, zmin = scene_bounds[0, 0], scene_bounds[0, 2]
    xmax, zmax = scene_bounds[1, 0], scene_bounds[1, 2]
    triangles_by_island = []
    for island_id in range(pathfinder.num_islands):
        triangles = np.asarray(pathfinder.build_navmesh_vertices(island_id)).reshape(-1, 3, 3)
        if len(triangles):
            triangles_by_island.append((island_id, triangles))

    output_dirs = sorted(args.output.glob('route_*'))
    if not output_dirs:
        raise RuntimeError(f'没有找到路线输出目录: {args.output}')

    for output_dir in output_dirs:
        if args.route and output_dir.name != args.route:
            continue
        path_file = output_dir / 'path.npy'
        trajectory_file = output_dir / 'trajectory.npy'
        if not path_file.exists() or not trajectory_file.exists():
            continue
        path = np.load(path_file)
        trajectory = np.load(trajectory_file)
        route_config = config
        config_path = output_dir / 'effective_config.yaml'
        if config_path.exists():
            with config_path.open(encoding='utf-8') as stream:
                route_config = yaml.safe_load(stream)
        frames = [(point, 0.0, index / config['motion']['fps']) for index, point in enumerate(trajectory)]
        island = int(pathfinder.get_island(path[0]))
        topdown_plots(pathfinder, path, frames, scene_dir, output_dir, route_config, island)

        if route_config['output'].get('gaussian_navmesh_overlay', False):
            visual = route_config['visualization']
            tile_size = float(visual.get('gaussian_topdown_tile_size_m', 64.0))
            tile_pixels = int(visual.get('gaussian_topdown_tile_resolution_px', 768))
            camera_height = float(visual.get('gaussian_topdown_camera_height_m', 8.0))
            above = float(visual.get('gaussian_topdown_height_above_walkable_m', 3.0))
            below = float(visual.get('gaussian_topdown_depth_below_walkable_m', 0.5))
            topdown = hs.CameraSensorSpec()
            topdown.uuid = 'topdown_rgb'
            topdown.sensor_type = hs.SensorType.COLOR
            topdown.sensor_subtype = hs.SensorSubType.PINHOLE
            topdown.resolution = [tile_pixels, tile_pixels]
            topdown.position = [0.0, 0.0, 0.0]
            topdown.orientation = [-float(np.pi / 2), 0.0, 0.0]
            topdown.hfov = float(np.degrees(2 * np.arctan(tile_size / (2 * camera_height))))
            topdown.near = max(0.01, camera_height - above)
            topdown.far = camera_height + below
            agent = hs.agent.AgentConfiguration()
            agent.sensor_specifications = [topdown]
            sim_cfg = hs.SimulatorConfiguration()
            sim_cfg.scene_id = str((scene_dir / 'wanda.stage_config.json').resolve())
            sim_cfg.enable_physics = False
            sim_cfg.load_semantic_mesh = False
            sim_cfg.gpu_device_id = 0
            with hs.Simulator(hs.Configuration(sim_cfg, [agent])) as sim:
                rgb, extent = render_ground_sliced_gaussian_topdown(
                    sim, triangles_by_island, route_config, scene_bounds)
            Image.fromarray(rgb).save(output_dir / 'gaussian_topdown_raw.png')
            width = int(visual['image_width_px'])
            height = int(visual['image_height_px'])
            projected = [(idx, project_navmesh_world_to_topdown_pixels(tri, extent, width, height))
                         for idx, tri in triangles_by_island]
            save_gaussian_navmesh_overlay(rgb, projected,
                                          (-0.5, width-0.5, -0.5, height-0.5),
                                          output_dir / 'gaussian_navmesh_topdown.png', route_config)
        print(f'Refreshed visualizations: {output_dir}', flush=True)


if __name__ == '__main__':
    main()
