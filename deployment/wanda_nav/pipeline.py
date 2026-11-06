"""按 YAML 配置运行 Habitat-GS 的 NavMesh、导航与 Gaussian RGB 闭环验证。"""
import json
import math
import time
from pathlib import Path

import habitat_sim as hs
import imageio.v2 as imageio
import magnum as mn
import numpy as np
from PIL import Image

from .planning import (build_motion, choose_route, load_camera_trajectory,
                       quaternion_xyzw, resolve_path, route_within_camera_buffer,
                       yaw_quaternion)
from .visualization import topdown_plots, save_gaussian_navmesh_overlay, project_navmesh_to_topdown_pixels


def run_one(args, config, render_sim=None):
    scene_dir = args.scene_dir or resolve_path(config["scene"]["directory"])
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_config = json.loads((scene_dir / "wanda.stage_config.json").read_text(encoding="utf-8"))
    collision_mesh = Path(stage_config["collision_asset"])
    if not collision_mesh.is_absolute():
        collision_mesh = scene_dir / collision_mesh
    navmesh_path = scene_dir / "wanda.navmesh"
    report = {"config_file": str(args.config.resolve()), "scene_dir": str(scene_dir.resolve())}

    settings = hs.NavMeshSettings(); settings.set_defaults()
    nav = config["navmesh"]
    settings.cell_size, settings.cell_height = nav["cell_size_m"], nav["cell_height_m"]
    settings.agent_height, settings.agent_radius = nav["agent_height_m"], nav["agent_width_m"] / 2
    settings.agent_max_climb, settings.agent_max_slope = nav["max_climb_m"], nav["max_slope_deg"]
    if nav["edge_max_error_m"] > 0:
        settings.edge_max_error = nav["edge_max_error_m"]
    if config["scene"]["bake_navmesh"] and not args.skip_bake:
        simulator_cfg = hs.SimulatorConfiguration()
        simulator_cfg.scene_id = str(collision_mesh)
        simulator_cfg.create_renderer = False; simulator_cfg.load_semantic_mesh = False
        with hs.Simulator(hs.Configuration(simulator_cfg, [hs.agent.AgentConfiguration()])) as sim:
            start_time = time.time()
            assert sim.recompute_navmesh(sim.pathfinder, settings), "NavMesh bake failed"
            sim.pathfinder.save_nav_mesh(str(navmesh_path))
            report["bake_seconds"] = time.time() - start_time

    pathfinder = hs.PathFinder(); assert pathfinder.load_nav_mesh(str(navmesh_path))
    pathfinder.seed(config["navigation"]["seed"])
    assert pathfinder.navigable_area > 10
    largest_island = int(np.argmax([pathfinder.island_area(i) for i in range(pathfinder.num_islands)]))
    path, island = choose_route(pathfinder, config)
    points = np.asarray(path.points, dtype=np.float32)
    camera_prior = load_camera_trajectory(config)
    camera_buffer_ok = route_within_camera_buffer(
        points, camera_prior, config["navigation"].get("camera_trajectory_buffer_m", 0.0))
    frames = build_motion(pathfinder, points, config)
    if not route_within_camera_buffer(
            np.asarray([frame[0] for frame in frames]), camera_prior,
            config["navigation"].get("camera_trajectory_buffer_m", 0.0)):
        raise RuntimeError("实际 NavMesh 移动轨迹离开 cameras.json 采集轨迹缓冲区")
    np.save(output_dir / "path.npy", points)
    np.save(output_dir / "trajectory.npy", np.asarray([frame[0] for frame in frames]))
    report.update(navmesh=str(navmesh_path.resolve()), navigable_area_m2=float(pathfinder.navigable_area),
                  num_islands=int(pathfinder.num_islands), largest_island=largest_island, route_island=island,
                  largest_island_area_m2=float(pathfinder.island_area(largest_island)),
                  all_island_areas_m2=[float(pathfinder.island_area(i)) for i in range(pathfinder.num_islands)],
                  bounds=[np.asarray(x).tolist() for x in pathfinder.get_bounds()],
                  navmesh_settings={"cell_size_m": nav["cell_size_m"], "cell_height_m": nav["cell_height_m"],
                                    "agent_width_m": nav["agent_width_m"], "agent_radius_m": nav["agent_width_m"] / 2,
                                    "agent_height_m": nav["agent_height_m"], "max_climb_m": nav["max_climb_m"],
                                    "max_slope_deg": nav["max_slope_deg"], "edge_max_error_m": nav["edge_max_error_m"]},
                  agent_width_m=nav["agent_width_m"], agent_radius_m=nav["agent_width_m"] / 2,
                  agent_height_m=nav["agent_height_m"], geodesic_distance_m=float(path.geodesic_distance),
                  navigation_steps=len(frames) - 1, goal_error_m=float(np.linalg.norm(frames[-1][0] - points[-1])),
                  all_steps_navigable=all(pathfinder.is_navigable(frame[0].astype(np.float32), 0.15) for frame in frames))
    report["camera_trajectory_prior"] = {
        "enabled": camera_prior is not None,
        "source_file": camera_prior["source_file"] if camera_prior else None,
        "source_pose_count": camera_prior["source_count"] if camera_prior else 0,
        "unique_position_count": camera_prior["unique_count"] if camera_prior else 0,
        "buffer_m": float(config["navigation"].get("camera_trajectory_buffer_m", 0.0)),
        "endpoints_prefer_camera_trajectory": bool(config["navigation"].get(
            "prefer_camera_trajectory_endpoints", False)),
        "route_within_buffer": bool(camera_buffer_ok),
        "coordinate_conversion": "source Z-up (x,y,z) -> Habitat Y-up (x,z,-y)",
        "endpoints_source": ("camera_trajectory_xyz_snapped_to_navmesh"
                             if camera_prior is not None and config["navigation"].get(
                                 "prefer_camera_trajectory_endpoints", True)
                             and config["navigation"].get("start") is None
                             else "configured_or_navmesh_sample"),
    }
    topdown_plots(pathfinder, points, frames, scene_dir, output_dir, config, island)

    if config["output"].get("gaussian_navmesh_overlay", False) and not args.bake_only:
        import matplotlib
        matplotlib.use("Agg")
        from PIL import Image
        height = int(config["visualization"]["image_height_px"])
        width = int(config["visualization"]["image_width_px"])
        world_bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float64)
        xmin, xmax = world_bounds[:, 0]
        zmin, zmax = world_bounds[:, 2]
        camera_y = float(world_bounds[1, 1] + config["visualization"]["gaussian_topdown_height_m"])
        span = max(float(xmax - xmin), float(zmax - zmin) * width / height) * 1.10
        focal = width * (camera_y - float(world_bounds[1, 1])) / span

        topdown_sensor = hs.CameraSensorSpec()
        topdown_sensor.uuid = "topdown_rgb"
        topdown_sensor.sensor_type = hs.SensorType.COLOR
        topdown_sensor.resolution = [height, width]
        topdown_sensor.position = mn.Vector3(0, 0, 0)
        topdown_sensor.orientation = mn.Vector3(-math.pi / 2, 0, 0)
        topdown_sensor.hfov = math.degrees(2 * math.atan(width / (2 * focal)))
        topdown_sensor.near = config["camera"]["near_m"]
        topdown_sensor.far = camera_y - float(world_bounds[0, 1]) + 100.0
        if render_sim is None:
            topdown_agent = hs.agent.AgentConfiguration()
            topdown_agent.sensor_specifications = [topdown_sensor]
            topdown_sim_cfg = hs.SimulatorConfiguration()
            topdown_sim_cfg.scene_id = str((scene_dir / "wanda.stage_config.json").resolve())
            topdown_sim_cfg.enable_physics = False
            topdown_sim_cfg.load_semantic_mesh = False
            topdown_sim_cfg.gpu_device_id = 0
            with hs.Simulator(hs.Configuration(topdown_sim_cfg, [topdown_agent])) as topdown_sim:
                topdown_state = hs.AgentState()
                topdown_state.position = mn.Vector3(float((xmin + xmax) / 2), camera_y,
                                                    float((zmin + zmax) / 2))
                topdown_state.rotation = yaw_quaternion(0.0)
                topdown_sim.initialize_agent(0, topdown_state)
                topdown_rgb = topdown_sim.get_sensor_observations()["topdown_rgb"][:, :, :3]
        else:
            # 批量入口已在 Simulator 创建前注册俯视 sensor，和行走相机共用 Gaussian 场景。
            topdown_state = hs.AgentState()
            topdown_state.position = mn.Vector3(float((xmin + xmax) / 2), camera_y,
                                                float((zmin + zmax) / 2))
            topdown_state.rotation = yaw_quaternion(0.0)
            render_sim.initialize_agent(0, topdown_state)
            topdown_rgb = render_sim.get_sensor_observations()["topdown_rgb"][:, :, :3]
            restore_state = hs.AgentState()
            restore_state.position = mn.Vector3(*frames[-1][0].tolist())
            restore_state.rotation = yaw_quaternion(frames[-1][1])
            render_sim.initialize_agent(0, restore_state)
        Image.fromarray(topdown_rgb).save(output_dir / "gaussian_topdown_raw.png")
        projected_islands = []
        for island_id in range(pathfinder.num_islands):
            triangles = np.asarray(pathfinder.build_navmesh_vertices(island_id)).reshape(-1, 3, 3)
            if len(triangles):
                pixels = project_navmesh_to_topdown_pixels(
                    triangles, camera_y, focal, (xmin + xmax) / 2, (zmin + zmax) / 2,
                    width, height)
                projected_islands.append((island_id, pixels))
        save_gaussian_navmesh_overlay(
            topdown_rgb, projected_islands, (-0.5, width - 0.5, -0.5, height - 0.5),
            output_dir / "gaussian_navmesh_topdown.png", config)

    if not args.bake_only:
        camera = config["camera"]
        simulator_cfg = hs.SimulatorConfiguration()
        simulator_cfg.scene_id = str(scene_dir / "wanda.stage_config.json")
        simulator_cfg.enable_physics = False; simulator_cfg.load_semantic_mesh = False; simulator_cfg.gpu_device_id = 0
        sensor = hs.CameraSensorSpec(); sensor.uuid = "rgb"; sensor.sensor_type = hs.SensorType.COLOR
        sensor.resolution = [camera["height"], camera["width"]]; sensor.hfov = camera["hfov_deg"]
        sensor.position = mn.Vector3(0, config["motion"]["camera_height_m"], 0)
        sensor.near, sensor.far = camera["near_m"], camera["far_m"]
        agent = hs.agent.AgentConfiguration(); agent.sensor_specifications = [sensor]
        render_times, image_stds, frame_records = [], [], []
        video_path = output_dir / "navigation_rgb.mp4"
        if render_sim is None:
            sim = hs.Simulator(hs.Configuration(simulator_cfg, [agent]))
            owns_simulator = True
        else:
            sim = render_sim
            owns_simulator = False
        try:
            assert sim.pathfinder.load_nav_mesh(str(navmesh_path)); assert hs.cuda_enabled, "CUDA disabled"
            writer_context = (imageio.get_writer(str(video_path), fps=config["motion"]["fps"], macro_block_size=2)
                              if config["output"]["video"] else None)
            try:
                for index, (position, yaw, timestamp) in enumerate(frames):
                    state = hs.AgentState(); state.position = mn.Vector3(*position.tolist()); state.rotation = yaw_quaternion(yaw)
                    sim.initialize_agent(0, state)
                    render_start = time.perf_counter()
                    rgb = sim.get_sensor_observations()["rgb"][:, :, :3]
                    render_times.append(time.perf_counter() - render_start)
                    image_stds.append(float(rgb.std()))
                    if writer_context:
                        writer_context.append_data(rgb)
                    if config["output"]["rgb_keyframes"] and index in {0, len(frames) // 2, len(frames) - 1}:
                        Image.fromarray(rgb).save(output_dir / f"rgb_{index:04d}.png")
                    camera_position = position + np.array([0, config["motion"]["camera_height_m"], 0])
                    frame_records.append({"frame": index, "timestamp_s": float(timestamp),
                                          "agent_position_xyz_m": [float(x) for x in position],
                                          "camera_position_xyz_m": [float(x) for x in camera_position],
                                          "camera_orientation_quaternion_xyzw": quaternion_xyzw(state.rotation),
                                          "yaw_rad": float(yaw), "render_seconds": float(render_times[-1])})
                    if index % 100 == 0:
                        print(f"Rendered {index}/{len(frames)} frames", flush=True)
            finally:
                if writer_context:
                    writer_context.close()
        finally:
            if owns_simulator:
                sim.close()
        assert np.median(image_stds) > 10, "RGB frames have little variation"
        report.update(rgb_frames=len(render_times), render_mean_seconds=float(np.mean(render_times)),
                      rgb_std_median=float(np.median(image_stds)),
                      camera={"model": "D435 RGB nominal pinhole", "width": camera["width"], "height": camera["height"],
                              "hfov_deg": camera["hfov_deg"],
                              "fx": camera["width"] / (2 * math.tan(math.radians(camera["hfov_deg"]) / 2)),
                              "fy": camera["width"] / (2 * math.tan(math.radians(camera["hfov_deg"]) / 2)),
                              "cx": (camera["width"] - 1) / 2, "cy": (camera["height"] - 1) / 2,
                              "height_m": config["motion"]["camera_height_m"],
                              "note": "标称 RGB FoV 的针孔近似，无畸变，不是逐机标定内参"})
        if config["output"]["poses_json"]:
            pose_data = {"coordinate_frame": "Habitat world, Y-up; positions in meters",
                         "quaternion_convention": "xyzw (numpy-quaternion xyz followed by scalar w)",
                         "start_position_xyz_m": frame_records[0]["agent_position_xyz_m"],
                         "goal_position_xyz_m": [float(x) for x in points[-1]],
                         "actual_end_position_xyz_m": frame_records[-1]["agent_position_xyz_m"],
                         "geodesic_distance_m": float(path.geodesic_distance),
                         "frame_rate_fps": config["motion"]["fps"], "frames": frame_records}
            (output_dir / "navigation_poses.json").write_text(json.dumps(pose_data, ensure_ascii=False, indent=2), encoding="utf-8")

    report["environment"] = {"habitat_gs_version": hs.__version__, "cuda_enabled": bool(hs.cuda_enabled)}
    report["config"] = config
    preparation_file = scene_dir / "preparation.json"
    if preparation_file.exists():
        report["preparation"] = json.loads(preparation_file.read_text(encoding="utf-8"))
    (output_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


