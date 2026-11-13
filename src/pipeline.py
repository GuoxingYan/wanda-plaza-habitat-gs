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

from src.planning import (build_motion, camera_trajectory_island_stats, choose_route, load_camera_trajectory,
                       navmesh_agent_radius,
                       quaternion_xyzw, resolve_path, route_within_camera_buffer,
                       yaw_quaternion)
from src.visualization import (topdown_plots, save_gaussian_navmesh_overlay,
                            save_camera_density_comparison,
                            project_navmesh_world_to_topdown_pixels,
                            render_ground_sliced_gaussian_topdown)
from src.avoidance import (add_render_obstacle, composite_obstacle_rgb,
                        plan_around_synthetic_obstacle)
from src.mcap_output import NavigationMcapWriter, sanitize_depth


def run_one(args, config, render_sim=None):
    """执行单条路线的规划、RGB/Depth 渲染、MCAP 写入和验证。"""
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
    effective_agent_radius = navmesh_agent_radius(nav)
    settings.agent_height, settings.agent_radius = nav["agent_height_m"], effective_agent_radius
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
    path, island = choose_route(pathfinder, config, navmesh_path=navmesh_path)
    original_points = np.asarray(path.points, dtype=np.float32)
    points, obstacle = plan_around_synthetic_obstacle(pathfinder, original_points, config)
    frames = build_motion(pathfinder, points, config)
    if obstacle and obstacle.get("triggered"):
        actual_positions = np.asarray([frame[0] for frame in frames], dtype=np.float64)
        obstacle_xz = np.asarray(obstacle["center"], dtype=np.float64)[[0, 2]]
        actual_clearance = np.linalg.norm(actual_positions[:, [0, 2]] - obstacle_xz, axis=1)
        minimum_clearance = float(actual_clearance.min())
        required_clearance = float(obstacle["planning_clearance_radius_m"])
        if minimum_clearance < required_clearance - 0.03:
            raise RuntimeError(
                "实际逐帧轨迹未满足临时障碍净距："
                f"{minimum_clearance:.3f}m < {required_clearance:.3f}m")
        obstacle["actual_min_center_distance_m"] = minimum_clearance
        obstacle["actual_clearance_margin_m"] = minimum_clearance - (
            float(obstacle["radius_m"]) + float(nav["agent_width_m"]) / 2.0)
    camera_prior = load_camera_trajectory(config)
    density_report = None
    if camera_prior is not None:
        stats = camera_trajectory_island_stats(pathfinder, camera_prior, navmesh_path, config)
        filter_enabled = bool(config["navigation"].get("use_camera_trajectory_density_filter", True))
        density_report = {
            "enabled": filter_enabled,
            "source_file": camera_prior["source_file"],
            "candidate_count": int(len(stats["candidate_islands"])),
            "min_points": stats["min_points"],
            "min_points_per_100m2": stats["min_points_per_100m2"],
            "islands": [{"island": int(i), "area_m2": float(stats["areas_m2"][i]),
                         "camera_trajectory_points": int(stats["counts"][i]),
                         "points_per_100m2": float(stats["density_per_100m2"][i]),
                         "retained": bool(stats["keep"][i])}
                        for i in range(len(stats["areas_m2"]))],
        }
        stats["bounds"] = (float(np.asarray(pathfinder.get_bounds())[0, 0]),
                           float(np.asarray(pathfinder.get_bounds())[1, 0]),
                           float(np.asarray(pathfinder.get_bounds())[0, 2]),
                           float(np.asarray(pathfinder.get_bounds())[1, 2]))
        triangles = [(island_id, np.asarray(pathfinder.build_navmesh_vertices(island_id))
                      .reshape(-1, 3, 3)) for island_id in range(pathfinder.num_islands)]
        triangles = [(island_id, tri) for island_id, tri in triangles if len(tri)]
        if config["output"].get("camera_trajectory_density_comparison", True):
            mesh_path = Path(stage_config["collision_asset"])
            if not mesh_path.is_absolute():
                mesh_path = scene_dir / mesh_path
            save_camera_density_comparison(
                triangles, stats, np.asarray([frame[0] for frame in frames]),
                camera_prior["capture_segments"], mesh_path, scene_dir,
                output_dir / "camera_trajectory_density_comparison.png", config)
    elif camera_prior is None:
        density_report = {"enabled": False, "reason": "camera trajectory prior disabled or unavailable"}
    else:
        density_report = {"enabled": False, "reason": "density filter disabled by configuration"}
    camera_buffer_ok = route_within_camera_buffer(
        points, camera_prior, config["navigation"].get("camera_trajectory_buffer_m", 0.0))
    if not route_within_camera_buffer(
            np.asarray([frame[0] for frame in frames]), camera_prior,
            config["navigation"].get("camera_trajectory_buffer_m", 0.0)):
        raise RuntimeError("实际 NavMesh 移动轨迹离开 cameras.json 采集轨迹缓冲区")
    if config["output"].get("numpy_trajectory_files", True):
        np.save(output_dir / "path.npy", points)
        np.save(output_dir / "trajectory.npy", np.asarray([frame[0] for frame in frames]))
    report.update(navmesh=str(navmesh_path.resolve()), navigable_area_m2=float(pathfinder.navigable_area),
                  num_islands=int(pathfinder.num_islands), largest_island=largest_island, route_island=island,
                  largest_island_area_m2=float(pathfinder.island_area(largest_island)),
                  all_island_areas_m2=[float(pathfinder.island_area(i)) for i in range(pathfinder.num_islands)],
                  bounds=[np.asarray(x).tolist() for x in pathfinder.get_bounds()],
                  navmesh_settings={"cell_size_m": nav["cell_size_m"], "cell_height_m": nav["cell_height_m"],
                                    "agent_width_m": nav["agent_width_m"],
                                    "agent_radius_m": effective_agent_radius,
                                    "obstacle_clearance_m": nav.get("obstacle_clearance_m", 0.0),
                                    "agent_height_m": nav["agent_height_m"], "max_climb_m": nav["max_climb_m"],
                                    "max_slope_deg": nav["max_slope_deg"], "edge_max_error_m": nav["edge_max_error_m"]},
                  agent_width_m=nav["agent_width_m"], agent_radius_m=effective_agent_radius,
                  obstacle_clearance_m=nav.get("obstacle_clearance_m", 0.0),
                  agent_height_m=nav["agent_height_m"], geodesic_distance_m=float(path.geodesic_distance),
                  planned_path_length_m=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                  navigation_steps=len(frames) - 1, goal_error_m=float(np.linalg.norm(frames[-1][0] - points[-1])),
                  all_steps_navigable=all(pathfinder.is_navigable(frame[0].astype(np.float32), 0.15) for frame in frames))
    report["synthetic_obstacle"] = obstacle
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
    report["camera_trajectory_density_filter"] = density_report
    if (config["output"].get("trajectory_topdown", True)
            or config["output"].get("mesh_navmesh_overlay", True)):
        topdown_plots(pathfinder, points, frames, scene_dir, output_dir, config, island, obstacle)

    if config["output"].get("gaussian_navmesh_overlay", False) and not args.bake_only:
        import matplotlib
        matplotlib.use("Agg")
        world_bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float64)
        islands = []
        for island_id in range(pathfinder.num_islands):
            triangles = np.asarray(pathfinder.build_navmesh_vertices(island_id)).reshape(-1, 3, 3)
            if len(triangles):
                islands.append((island_id, triangles))
        if render_sim is None:
            raise RuntimeError("Gaussian 正交分块俯视需要复用已注册 topdown_rgb 的 Simulator")
        topdown_rgb, topdown_extent = render_ground_sliced_gaussian_topdown(
            render_sim, islands, config, world_bounds)
        restore_state = hs.AgentState()
        restore_state.position = mn.Vector3(*frames[-1][0].tolist())
        restore_state.rotation = yaw_quaternion(frames[-1][1])
        render_sim.initialize_agent(0, restore_state)
        Image.fromarray(topdown_rgb).save(output_dir / "gaussian_topdown_raw.png")
        width = int(config["visualization"]["image_width_px"])
        height = int(config["visualization"]["image_height_px"])
        projected = [(idx, project_navmesh_world_to_topdown_pixels(tri, topdown_extent, width, height))
                     for idx, tri in islands]
        save_gaussian_navmesh_overlay(
            topdown_rgb, projected, (-0.5, width - 0.5, -0.5, height - 0.5),
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
        sensors = [sensor]
        if config["output"].get("mcap", False):
            depth_sensor = hs.CameraSensorSpec()
            depth_sensor.uuid = "depth"
            depth_sensor.sensor_type = hs.SensorType.DEPTH
            depth_sensor.resolution = [camera["height"], camera["width"]]
            depth_sensor.hfov = camera["hfov_deg"]
            depth_sensor.position = mn.Vector3(0, config["motion"]["camera_height_m"], 0)
            depth_sensor.near, depth_sensor.far = camera["near_m"], camera["far_m"]
            sensors.append(depth_sensor)
        agent = hs.agent.AgentConfiguration(); agent.sensor_specifications = sensors
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
            render_obstacle = add_render_obstacle(sim, obstacle)
            keyframe_indices = {0, len(frames) // 2, len(frames) - 1}
            if obstacle and obstacle.get("triggered"):
                distances_to_obstacle = [float(np.linalg.norm(position[[0, 2]]
                    - np.asarray(obstacle["center"])[[0, 2]])) for position, _, _ in frames]
                closest_frame = int(np.argmin(distances_to_obstacle))
                keyframe_indices.add(closest_frame)
                obstacle["closest_approach_frame_index"] = closest_frame
            writer_context = (imageio.get_writer(str(video_path), fps=config["motion"]["fps"], macro_block_size=2)
                              if config["output"]["video"] else None)
            camera_info = {
                "width": int(camera["width"]), "height": int(camera["height"]),
                "fx": camera["width"] / (2 * math.tan(math.radians(camera["hfov_deg"]) / 2)),
                "fy": camera["width"] / (2 * math.tan(math.radians(camera["hfov_deg"]) / 2)),
                "cx": (camera["width"] - 1) / 2, "cy": (camera["height"] - 1) / 2,
            }
            mcap_writer = None
            if config["output"].get("mcap", False):
                mcap_writer = NavigationMcapWriter(
                    output_dir / f"trajectory_{output_dir.name}.mcap", output_dir.name,
                    camera_info, metadata={
                        "route_length_m": float(path.geodesic_distance),
                        "camera_height_m": float(config["motion"]["camera_height_m"]),
                        "linear_speed_mps": float(config["motion"]["linear_speed_mps"]),
                        "angular_speed_radps": float(config["motion"]["angular_speed_radps"]),
                        "depth_far_clip_m": float(camera["far_m"]),
                        "depth_preview": config["camera"].get("depth_preview", {}),
                    })
            try:
                for index, (position, yaw, timestamp) in enumerate(frames):
                    state = hs.AgentState(); state.position = mn.Vector3(*position.tolist()); state.rotation = yaw_quaternion(yaw)
                    sim.initialize_agent(0, state)
                    render_start = time.perf_counter()
                    observations = sim.get_sensor_observations()
                    rgb = observations["rgb"][:, :, :3]
                    rgb = composite_obstacle_rgb(
                        rgb, obstacle,
                        position + np.array([0, config["motion"]["camera_height_m"], 0]),
                        yaw, camera)
                    render_times.append(time.perf_counter() - render_start)
                    image_stds.append(float(rgb.std()))
                    if writer_context:
                        writer_context.append_data(rgb)
                    if config["output"]["rgb_keyframes"] and index in keyframe_indices:
                        Image.fromarray(rgb).save(output_dir / f"rgb_{index:04d}.png")
                    camera_position = position + np.array([0, config["motion"]["camera_height_m"], 0])
                    frame_records.append({"frame_index": index, "timestamp_s": float(timestamp),
                                          "position": [float(x) for x in camera_position],
                                          "xyzw": quaternion_xyzw(state.rotation)})
                    if mcap_writer is not None:
                        depth = np.asarray(observations["depth"], dtype=np.float32)
                        if depth.ndim == 3:
                            depth = depth[..., 0]
                        # 以相机远裁剪距离填补空洞，并钳制远裁剪面外值。
                        depth = sanitize_depth(depth, camera["far_m"])
                        stamp_ns = mcap_writer.timestamp_ns(timestamp)
                        mcap_writer.add_frame(stamp_ns, rgb, depth,
                                              camera_position, frame_records[-1]["xyzw"], index)
                    if index % 100 == 0:
                        print(f"Rendered {index}/{len(frames)} frames", flush=True)
            finally:
                if writer_context:
                    writer_context.close()
                if mcap_writer is not None:
                    mcap_writer.finish()
                if render_obstacle is not None:
                    obstacle_object, obstacle_mesh = render_obstacle
                    sim.get_rigid_object_manager().remove_object_by_id(obstacle_object.object_id)
                    Path(obstacle_mesh).unlink(missing_ok=True)
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
        if config["output"].get("poses_json", True):
            pose_data = {
                "coordinate_frame": {
                    "name": "Y-up",
                    "position_unit": "m",
                    "orientation_format": "xyzw",
                },
                "route": {
                    "route_id": output_dir.name,
                    "seed": int(config["navigation"]["seed"]),
                    "start": {"position": frame_records[0]["position"],
                              "xyzw": frame_records[0]["xyzw"]},
                    "goal": {"position": frame_records[-1]["position"],
                             "xyzw": frame_records[-1]["xyzw"]},
                    "path_length": float(path.geodesic_distance),
                },
                "camera": {
                    "model": "D435 RGB nominal pinhole",
                    "width": int(camera["width"]),
                    "height": int(camera["height"]),
                    "hfov_deg": float(camera["hfov_deg"]),
                },
                "frames": frame_records,
            }
            (output_dir / "navigation_poses.json").write_text(json.dumps(pose_data, ensure_ascii=False, indent=2), encoding="utf-8")

    report["environment"] = {"habitat_gs_version": hs.__version__, "cuda_enabled": bool(hs.cuda_enabled)}
    report["config"] = config
    preparation_file = scene_dir / "preparation.json"
    if preparation_file.exists():
        report["preparation"] = json.loads(preparation_file.read_text(encoding="utf-8"))
    if config["output"].get("validation_json", True):
        (output_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


