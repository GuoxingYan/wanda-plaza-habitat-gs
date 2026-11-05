"""按 YAML 配置运行 Habitat-GS 的 NavMesh、导航与 Gaussian RGB 闭环验证。"""
import json
import math
import time

import habitat_sim as hs
import imageio.v2 as imageio
import magnum as mn
import numpy as np
from PIL import Image

from .planning import build_motion, choose_route, quaternion_xyzw, resolve_path, yaw_quaternion
from .visualization import topdown_plots


def run_one(args, config):
    scene_dir = args.scene_dir or resolve_path(config["scene"]["directory"])
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    navmesh_path = scene_dir / "wanda_crop.navmesh"
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
        simulator_cfg.scene_id = str(scene_dir / "wanda_crop_mesh_yup.ply")
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
    frames = build_motion(pathfinder, points, config)
    np.save(output_dir / "path.npy", points)
    np.save(output_dir / "trajectory.npy", np.asarray([frame[0] for frame in frames]))
    report.update(navmesh=str(navmesh_path.resolve()), navigable_area_m2=float(pathfinder.navigable_area),
                  num_islands=int(pathfinder.num_islands), largest_island=largest_island, route_island=island,
                  largest_island_area_m2=float(pathfinder.island_area(largest_island)),
                  bounds=[np.asarray(x).tolist() for x in pathfinder.get_bounds()],
                  navmesh_settings={"cell_size_m": nav["cell_size_m"], "cell_height_m": nav["cell_height_m"],
                                    "agent_width_m": nav["agent_width_m"], "agent_radius_m": nav["agent_width_m"] / 2,
                                    "agent_height_m": nav["agent_height_m"], "max_climb_m": nav["max_climb_m"],
                                    "max_slope_deg": nav["max_slope_deg"], "edge_max_error_m": nav["edge_max_error_m"]},
                  agent_width_m=nav["agent_width_m"], agent_radius_m=nav["agent_width_m"] / 2,
                  agent_height_m=nav["agent_height_m"], geodesic_distance_m=float(path.geodesic_distance),
                  navigation_steps=len(frames) - 1, goal_error_m=float(np.linalg.norm(frames[-1][0] - points[-1])),
                  all_steps_navigable=all(pathfinder.is_navigable(frame[0].astype(np.float32), 0.15) for frame in frames))
    topdown_plots(pathfinder, points, frames, scene_dir, output_dir, config, island)

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
        with hs.Simulator(hs.Configuration(simulator_cfg, [agent])) as sim:
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


