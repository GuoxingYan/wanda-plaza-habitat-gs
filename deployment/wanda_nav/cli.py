"""批量执行入口：一次烘焙，按独立种子生成多条轨迹及执行配置快照。"""
import argparse
import copy
import json
from pathlib import Path
import traceback
import yaml
from .planning import load_config
from .pipeline import run_one
from .planning import resolve_path
import habitat_sim as hs
import magnum as mn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.yaml')
    parser.add_argument('--scene-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results_full')
    parser.add_argument('--skip-bake', action='store_true')
    parser.add_argument('--bake-only', action='store_true')
    parser.add_argument('--force', action='store_true', help='重建已完成的路线目录')
    parser.add_argument('--count', type=int, help='覆盖 YAML 中的轨迹条数')
    parser.add_argument('--single', action='store_true', help='只运行一条路线（用于完整场景预验证）')
    args = parser.parse_args()
    config = load_config(args.config)
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
    agent.sensor_specifications = [sensor]
    pending = []
    for index in range(count):
        current = copy.deepcopy(config)
        current['navigation']['seed'] += index
        run_args = copy.copy(args)
        run_args.output = args.output / f'route_{index:04d}'
        # 批次使用同一份已验证 NavMesh；外部预先烘焙可避免反复初始化模拟器。
        run_args.skip_bake = True
        run_args.output.mkdir(parents=True, exist_ok=True)
        done_file = run_args.output / 'validation.json'
        report = json.loads(done_file.read_text()) if done_file.exists() else {}
        required = ['validation.json', 'navigation_rgb.mp4', 'navigation_poses.json',
                    'trajectory_topdown.png', 'mesh_navmesh_topdown.png']
        if (not args.force and not (run_args.output / 'failure.json').exists()
                and report.get('rgb_frames', 0) > 0
                and report.get('config') == current
                and all((run_args.output / name).is_file() for name in required)):
            print(f'[{index + 1}/{count}] skip completed {run_args.output}', flush=True)
            continue
        # 每条结果保存实际生效配置，便于复现实验及查看可视化参数。
        (run_args.output / 'effective_config.yaml').write_text(
            yaml.safe_dump(current, allow_unicode=True, sort_keys=False), encoding='utf-8')
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
