"""批量执行入口：一次烘焙，按独立种子生成多条轨迹及执行配置快照。"""
import argparse
import copy
from pathlib import Path
import yaml
from .planning import load_config
from .pipeline import run_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.yaml')
    parser.add_argument('--scene-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results_crop')
    parser.add_argument('--skip-bake', action='store_true')
    parser.add_argument('--bake-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    count = config['generation']['trajectory_count']
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('generation.trajectory_count 必须为正整数')
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        current = copy.deepcopy(config)
        current['navigation']['seed'] += index
        run_args = copy.copy(args)
        run_args.output = args.output / f'route_{index:04d}'
        run_args.skip_bake = args.skip_bake or index > 0
        run_args.output.mkdir(parents=True, exist_ok=True)
        # 每条结果保存实际生效配置，便于复现实验及查看可视化参数。
        (run_args.output / 'effective_config.yaml').write_text(
            yaml.safe_dump(current, allow_unicode=True, sort_keys=False), encoding='utf-8')
        run_one(run_args, current)
