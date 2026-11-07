"""Show observed-map search inside the expanded Jianchang street network."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from evaluate_rescue_coverage import StreetMap, StreetSearchEnv, within_lines
from rescue_search_env import SCENARIOS
from train_rescue_search import write_json


CASE_PATH = Path(__file__).with_name('cases')/'xichang-jianchang-core.geojson'
SCHEMA = 'xichang-interior-coverage-v1'


def run(street_map, scenario, seed, coverage):
    env = StreetSearchEnv(street_map=street_map, scenario=scenario, required_coverage=coverage,
                          max_steps=2048, supplement=True)
    env.reset(options={'map_seed': seed})
    path = [env.pos.copy()]
    first_complete = None
    while True:
        _, _, done, cut, info = env.step(env.baseline_action())
        path.append(env.pos.copy())
        if first_complete is None and info['found'] == 3:
            first_complete = env.t
        if done or cut:
            break
    route = street_map.cell_centers(path)
    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    road_lengths = {}
    for (category, line), properties in zip(street_map.features, street_map.feature_properties):
        name = properties.get('name')
        if category == 'street' and name:
            near = within_lines((route[:-1]+route[1:])/2, line, street_map.cell)
            road_lengths[name] = road_lengths.get(name, 0.) + float(lengths[near].sum())
    report = dict(info, first_all_cues_step=first_complete, distance_m=float(lengths.sum()),
                  named_road_nearby_distance_m=road_lengths,
                  path_grid=np.asarray(path).tolist(), source_sha256=sha256(CASE_PATH.read_bytes()).hexdigest())
    return report, env


def plot_pair(street_map, runs, output):
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.colors import ListedColormap

    chinese_font = Path('C:/Windows/Fonts/simhei.ttf')
    font = FontProperties(fname=str(chinese_font)).get_name() if chinese_font.exists() else 'sans-serif'
    with rc_context({'font.family': [font, 'DejaVu Sans'], 'axes.unicode_minus': False}):
        figure = Figure(figsize=(13, 8.6), layout='constrained', facecolor='#faf9f6')
        FigureCanvasAgg(figure)
        figure.suptitle('西昌建昌古城 · 内部街巷搜索', fontsize=23)
        lower = street_map.origin
        upper = lower + street_map.size*street_map.cell
        axes = figure.subplots(1, 2)
        for axis, (report, env), title in zip(axes, runs, ('找到三条线索即结束', '继续搜索可达路网')):
            axis.imshow((~street_map.blocked).T, origin='lower', interpolation='nearest',
                        extent=(lower[0], upper[0], lower[1], upper[1]),
                        cmap=ListedColormap(['#e8e5de', '#fffef9']), vmin=0, vmax=1)
            for category, line in street_map.features:
                if category == 'wall':
                    axis.plot(*line.T, color='#77726a', linewidth=1.5)
                elif category == 'building':
                    axis.fill(*line.T, color='#99938a')
            points = street_map.cell_centers(np.argwhere(env.swept))
            axis.scatter(*points.T, s=5, marker='s', color='#b4d8c6', linewidths=0)
            route = street_map.cell_centers(report['path_grid'])
            split = report['first_all_cues_step'] or len(route)-1
            axis.plot(*route[:split+1].T, color='#146b72', linewidth=1.2)
            axis.plot(*route[split:].T, color='#a34595', linewidth=1.4)
            axis.scatter(*route[0], marker='s', s=42, color='#146b72', zorder=5)
            axis.scatter(*route[-1], marker='o', s=55, edgecolors='white', color='#a34595', zorder=5)
            for idx in range(35, len(route)-1, 70):
                end = min(idx+3, len(route)-1)
                if np.linalg.norm(route[end]-route[idx]) > 0:
                    axis.annotate('', xy=route[end], xytext=route[idx],
                                  arrowprops={'arrowstyle': '->', 'color': '#146b72', 'lw': 1})
            axis.add_patch(Rectangle((-280, -244), 512, 512, fill=False,
                                     linestyle=(0, (4, 3)), edgecolor='#c08335', linewidth=1.2))
            labels = {'什字街':(-475, 80), '仓街':(-350, 205), '北街':(-80, 550),
                      '陶家祠巷':(-370, 45), '后营巷':(15, 405), '府街':(75, 105),
                      '南街':(-190, -35), '涌泉街':(-40, -160)}
            for name, location in labels.items():
                axis.text(*location, name, fontsize=10, ha='center',
                          bbox={'facecolor': '#fffef9', 'edgecolor': 'none', 'alpha': .84, 'pad': 1})
            axis.text(-40, -20, '内巷数据缺失\n未建模区域', ha='center', color='#7d7569', fontsize=10)
            axis.set_title(f"{title}\n{report['steps']} 步 · 可达路网探测覆盖 {report['reachable_cue_coverage']:.1%}",
                           fontsize=13, pad=13)
            axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect='equal',
                     xlabel='东西向偏移 / 米', ylabel='南北向偏移 / 米')
            axis.tick_params(labelsize=9)
        handles = [Line2D([0], [0], color='#146b72', label='找到线索前的轨迹'),
                   Line2D([0], [0], color='#a34595', label='找到线索后继续搜索'),
                   Patch(color='#b4d8c6', label='已探测的道路栅格'),
                   Line2D([0], [0], color='#c08335', linestyle='--', label='上一版裁剪范围')]
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .07), ncol=4, frameon=False, fontsize=10)
        figure.text(.5, .025, '真实 OSM 道路几何 · 街宽与探测范围采用仿真设定 · 灰色街坊包含未建模区域\n'
                    '© OpenStreetMap contributors · ODbL 1.0 · openstreetmap.org/copyright',
                    ha='center', fontsize=9, color='#625e56')
        figure.get_layout_engine().set(rect=(0, .13, 1, .87))
        figure.savefig(output, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    street_map = StreetMap(CASE_PATH)
    sources = ('evaluate_xichang_interior.py', 'evaluate_rescue_coverage.py', 'rescue_coverage.py',
               'rescue_search_env.py', 'train_rescue_long_horizon.py', 'train_rescue_offline_multiscale.py',
               'train_rescue_scan_wait.py', 'train_rescue_guided.py')
    write_json(args.output/'protocol.json', dict(schema=SCHEMA, map=street_map.meta, seed=38000,
        scenarios=SCENARIOS, horizon=2048, coverage_goal=.98,
        source_hashes={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
        case_hash=sha256(CASE_PATH.read_bytes()).hexdigest(),
        scope='One expanded real street network, synthetic cues and disturbances; truth used only in scoring/termination'))
    results = {}
    for scenario in SCENARIOS:
        pair = []
        for label, coverage in [('cues_only', 0.), ('street_sweep', .98)]:
            report, env = run(street_map, scenario, 38000, coverage)
            write_json(args.output/f'{scenario}-{label}.json', dict(schema=SCHEMA, **report))
            results[f'{scenario}-{label}'] = {k: v for k, v in report.items() if k != 'path_grid'}
            pair.append((report, env))
            print(scenario, label, report['steps'], report['success'], report['reachable_cue_coverage'], flush=True)
        if scenario == 'corridor':
            plot_pair(street_map, pair, args.output/'xichang-interior.png')
        for _, env in pair:
            env.close()
    write_json(args.output/'summary.json', dict(schema=SCHEMA, results=results))


if __name__ == '__main__':
    main()
