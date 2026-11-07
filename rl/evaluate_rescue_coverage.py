"""Evaluate a cue-coverage fallback on detours and a public street-map case."""
import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from rescue_coverage import CoverageSearchEnv
from rescue_search_env import CUE_RANGE, SCENARIOS
from train_rescue_long_horizon import disjoint_splits
from train_rescue_offline_multiscale import trial
from train_rescue_search import write_json


SCHEMA = 'rescue-cue-coverage-v1'
CASE_PATH = Path(__file__).with_name('cases')/'xichang-jianchang.geojson'
STREET_HORIZON = 512


def within_lines(points, line, radius):
    hits = np.zeros(len(points), bool)
    for a, b in zip(line[:-1], line[1:]):
        delta = b-a
        squared_length = delta @ delta
        if squared_length > 0:
            along = np.clip((points-a) @ delta / squared_length, 0., 1.)
            distance = points-a-along[:, None]*delta
            hits |= (distance**2).sum(axis=1) <= radius**2
    return hits


class StreetMap:
    def __init__(self, path):
        from matplotlib.path import Path as Polygon

        data = json.loads(path.read_text(encoding='utf-8'))
        self.meta = data['case']
        self.size = int(self.meta['grid_size'])
        self.cell = float(self.meta['cell_m'])
        self.origin = np.asarray(self.meta['southwest_offset_m'], float)
        self.reference = np.asarray(self.meta['reference_lonlat'], float)
        self.meters_per_degree = np.array([111320*np.cos(np.deg2rad(self.reference[1])), 111320])
        if self.size < 8 or self.cell <= 0:
            raise ValueError('expected a positive cell size and a grid at least 8 cells wide')
        self.features = []
        self.feature_properties = []
        indices = np.indices((self.size, self.size)).reshape(2, -1).T
        points = self.cell_centers(indices)
        street = np.zeros(len(points), bool)
        obstacle = np.zeros(len(points), bool)
        for feature in data['features']:
            category = feature['properties']['kind']
            geometry = feature['geometry']
            coordinates = geometry['coordinates']
            points_m = (np.asarray(coordinates[0] if category=='building' else coordinates)
                        - self.reference) * self.meters_per_degree
            self.features.append((category, points_m))
            self.feature_properties.append(feature['properties'])
            if category == 'building':
                obstacle |= Polygon(points_m).contains_points(points)
            elif category == 'wall':
                obstacle |= within_lines(points, points_m, self.meta['wall_buffer_m'])
            elif category == 'street':
                highway = feature['properties']['highway']
                width = self.meta['street_buffers_m'].get(highway)
                if width is not None:
                    street |= within_lines(points, points_m, width)
            else:
                raise ValueError(f'unknown case feature {category!r}')
        self.blocked = (~street | obstacle).reshape(self.size, self.size)
        self.blocked[[0, -1], :] = self.blocked[:, [0, -1]] = True
        free = np.argwhere(~self.blocked)
        if not len(free):
            raise ValueError('street case has no traversable cells')
        delta = self.cell_centers(free) - np.asarray(self.meta['start_offset_m'])
        self.start = free[np.argmin((delta**2).sum(axis=1))]

    def cell_centers(self, indices):
        return self.origin + (np.asarray(indices)+.5)*self.cell


class StreetSearchEnv(CoverageSearchEnv):
    def __init__(self, *args, street_map, required_coverage=0., **kwargs):
        super().__init__(*args, **kwargs)
        if not 0. <= required_coverage <= 1.:
            raise ValueError('required coverage must be between zero and one')
        self.street_map = street_map
        self.size = street_map.size
        self.required_coverage = required_coverage

    def reset(self, *, seed=None, options=None):
        self.searchable = np.zeros((self.size, self.size), bool)
        obs, _ = super().reset(seed=seed, options=options)
        # True reachability is used only for evaluation and termination.
        for position, _, _ in self._reachable_routes(~self.blocked):
            self.searchable[tuple(position)] = True
        return obs, self._info()

    def _reachable_coverage(self):
        return float((self.swept & self.searchable).sum() / max(1, self.searchable.sum()))

    def _task_success(self):
        return super()._task_success() and self._reachable_coverage() >= self.required_coverage

    def _info(self):
        return dict(super()._info(), reachable_cue_coverage=self._reachable_coverage(),
                    required_coverage=self.required_coverage)

    def _start_position(self):
        return self.street_map.start.copy()

    def _make_map(self, seed):
        if seed in self._cache:
            return self._cache[seed]
        rng = np.random.default_rng(seed)
        blocked = self.street_map.blocked.copy()
        start = self.street_map.start
        routes = list(self._reachable_routes(~blocked, start=start))
        if self.scenario == 'rubble':
            eligible = np.asarray([p for p, _, distance in routes if distance >= 8])
            minimum = .8 * len(routes)
            additions = 0
            for position in rng.permutation(eligible):
                cell = tuple(position)
                blocked[cell] = True
                remaining = list(self._reachable_routes(~blocked, start=start))
                if len(remaining) < minimum:
                    blocked[cell] = False
                    continue
                routes = remaining
                additions += 1
                if additions == 6:
                    break
        candidates = np.asarray([p for p, _, distance in routes if distance >= 8])
        if len(candidates) < 6:
            raise ValueError('street case has too few reachable cue cells')
        targets = [group[rng.integers(len(group))] for group in np.array_split(candidates, 3)]
        nuisance = candidates[~(candidates[:, None] == targets).all(axis=2).any(axis=1)]
        targets.append(nuisance[rng.integers(len(nuisance))])
        result = blocked, np.asarray(targets), np.array([True, True, True, False])
        self._cache[seed] = result
        return result


def fresh_detours():
    envs = [CoverageSearchEnv(scenario=scenario) for scenario in SCENARIOS]

    def layouts(seed):
        return {sha256(env._make_map(seed)[0].tobytes()).hexdigest() for env in envs}

    used = set().union(*(layouts(seed) for seeds in disjoint_splits().values() for seed in seeds))
    selected = []
    for seed in range(36000, 37000):
        hashes = layouts(seed)
        if used.isdisjoint(hashes):
            selected.append(seed)
            used.update(hashes)
            if len(selected) == 16:
                return selected
    raise RuntimeError('insufficient unused detour layouts')


def plot_street_pair(street_map, seed, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10.5, 6), layout='constrained', facecolor='white')
    FigureCanvasAgg(figure)
    figure.suptitle('Jianchang, Xichang | Paired rubble search', fontsize=17, fontweight='bold')
    southwest = street_map.origin
    northeast = southwest + street_map.size*street_map.cell
    for axis, supplement in zip(figure.subplots(1, 2), (False, True)):
        env = StreetSearchEnv(scenario='rubble', street_map=street_map, supplement=supplement,
                              max_steps=STREET_HORIZON)
        env.reset(options={'map_seed':seed})
        path = [env.pos.copy()]
        while True:
            _, _, done, cut, info = env.step(env.baseline_action())
            path.append(env.pos.copy())
            if done or cut:
                break
        axis.set_facecolor('#f0f2f3')
        for category, points in street_map.features:
            if category=='street':
                axis.plot(*points.T, color='#b8c1ca', linewidth=.8, zorder=1)
            elif category=='wall':
                axis.plot(*points.T, color='#697787', linewidth=1.3, zorder=1)
            else:
                axis.fill(*points.T, color='#697787', zorder=1)
        cells = street_map.cell_centers(np.argwhere(env.swept))
        axis.scatter(*cells.T, marker='s', s=15, linewidths=0, color='#c6e8de', zorder=2)
        debris = street_map.cell_centers(np.argwhere(env.blocked & ~street_map.blocked))
        axis.scatter(*debris.T, marker='x', color='#a05131', s=24, zorder=4)
        route = street_map.cell_centers(path)
        axis.plot(*route.T, color='#156a72', linewidth=1.6, zorder=3)
        axis.scatter(*route[0], marker='s', color='#143c5d', s=32, zorder=4)
        for target, found in zip(env.targets[env.real], env.scanned[env.real]):
            axis.scatter(*street_map.cell_centers(target), marker='*', s=115,
                         facecolors='#dc762b' if found else 'none', edgecolors='#9d4b13', zorder=5)
        title = 'Frontier + cue coverage' if supplement else 'Frontier only'
        axis.set_title(f"{title}\n{info['found']}/3 cues confirmed in {info['steps']} steps", fontsize=11)
        axis.set(xlim=(southwest[0], northeast[0]), ylim=(southwest[1], northeast[1]), aspect='equal',
                 xlabel='East offset (m)', ylabel='North offset (m)')
        env.close()
    figure.text(.5, .01, 'Teal route  |  pale green swept cells  |  crosses simulated debris  |  stars synthetic cues\n'
                '© OpenStreetMap contributors, ODbL · openstreetmap.org/copyright · simulation only',
                ha='center', va='bottom', fontsize=8)
    figure.get_layout_engine().set(rect=(0, .065, 1, .935))
    figure.savefig(output, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    street_map = StreetMap(CASE_PATH)
    tests = {'reproduction': [35012], 'detour_holdout': fresh_detours(), 'street_case': list(range(37000, 37016))}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'protocol.json', dict(schema=SCHEMA, scenarios=SCENARIOS, seeds=tests,
        map=street_map.meta, map_sha256=sha256(CASE_PATH.read_bytes()).hexdigest(), cue_range_cells=CUE_RANGE,
        horizons={case: STREET_HORIZON if case == 'street_case' else 256 for case in tests},
        case_scope='One real street geometry; simulated corridor widths, blockages, outages and clues'))
    results = {}
    for case, seeds in tests.items():
        env = StreetSearchEnv if case == 'street_case' else CoverageSearchEnv
        case_kwargs = {'street_map':street_map} if case == 'street_case' else {}
        horizon = STREET_HORIZON if case == 'street_case' else 256
        results[case] = {}
        for supplement, name in [(False, 'frontier'), (True, 'coverage')]:
            label = f'{case}-{name}'
            print(f'Evaluating {label}', flush=True)
            score = trial(None, seeds, args.output, label, schema=SCHEMA, horizon=horizon,
                          environment_class=partial(env, supplement=supplement, max_steps=horizon, **case_kwargs))
            rows = json.loads((args.output/f'{label}.json').read_text(encoding='utf-8'))['episodes']
            score['mean_cue_coverage'] = float(np.mean([row['cue_coverage'] for row in rows]))
            results[case][name] = score
    plot_street_pair(street_map, tests['street_case'][0], args.output/'xichang-map.png')
    write_json(args.output/'summary.json', dict(schema=SCHEMA, results=results))


if __name__ == '__main__':
    main()
