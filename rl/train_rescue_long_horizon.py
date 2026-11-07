"""Paired architecture and frontier-hint ablation on long rescue routes."""
import argparse
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

from rescue_search_env import SCENARIOS
from train_rescue_offline_multiscale import OfflineSearchEnv, collect, fit, trial
from train_rescue_scan_wait import WAIT
from train_rescue_search import now, write_json


SCHEMA = 'rescue-detour-confirmation-v1'
HINT_CHANNELS = (*range(24, 32), 37)
SUBGOAL_STEPS = 24


class DetourSearchEnv(OfflineSearchEnv):
    def __init__(self, *args, hints=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hints = hints
        self.observation_space = gym.spaces.Box(-1., 1., (40,), np.float32)

    def _make_map(self, seed):
        if seed in self._cache:
            return self._cache[seed]
        rng = np.random.default_rng(seed)
        blocked = np.zeros((self.size, self.size), bool)
        blocked[[0, -1], :] = blocked[:, [0, -1]] = True
        walls = np.array([5, 9, 13]) + rng.integers(0, 2, size=3)
        openings = [rng.integers(13, 17), rng.integers(2, 6), rng.integers(13, 17)]
        for x, y, width in zip(walls, openings, rng.integers(2, 4, size=3)):
            blocked[x, 1:-1] = True
            blocked[x, y:y+width] = False
        targets = np.array([(18, rng.integers(lo, hi)) for lo, hi in ((2, 5), (6, 9), (10, 13))]
                           + [(3, rng.integers(5, 13))])
        transpose = bool(rng.integers(0, 2))
        if self.scenario == 'rubble':
            for x in (walls[0]+2, walls[1]+2, walls[2]+2):
                blocked[x, rng.choice(np.arange(3, 17), size=2, replace=False)] = True
        if transpose:
            blocked, targets = blocked.T.copy(), targets[:, ::-1].copy()
        result = (blocked, targets, np.array([True, True, True, False]))
        self._cache[seed] = result
        return result

    def reset(self, *, seed=None, options=None):
        self.first_scan_step = -2
        self.started_scans = 0
        return super().reset(seed=seed, options=options)

    def _observe_map(self):
        self.frontier_cache = None
        super()._observe_map()

    def _frontier(self):
        if self.frontier_cache is None:
            self.frontier_cache = super()._frontier()
        return self.frontier_cache

    def _obs(self):
        obs = np.append(super()._obs(), np.float32(self.first_scan_step == self.t))
        if not self.hints:
            obs[list(HINT_CHANNELS)] = 0.
        return obs

    def _scan_visible(self):
        if self.first_scan_step == self.t-1:
            self.first_scan_step = -2
            return super()._scan_visible()
        if self._visible():
            self.first_scan_step = self.t
            self.started_scans += 1
        return 0.

    def _shield_replacement(self, correction):
        # A blocked or stale movement must not inject the teacher's route.
        return WAIT

    def _info(self):
        return dict(super()._info(), started_scans=self.started_scans)


def disjoint_splits():
    """Hold out obstacle layouts, including their paired dropout variants."""
    envs = [DetourSearchEnv(scenario=scenario) for scenario in SCENARIOS]
    used = set()
    splits = {}
    for name, start, count in [('train', 33000, 64), ('validation', 34000, 12), ('test', 35000, 16)]:
        seeds = []
        for seed in range(start, start+1000):
            layouts = {sha256(env._make_map(seed)[0].tobytes()).hexdigest() for env in envs}
            if used.isdisjoint(layouts):
                seeds.append(seed)
                used.update(layouts)
                if len(seeds) == count:
                    break
        if len(seeds) != count:
            raise RuntimeError(f'insufficient distinct layouts for {name}')
        splits[name] = seeds
    return splits


def without_hints(data):
    obs = data['observation'].copy()
    obs[:, HINT_CHANNELS] = 0.
    return dict(data, observation=obs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    out = args.output
    splits = disjoint_splits()
    model_seeds = [31, 32, 33]
    epochs = 18
    sources = ('train_rescue_long_horizon.py', 'train_rescue_offline_multiscale.py',
               'train_rescue_scan_wait.py', 'train_rescue_guided.py', 'train_rescue_search.py',
               'rescue_search_env.py')
    write_json(out/'protocol.json', dict(schema=SCHEMA, scenarios=SCENARIOS, splits=splits,
        model_seeds=model_seeds, epochs=epochs, device='cpu', subgoal_steps=SUBGOAL_STEPS,
        masked_channels=HINT_CHANNELS, shield='replace unsafe motion with passive wait',
        completion='all three real synthetic cues confirmed by two consecutive valid scans',
        sources={name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}))

    collection, results, data = {}, {}, {}

    def status(state, **fields):
        write_json(out/'status.json', dict(schema=SCHEMA, status=state, updated_at=now(),
                   collection=collection, results=results, **fields))

    try:
        for name in ('train', 'validation'):
            status('collecting', partition=name)
            started = perf_counter()
            data[name], episodes = collect(splits[name], SUBGOAL_STEPS, environment_class=DetourSearchEnv)
            seconds = perf_counter()-started
            np.savez_compressed(out/f'{name}.npz', **data[name])
            write_json(out/f'{name}-episodes.json', episodes)
            collection[name] = dict(episodes=len(episodes), steps=len(data[name]['action']),
                                    successes=sum(row['success'] for row in episodes), seconds=seconds)

        status('evaluating', controller='frontier')
        results['frontier'] = trial(None, splits['test'], out, 'frontier',
                                    environment_class=DetourSearchEnv, schema=SCHEMA)
        for hints in (True, False):
            condition = 'with_hint' if hints else 'without_hint'
            env = partial(DetourSearchEnv, hints=hints)
            train, valid = (data[name] if hints else without_hints(data[name]) for name in ('train', 'validation'))
            for architecture, multiscale in [('single_step', False), ('two_scale', True)]:
                arm = f'{condition}-{architecture}'
                results[arm] = []
                for seed in model_seeds:
                    label = f'{arm}-{seed}'
                    status('training', controller=label)
                    model, training = fit(train, valid, multiscale, seed, epochs, out/f'{label}.pt', schema=SCHEMA)
                    write_json(out/f'{label}-training.json', training)
                    status('evaluating', controller=label)
                    result = trial(model, splits['test'], out, label, environment_class=env, schema=SCHEMA)
                    results[arm].append(dict(seed=seed, training_seconds=training['training_seconds'], **result))
        status('complete')
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
