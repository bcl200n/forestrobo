"""Bounded, reproducible v3 curriculum and held-out evaluation.

python rl/run_forest_experiment.py --steps 262144 --worlds 4
Steps are per embodiment, split equally across two explicitly recorded stages.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from forest_v3 import ForestHistoryEnv, SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def wilson(successes, count):
    if count == 0:
        return [0., 1.]
    z = 1.95996398454
    p = successes/count
    center = (p+z*z/(2*count))/(1+z*z/count)
    delta = z*math.sqrt(p*(1-p)/count+z*z/(4*count**2))/(1+z*z/count)
    return [max(0., center-delta), min(1., center+delta)]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['profile'], row['scenario'])].append(row)
    result = []
    for (profile, scenario), episodes in groups.items():
        n = len(episodes)
        successes = sum(r['is_success'] for r in episodes)
        result.append(dict(profile=profile, scenario=scenario, episodes=n,
            success_rate=successes/n, success_interval_wilson=wilson(successes, n),
            failures=dict(Counter(r['failure'] or ('timeout' if r['truncated'] else 'success') for r in episodes)),
            mean_progress_m=float(np.mean([r['progress_m'] for r in episodes])),
            mechanical_energy_j=float(np.mean([r['energy_j'] for r in episodes])),
            note='Conditional episode interval; not a population estimate over unseen forest types.'))
    return result


def evaluate(model, embodiment, output, episodes=2, seeds=(10000, 10001), max_steps=750,
             environment_class=ForestHistoryEnv, schema=SCHEMA):
    rows = []
    for profile in ('smooth', 'std'):
        for scenario in ('nominal', 'degraded', 'blackout'):
            for terrain_seed in seeds:
                env = environment_class(terrain_seed, profile, embodiment, scenario, randomize=True, max_steps=max_steps)
                try:
                    for episode_seed in range(episodes):
                        obs, reset_info = env.reset(seed=episode_seed)
                        start_distance = env.unwrapped._distance
                        reward, energy, path, timings = 0., 0., 0., []
                        for step in range(max_steps):
                            t0 = time.perf_counter()
                            action = np.zeros(env.action_space.shape, np.float32) if model is None else model.predict(obs, deterministic=True)[0]
                            timings.append((time.perf_counter()-t0)*1000)
                            obs, r, terminated, truncated, info = env.step(action)
                            reward += r
                            energy += info['energy_j']
                            path += info['path_m']
                            if terminated or truncated:
                                break
                        rows.append(dict(embodiment=embodiment, profile=profile, scenario=scenario,
                            terrain_seed=terrain_seed, episode_seed=episode_seed,
                            domain=reset_info['domain'], steps=step+1, reward=reward,
                            is_success=info['is_success'], failure=info['failure'], truncated=truncated,
                            progress_m=start_distance-info['goal_distance'], goal_distance=info['goal_distance'],
                            path_m=path, energy_j=energy, mechanical_j_per_m=energy/path if path>.01 else None,
                            inference_p95_ms=float(np.percentile(timings, 95))))
                        write_json(output, dict(schema=schema, status='running', episodes=rows, summary=summarize(rows)))
                finally:
                    env.close()
    report = dict(schema=schema, status='complete', baseline='zero_action' if model is None else 'PPO',
                  episodes=rows, summary=summarize(rows))
    write_json(output, report)
    return report


class Progress(BaseCallback):
    def __init__(self, path, embodiment, stage, budget):
        super().__init__()
        self.path, self.embodiment, self.stage, self.budget = path, embodiment, stage, budget
        self.last = 0

    def _on_step(self):
        if self.num_timesteps-self.last >= 4096:
            self.last = self.num_timesteps
            write_json(self.path, dict(status='training', embodiment=self.embodiment,
                stage=self.stage, timesteps=self.num_timesteps, requested_steps_per_embodiment=self.budget,
                updated_at=datetime.now(timezone.utc).isoformat()))
        return True


def train_one(kind, args, root):
    directory = root/kind
    directory.mkdir(parents=True, exist_ok=True)
    model = None
    stages = [('smooth', 'nominal'), ('std', 'degraded')]
    for stage, (profile, scenario) in enumerate(stages):
        env = DummyVecEnv([lambda i=i: Monitor(ForestHistoryEnv(
            terrain_seed=100+stage*100+i, profile=profile, embodiment=kind,
            scenario=scenario), filename=str(directory/f'stage{stage}_world{i}.monitor.csv'))
            for i in range(args.worlds)])
        env.seed(args.seed+stage*1000)
        try:
            if model is None:
                model = PPO('MlpPolicy', env, seed=args.seed, device='cpu',
                    n_steps=512, batch_size=256, n_epochs=5, learning_rate=3e-4,
                    policy_kwargs=dict(net_arch=dict(pi=[128,128], vf=[128,128])), verbose=1)
            else:
                model.set_env(env)
            callbacks = [Progress(root/'status.json', kind, stage, args.steps),
                CheckpointCallback(save_freq=max(1, 32768//args.worlds),
                    save_path=str(directory/'checkpoints'), name_prefix=f'stage{stage}')]
            model.learn(total_timesteps=max(512*args.worlds, args.steps//2),
                        reset_num_timesteps=(stage==0), callback=callbacks)
            model.save(str(directory/f'stage{stage}'))
        finally:
            env.close()
    model.save(str(directory/'final'))
    protocol = dict(schema=SCHEMA, embodiment=kind, actual_timesteps=model.num_timesteps,
        requested_timesteps=args.steps, training_seed=args.seed, history=4,
        terrain_seeds=[list(range(100,100+args.worlds)), list(range(200,200+args.worlds))],
        stages=stages, action_dim=16 if kind=='wheel' else 12,
        domain_seed_by_stage=[args.seed,args.seed+1000],
        checkpoint_sha256=hashlib.sha256((directory/'final.zip').read_bytes()).hexdigest(),
        limitation='Simulated sensor model and fixed generic robot; not deployable or real-hardware validated.')
    write_json(directory/'protocol.json', protocol)
    write_json(root/'status.json', dict(status='evaluating', embodiment=kind))
    evaluate(None, kind, directory/'baseline.json', args.episodes)
    evaluate(model, kind, directory/'evaluation.json', args.episodes)
    return protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=262144)
    p.add_argument('--worlds', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--embodiments', nargs='+', choices=['foot','wheel'], default=['foot','wheel'])
    p.add_argument('--output', default=str(ROOT/'rl/models/forest_v3_experiment'))
    args = p.parse_args()
    if args.steps < 1 or not 1 <= args.worlds <= 32 or args.episodes < 1:
        p.error('Invalid steps/worlds/episodes')
    torch.set_num_threads(2)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(root/'request.json', dict(vars(args), schema=SCHEMA,
        started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        completed = [train_one(kind, args, root) for kind in args.embodiments]
        write_json(root/'status.json', dict(status='complete', protocols=completed,
            elapsed_s=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as exc:
        write_json(root/'status.json', dict(status='failed', error=repr(exc), elapsed_s=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()
