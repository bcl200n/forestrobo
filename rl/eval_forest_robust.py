"""Evaluate v2 policies on held-out terrains; write episode-level evidence."""
import argparse
import json
from pathlib import Path
from stable_baselines3 import PPO
from forest_robust import RobustForestEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--output', default='forest_eval.json')
    a = p.parse_args()
    model = PPO.load(a.model, device='cpu')
    rows = []
    for profile in ('smooth', 'std'):
        for faults in (False, True):
            for terrain_seed in (10000, 10001, 10002):
                env = RobustForestEnv(terrain_seed, profile, faults)
                try:
                    for seed in range(a.episodes):
                        obs, _ = env.reset(seed=seed)
                        total = 0.
                        for step in range(500):
                            act, _ = model.predict(obs, deterministic=True)
                            obs, r, term, trunc, info = env.step(act)
                            total += r
                            if term or trunc:
                                break
                        rows.append(dict(profile=profile, faults=faults,
                            terrain_seed=terrain_seed, episode_seed=seed,
                            steps=step+1, reward=total, truncated=trunc, **info))
                finally:
                    env.close()
    Path(a.output).write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('Episodes:', len(rows), 'Success rate:', sum(r['is_success'] for r in rows)/len(rows))


if __name__ == '__main__':
    main()
