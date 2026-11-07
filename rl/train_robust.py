"""Train a NEW policy; legacy policies have a different observation meaning."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from forest_robust import RobustForestEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1_000_000)
    p.add_argument('--worlds', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='models/robust_v2')
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Generation happens sequentially; every world owns a private asset directory.
    env = DummyVecEnv([lambda i=i: Monitor(RobustForestEnv(100+i))
                       for i in range(args.worlds)])
    try:
        model = PPO('MlpPolicy', env, seed=args.seed, verbose=1,
                    n_steps=512, batch_size=128, device='cpu')
        model.learn(total_timesteps=args.steps)
        model.save(str(out / 'final'))
        (out / 'protocol.txt').write_text(
            f'obs_version=2\ntrain_terrain_seeds={list(range(100,100+args.worlds))}\n'
            f'policy_seed={args.seed}\nsteps={args.steps}\n', encoding='utf-8')
    finally:
        env.close()


if __name__ == '__main__':
    main()
