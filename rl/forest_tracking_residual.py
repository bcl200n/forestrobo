"""Train bounded velocity corrections around the Cartesian gait in MuJoCo."""
import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from cartesian_gait import targets
from train_rescue_search import now, write_json
from velocity_env import VelocityEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotics.tracking_shadow import SCHEMA, tracking_drive, tracking_feedback, tracking_observation


class ForestTrackingEnv(gym.Env):
    def __init__(self, slope=0., level='easy', horizon=250):
        self.physics = VelocityEnv('foot', slope_deg=slope, max_steps=horizon)
        self.level = level
        self.action_space = gym.spaces.Box(-1, 1, (3,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (14,), np.float32)

    def _obs(self):
        _, _, gravity = self.physics.state()
        return tracking_observation(self.reference, self.filtered, gravity,
                                    self.integral, self.physics.data.time * 2 * np.pi * 1.8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        default = [self.np_random.uniform(.12, .28), 0, 0]
        if self.level == 'mixed':
            default = [self.np_random.uniform(-.12, .32), self.np_random.uniform(-.14, .14),
                       self.np_random.uniform(-.3, .3)]
        self.reference = np.asarray(options.get('command', default), float)
        self.physics.reset(seed=seed, options={'command': self.reference})
        self.filtered = np.zeros(3)
        self.integral = np.zeros(3)
        return self._obs(), {'schema': SCHEMA}

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('Three finite residuals required')
        physics = self.physics
        linear, angular, _ = physics.state()
        measured = np.r_[linear[:2], angular[2]]
        if self.level == 'mixed':
            measured += self.np_random.normal(0., .025, size=3)
        self.filtered, self.integral = tracking_feedback(self.reference, self.filtered, self.integral, measured)
        drive = tracking_drive(self.reference, self.filtered, self.integral, action)
        joints = targets(physics.data.time, drive)
        _, _, fallen, timeout, info = physics.step(np.clip((joints - physics.q0) / .4, -1, 1))
        error = np.asarray(info['tracking_error'])
        reward = float(np.exp(-np.sum(error[:2]**2) / .08)
                       + .5 * np.exp(-error[2]**2 / .12)
                       - .025 * np.mean(np.clip(action, -1, 1)**2) - .0005 * info['energy_j'])
        if fallen:
            reward -= 5
        return self._obs(), reward, fallen, timeout, info

    def close(self):
        self.physics.close()


TRACKING_CASES = ((.25, 0., 0.), (.20, .10, .20), (.20, -.10, -.20), (-.10, 0., .18), (0., 0., 0.))


def evaluate(model, *, slopes=(-5., 0., 5.), seeds=(37001, 37002), cases=TRACKING_CASES,
             floor_friction=1., payload_kg=0.):
    rows = []
    for slope in slopes:
        env = ForestTrackingEnv(slope, 'mixed')
        try:
            physics = env.physics
            if floor_friction != 1.:
                floor = physics.model.geom('calibration_floor')
                floor.friction[0] = floor_friction
                # Use the floor coefficient even when the contacting foot has higher friction.
                floor.priority[0] = 1
            if payload_kg:
                # An idealized point payload at the trunk's existing center of mass.
                physics.model.body_mass[1] += payload_kg
                mujoco.mj_setConst(physics.model, physics.data)
            for command in cases:
                for seed in seeds:
                    obs, _ = env.reset(seed=seed, options={'command': command})
                    errors = []
                    energy = 0.
                    for step in range(env.physics.max_steps):
                        action = np.zeros(3) if model is None else model.predict(obs, deterministic=True)[0]
                        obs, _, fallen, timeout, info = env.step(action)
                        energy += info['energy_j']
                        if step >= 25 or fallen:
                            errors.append(info['tracking_error'])
                        if fallen or timeout:
                            break
                    rms = np.sqrt(np.mean(np.square(errors), axis=0))
                    # Penalize early termination separately from tracking error.
                    error_score = float(np.linalg.norm(rms * [1, 1, .4]))
                    rows.append(dict(slope_deg=slope, command=command, seed=seed,
                                     fallen=fallen, steps=step+1, rmse=rms.tolist(), energy_j=energy,
                                     score=error_score + float(fallen)))
        finally:
            env.close()
    scores = np.array([row['score'] for row in rows])
    tail = math.ceil(len(scores) * .2)
    return dict(episodes=rows, mean_score=float(scores.mean()),
                worst_20pct_score=float(np.sort(scores)[-tail:].mean()),
                falls=sum(row['fallen'] for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=49152)
    args = parser.parse_args()
    if args.steps < 2048:
        parser.error('at least 2048 training steps required')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = None
    total_steps = 0
    try:
        stages = [('easy', [0.]*4, .25), ('mixed', [-3., 0., 0., 3.], .75)]
        write_json(out/'protocol.json', dict(schema=SCHEMA, steps=args.steps, policy_seed=315,
            stages=[dict(name=name, slopes_deg=slopes, fraction=fraction, vec_seed=315+i)
                    for i, (name, slopes, fraction) in enumerate(stages)]))
        for stage, (level, slopes, fraction) in enumerate(stages):
            env = DummyVecEnv([partial(ForestTrackingEnv, slope, level) for slope in slopes])
            try:
                if model is None:
                    model = PPO('MlpPolicy', env, seed=315, device='cpu', n_steps=256, batch_size=128,
                                n_epochs=4, learning_rate=2e-4, policy_kwargs={'net_arch':[64,64], 'log_std_init':-1.5})
                    with torch.no_grad():
                        model.policy.action_net.weight.zero_()
                        model.policy.action_net.bias.zero_()
                else:
                    model.set_env(env)
                env.seed(315+stage)
                write_json(out/'status.json', dict(status='training', phase=level, actual_steps=total_steps,
                                                   updated_at=now(), schema=SCHEMA))
                model.learn(total_timesteps=int(args.steps*fraction), reset_num_timesteps=False,
                            callback=CheckpointCallback(save_freq=2048, save_path=str(out), name_prefix=level))
                total_steps = model.num_timesteps
                model.save(out/f'{level}.zip')
            finally:
                env.close()
        write_json(out/'status.json', dict(status='evaluating', actual_steps=total_steps, updated_at=now()))
        baseline, learned = evaluate(None), evaluate(model)
        write_json(out/'baseline.json', baseline)
        write_json(out/'evaluation.json', learned)
        passed = (learned['falls'] <= baseline['falls'] and
                  learned['mean_score'] < baseline['mean_score'] and
                  learned['worst_20pct_score'] < baseline['worst_20pct_score'])
        result = dict(status='complete', actual_steps=total_steps, updated_at=now(), schema=SCHEMA,
                      passed_tracking_gate=passed, deployment_ready=False,
                      baseline={k:v for k,v in baseline.items() if k!='episodes'},
                      learned={k:v for k,v in learned.items() if k!='episodes'})
        write_json(out/'status.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(status='failed', error=repr(exc), updated_at=now()))
        raise


if __name__ == '__main__':
    main()
