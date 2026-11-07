"""Equal-head PPO comparison of aliased scan versus explicit passive wait."""
import argparse
from functools import partial
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from rescue_search_env import SCENARIOS
from train_rescue_guided import RescueStressEnv, summarize
from train_rescue_search import evaluate, now, write_json


SCHEMA = 'rescue-scan-wait-v1'
SCAN, WAIT = 8, 9


class ScanWaitEnv(RescueStressEnv):
    def __init__(self, *args, split=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        self.action_space = gym.spaces.Discrete(10)

    def reset(self, *, seed=None, options=None):
        self.scans = self.waits = self.empty_stationary = self.stationary_streak = 0
        self.longest_streak = self.outage_steps = self.sighted_waits = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('invalid scan/wait action')
        # The control has the same 10 logits. Its extra action aliases scan.
        applied = SCAN if action == WAIT and not self.split else int(action)
        cues_before = len(self.candidate_events)
        sighted = bool(self._visible()) and self.link_valid
        obs, reward, term, trunc, info = super().step(applied)
        stationary = info['applied_action'] in (SCAN, WAIT)
        valid = self.link_valid and self.sensor_valid
        if not valid:
            self.outage_steps += 1
        elif stationary:
            passive = info['applied_action'] == WAIT
            self.waits += int(passive)
            self.scans += int(not passive)
            self.sighted_waits += int(passive and sighted)
            if len(self.candidate_events) == cues_before:
                self.empty_stationary += 1
                self.stationary_streak += 1
            else:
                self.stationary_streak = 0
        else:
            self.stationary_streak = 0
        self.longest_streak = max(self.longest_streak, self.stationary_streak)
        info.update(policy_action=int(action), scans=self.scans, passive_waits=self.waits,
                    empty_stationary_steps=self.empty_stationary, outage_steps=self.outage_steps,
                    longest_empty_stationary_streak=self.longest_streak,
                    sighted_passive_waits=self.sighted_waits)
        return obs, reward, term, trunc, info


def make_world(rank, split):
    return ScanWaitEnv(scenario=SCENARIOS[rank % 3], split=split,
                       map_seeds=range(800+rank*16, 816+rank*16), seed=1800+rank)


class Progress(BaseCallback):
    def __init__(self, out, label, target):
        super().__init__()
        self.out, self.label, self.target = out, label, target

    def _on_step(self):
        if self.n_calls == 1 or self.n_calls % 256 == 0:
            write_json(self.out/'status.json', dict(status='training', arm=self.label,
                actual_steps=self.num_timesteps, target_steps=self.target, updated_at=now()))
        return True


def metrics(report, *, horizon=256):
    rows = report['episodes']
    result = summarize(report, horizon=horizon)
    result['mean_found'] = report['mean_found']
    for field in ('scans', 'passive_waits', 'empty_stationary_steps', 'outage_steps',
                  'longest_empty_stationary_streak', 'sighted_passive_waits'):
        result['mean_'+field] = sum(row[field] for row in rows)/len(rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=196608)
    parser.add_argument('--worlds', type=int, default=6)
    args = parser.parse_args()
    if args.steps < 1 or not 3 <= args.worlds <= 12:
        parser.error('positive step budget and 3–12 workers required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    seeds = list(range(29000, 29008))
    write_json(args.output/'protocol.json', dict(schema=SCHEMA, worlds=args.worlds,
        steps_per_arm=args.steps, train_seeds=list(range(800, 800+args.worlds*16)),
        test_seeds=seeds, policy_seed=1800, action_count=10,
        control='action 9 aliases scan', split='action 9 waits without collecting cues',
        reward='same environment reward in both arms, including identical time cost'))
    results = {}
    try:
        for label, split in [('alias_scan',False), ('split_wait',True)]:
            env = SubprocVecEnv([partial(make_world, rank, split) for rank in range(args.worlds)],
                               start_method='spawn')
            try:
                model = PPO('MlpPolicy', env, seed=1800, device='cpu', n_steps=256,
                            batch_size=256, n_epochs=4, learning_rate=3e-4, ent_coef=.015,
                            policy_kwargs={'net_arch':[64,64]})
                model.learn(total_timesteps=args.steps, callback=[Progress(args.output,label,args.steps),
                    CheckpointCallback(save_freq=max(1,32768//args.worlds),
                                       save_path=str(args.output/label))])
                model.save(args.output/f'{label}.zip')
                results[label] = dict(actual_steps=model.num_timesteps)
            finally:
                env.close()
        write_json(args.output/'status.json', dict(status='evaluating', updated_at=now(), results=results))
        for label, split in [('frontier',False), ('alias_scan',False), ('split_wait',True)]:
            model = None if label == 'frontier' else PPO.load(args.output/f'{label}.zip',device='cpu')
            report = evaluate(model,seeds,args.output,label,
                              environment_class=partial(ScanWaitEnv,split=split),schema=SCHEMA)
            results.setdefault(label, {}).update(metrics(report))
        status = dict(status='complete', schema=SCHEMA, updated_at=now(), results=results)
        write_json(args.output/'status.json', status)
        print(status, flush=True)
    except BaseException as exc:
        write_json(args.output/'status.json', dict(status='failed', error=repr(exc), updated_at=now()))
        raise


if __name__ == '__main__':
    main()
