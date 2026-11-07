"""Paired PPO experiment with sparse corrective supervision on student rollouts."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rescue_search_env import RescueSearchEnv, SCENARIOS
from train_rescue_search import evaluate, now, write_json


SCHEMA = 'rescue-guided-search-v2'


class RescueStressEnv(RescueSearchEnv):
    def __init__(self, *args, shield=True, **kwargs):
        super().__init__(*args, rubble_count=28, corridor_gap=2, **kwargs)
        self.shield = shield

    def reset(self, *, seed=None, options=None):
        self.overrides = 0
        obs, info = super().reset(seed=seed, options=options)
        rng = np.random.default_rng(self.map_seed + 67891)
        outage_end = max(11, self.max_steps-11)
        self.link_outages = [(int(t), int(t+rng.integers(5,17))) for t in rng.integers(10,outage_end,size=4)]
        self.sensor_outages = [(int(t), int(t+rng.integers(4,13))) for t in rng.integers(6,outage_end,size=5)]
        return obs, info

    def _sensor_status(self):
        if self.scenario != 'dropout':
            return True, True
        return (not any(start <= self.t < end for start,end in self.link_outages),
                not any(start <= self.t < end for start,end in self.sensor_outages))

    def _shield_replacement(self, correction):
        return int(correction)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('invalid rescue action')
        proposal = int(action)
        legal = {i for i,_ in self._neighbors(self.pos, self.known == 1)}
        unsafe = proposal < 8 and (proposal not in legal or not self.link_valid or not self.sensor_valid)
        missed_cue = proposal < 8 and self.link_valid and self.sensor_valid and bool(self._visible())
        teacher = self.baseline_action() if unsafe or missed_cue else None
        applied = self._shield_replacement(teacher) if self.shield and unsafe else proposal
        overridden = applied != proposal
        self.overrides += int(overridden)
        obs, reward, terminated, truncated, info = super().step(applied)
        info.update(proposed_action=proposal, applied_action=applied,
                    correction_action=teacher if teacher != proposal else None,
                    shield_overrides=self.overrides)
        return obs, reward - .3 * int(overridden), terminated, truncated, info


def make_world(rank):
    return Monitor(RescueStressEnv(scenario=SCENARIOS[rank % 3],
                   map_seeds=range(500+rank*16, 516+rank*16), seed=20261107+rank))


def imitate(model, observations, actions, weight):
    obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=model.device)
    labels = torch.as_tensor(actions, dtype=torch.long, device=model.device)
    dist = model.policy.get_distribution(obs)
    loss = -weight * dist.log_prob(labels).mean()
    optimizer = model.policy.optimizer
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
    optimizer.step()
    return float(loss.detach())


def warm_start(model):
    observations, actions = [], []
    for i, scenario in enumerate(SCENARIOS):
        env = RescueStressEnv(scenario=scenario, map_seeds=range(500+i*16, 516+i*16))
        for seed in range(500+i*16, 508+i*16):
            obs, _ = env.reset(seed=seed, options={'map_seed':seed})
            for _ in range(env.max_steps):
                action = env.baseline_action()
                observations.append(obs)
                actions.append(action)
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
        env.close()
    rng = np.random.default_rng(117)
    for _ in range(3):
        order = rng.permutation(len(actions))
        for start in range(0, len(order), 128):
            batch = order[start:start+128]
            imitate(model, [observations[i] for i in batch], [actions[i] for i in batch], 1.)
    return len(actions)


class CorrectedPPO(PPO):
    """Keep original PPO actions and log-probs. Auxiliary labels are separate."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrections = []
        self.correction_count = 0

    def train(self):
        super().train()
        if self.corrections:
            batch = self.corrections[:512]
            observations, labels = zip(*batch)
            loss = imitate(self, observations, labels, .15)
            self.logger.record('train/correction_loss', loss)
            self.corrections.clear()


class ResearchProgress(BaseCallback):
    def __init__(self, out, label, target, demonstrations, corrected):
        super().__init__()
        self.out = out
        self.label = label
        self.target = target
        self.demonstrations = demonstrations
        self.corrected = corrected

    def _on_step(self):
        if self.corrected:
            for obs, info in zip(self.model._last_obs, self.locals['infos']):
                label = info.get('correction_action')
                if label is not None:
                    self.model.corrections.append((obs.copy(), int(label)))
                    self.model.correction_count += 1
        if self.n_calls == 1 or self.n_calls % 256 == 0:
            write_json(self.out/'status.json', dict(status='training', arm=self.label,
                actual_steps=self.num_timesteps, target_steps=self.target, updated_at=now(),
                warm_start_samples=self.demonstrations,
                correction_samples=getattr(self.model,'correction_count',0), schema=SCHEMA))
        return True


def summarize(report, *, horizon=256):
    rows = report['episodes']
    # Failures cost a full horizon plus 60 steps for each missed real cue.
    costs = np.asarray([r['steps'] if r['success'] else horizon+60*(3-r['found']) for r in rows])
    tail = max(1, int(np.ceil(.2*len(costs))))
    return dict(successes=report['successes'], cases=len(rows),
                mean_completion_cost=float(costs.mean()), worst_20pct_cost=float(np.sort(costs)[-tail:].mean()),
                collision_episodes=report['collision_episodes'], unsafe_motion=report['unsafe_motion'],
                shield_overrides=sum(r.get('shield_overrides',0) for r in rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=98304)
    p.add_argument('--worlds', type=int, default=6)
    p.add_argument('--prior', type=Path)
    args = p.parse_args()
    if args.steps < 1 or not 3 <= args.worlds <= 12:
        p.error('positive steps and 3–12 workers required')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    seeds = list(range(27000,27008))
    protocol = dict(schema=SCHEMA, worlds=args.worlds, target_steps_per_arm=args.steps,
                    seeds_train=list(range(500,500+16*args.worlds)), seeds_test=seeds,
                    stress='2-cell openings, 28 rubble placements, seeded burst outages',
                    shield='observed adjacent free and current data validity; same for both training arms',
                    comparisons=['frontier','frozen_v1_if_provided','plain_ppo','guided_ppo'])
    write_json(out/'protocol.json', protocol)
    write_json(out/'status.json', dict(status='training', arm='initializing', actual_steps=0, updated_at=now()))
    summaries = {}
    try:
        for label, algorithm in [('plain_ppo',PPO), ('guided_ppo',CorrectedPPO)]:
            env = SubprocVecEnv([partial(make_world,i) for i in range(args.worlds)], start_method='spawn')
            try:
                model = algorithm('MlpPolicy', env, seed=117, n_steps=256, batch_size=256, n_epochs=4,
                                  learning_rate=3e-4, ent_coef=.015, device='cpu', policy_kwargs={'net_arch':[64,64]})
                guided = label == 'guided_ppo'
                demos = warm_start(model) if guided else 0
                checkpoints = out/label
                callback = ResearchProgress(out,label,args.steps,demos,guided)
                model.learn(total_timesteps=args.steps, callback=[callback,
                    CheckpointCallback(save_freq=max(1,32768//args.worlds),save_path=str(checkpoints))])
                model.save(out/f'{label}.zip')
                summaries[label] = dict(actual_steps=model.num_timesteps, warm_start_samples=demos,
                                        correction_samples=getattr(model,'correction_count',0))
            finally:
                env.close()
        write_json(out/'status.json', dict(status='evaluating', training=summaries, updated_at=now()))
        policies = [('frontier',None)]
        if args.prior:
            policies.append(('frozen_v1',PPO.load(args.prior,device='cpu')))
        policies.extend((label,PPO.load(out/f'{label}.zip',device='cpu')) for label in summaries)
        scores = {label:summarize(evaluate(model,seeds,out,label,environment_class=RescueStressEnv,schema=SCHEMA))
                  for label,model in policies}
        report = dict(status='complete', schema=SCHEMA, updated_at=now(), training=summaries,
                      results=scores, deployment_ready=False)
        write_json(out/'status.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(status='failed',error=repr(exc),updated_at=now(),training=summaries))
        raise


if __name__ == '__main__':
    main()
