"""Compare an observed-progress supervisor on fresh simulated search maps."""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from train_rescue_guided import RescueStressEnv, SCHEMA, summarize
from train_rescue_search import evaluate, write_json


class ProgressFallbackEnv(RescueStressEnv):
    """Latches a frontier controller after extended observable stagnation."""

    def reset(self, **kwargs):
        self.fallback_active = False
        self.no_progress_steps = 0
        self.fallback_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('invalid rescue action')
        prior_observations = np.count_nonzero(self.known)
        prior_cues = len(self.candidate_events)
        had_signal = self.link_valid and self.sensor_valid
        fallback = self.fallback_active and had_signal
        requested = self.baseline_action() if fallback else action
        obs, reward, terminated, truncated, info = super().step(requested)
        self.fallback_steps += int(fallback)
        progress = (np.count_nonzero(self.known) > prior_observations or
                    len(self.candidate_events) > prior_cues)
        if progress:
            self.no_progress_steps = 0
        elif had_signal and self.link_valid and self.sensor_valid:
            self.no_progress_steps += 1
        if self.no_progress_steps >= 24:
            self.fallback_active = True
        info.update(policy_proposal=int(action), fallback_active=self.fallback_active,
                    fallback_steps=self.fallback_steps)
        return obs, reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = list(range(28000, 28008))
    write_json(args.output/'protocol.json', dict(seeds=seeds, threshold_steps=24,
        progress=['newly observed grid cells','newly recorded candidate cues'],
        outage_steps_counted=False, latch='frontier until episode end'))
    model = PPO.load(args.policy, device='cpu')
    scores = {}
    for label, policy, env in [('frontier', None, RescueStressEnv),
                               ('guided', model, RescueStressEnv),
                               ('guided_with_fallback', model, ProgressFallbackEnv)]:
        result = evaluate(policy, seeds, args.output, label, environment_class=env, schema=SCHEMA)
        scores[label] = summarize(result)
        scores[label]['fallback_episodes'] = sum(r.get('fallback_steps', 0)>0 for r in result['episodes'])
    write_json(args.output/'comparison.json', scores)
    print(scores, flush=True)


if __name__ == '__main__':
    main()
