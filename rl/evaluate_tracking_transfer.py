"""Compare stored PPO checkpoints to zero residual, and export a shadow replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn

from forest_tracking_residual import ForestTrackingEnv, evaluate
from robotics.tracking_shadow import NumpyActor, RESIDUAL_SCALE, SCHEMA, TrackingShadow
from train_rescue_search import write_json


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / 'rl/models/tracking_slope/phase1.zip'
FINAL = ROOT / 'rl/models/tracking_slope/final.zip'
SEEDS = (98200, 98201, 98202)
SLOPES = (-10., -5., 5., 10.)
COMMANDS = ((.22, 0., 0.), (.18, .08, .16), (.16, -.09, -.24), (-.08, 0., .14), (0., 0., 0.))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_actor(model, path):
    policy = model.policy
    layers = list(policy.mlp_extractor.policy_net)
    if (model.observation_space.shape != (14,) or model.action_space.shape != (3,)
            or not np.all(model.action_space.low == -1.) or not np.all(model.action_space.high == 1.)
            or policy.squash_output or type(policy.pi_features_extractor) is not FlattenExtractor
            or len(layers) % 2 or any(type(layer) is not nn.Linear for layer in layers[::2])
            or any(type(layer) is not nn.Tanh for layer in layers[1::2])
            or type(policy.action_net) is not nn.Linear):
        raise ValueError('expected a Tanh tracking MLP with a clipped linear actor head')
    arrays = {'schema': np.asarray(SCHEMA)}
    for i, layer in enumerate([*layers[::2], policy.action_net]):
        arrays[f'weight{i}'] = layer.weight.detach().cpu().numpy()
        arrays[f'bias{i}'] = layer.bias.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)


def verify_replay(model, output):
    path = output / 'tracking_actor.npz'
    export_actor(model, path)
    actor = NumpyActor(path)
    shadow = TrackingShadow(actor)
    frames, observations, errors = [], [], []
    env = ForestTrackingEnv(slope=5.)
    try:
        command = COMMANDS[1]
        obs, _ = env.reset(seed=SEEDS[0], options={'command': command})
        for step in range(env.physics.max_steps):
            tick = step * .02
            linear, angular, gravity = env.physics.state()
            frame = dict(tick_s=tick, sample_s=tick, reference=list(command), enabled=True,
                         body_velocity=[*linear[:2], angular[2]], gravity=gravity.tolist(),
                         phase_rad=env.physics.data.time * 2 * np.pi * 1.8)
            frames.append(frame)
            observations.append(obs)
            prediction = shadow.step(frame)
            assert prediction['reason'] == 'shadow_prediction', prediction
            action, _ = model.predict(obs, deterministic=True)
            errors.append(float(np.max(np.abs(prediction['residual'] - action * RESIDUAL_SCALE))))
            obs, _, fallen, timeout, _ = env.step(action)
            if fallen or timeout:
                break
    finally:
        env.close()

    # Include out-of-distribution observations to exercise action clipping.
    random_obs = np.random.default_rng(811).normal(0, 3, (4096, 14)).astype(np.float32)
    observations = np.concatenate([observations, random_obs])
    expected, _ = model.predict(observations, deterministic=True)
    error = float(np.max(np.abs(actor.action(observations) - expected)))
    if max(error, max(errors)) >= 1e-5:
        raise RuntimeError('NumPy actor or reconstructed observations differ from PPO')

    end = frames[-1]['tick_s']
    faults = (
        ({'tick_s': end}, 'nonmonotonic_tick'),
        ({'tick_s': end+.02, 'sample_s': end-.2}, 'stale_or_future_sample'),
        ({'tick_s': end+.04, 'sample_s': end+.05}, 'stale_or_future_sample'),
        ({'tick_s': end+.06, 'sample_s': end}, 'repeated_or_old_sample'),
        ({'tick_s': end+.08, 'sample_s': end+.08, 'enabled': False}, 'external_disable'),
        ({'tick_s': end+.10, 'sample_s': end+.10, 'reference': [0., 0., 0.]}, 'zero_reference'),
        ({'tick_s': end+.12, 'sample_s': end+.12, 'reference': [.9, 0., 0.]}, 'reference_outside_training_range'),
        ({'tick_s': end+.14, 'sample_s': end+.14, 'gravity': [0., 0., 0.]}, 'invalid_input'),
        ({'tick_s': end+.20, 'sample_s': end+.20}, 'tick_gap'),
    )
    reference_frame = frames[-1]
    reasons = Counter()
    for updates, reason in faults:
        frame = {**reference_frame, **updates}
        result = shadow.step(frame)
        if result != {'reason': reason, 'residual': [0., 0., 0.]}:
            raise RuntimeError(f'expected {reason}, got {result}')
        reasons[reason] += 1
        frames.append(frame)
    resumed = {**reference_frame, 'tick_s': end+.22, 'sample_s': end+.22}
    if shadow.step(resumed)['reason'] != 'shadow_prediction':
        raise RuntimeError('shadow replay did not recover after valid data returned')
    frames.append(resumed)
    (output / 'shadow_frames.jsonl').write_text(''.join(json.dumps(frame, allow_nan=False) + '\n'
                                                       for frame in frames), encoding='utf-8')
    return dict(actor_sha256=sha256(path), compared_observations=len(observations),
                max_abs_action_difference=error, max_abs_replay_residual_difference=max(errors),
                valid_replay_frames=len(errors), fault_frames=dict(reasons), recovered=True)


def metrics(report):
    rows = report['episodes']
    stationary = [row for row in rows if not any(row['command'])]
    return dict(cases=len(rows), mean_score=report['mean_score'],
                worst_20pct_score=report['worst_20pct_score'], falls=report['falls'],
                mean_episode_rmse=np.mean([row['rmse'] for row in rows], axis=0).tolist(),
                stationary_mean_episode_rmse=np.mean([row['rmse'] for row in stationary], axis=0).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    phase1 = PPO.load(PHASE1, device='cpu')
    final = PPO.load(FINAL, device='cpu')
    results = dict(schema=SCHEMA, deployment_ready=False, phase1_steps=phase1.num_timesteps,
                   final_additional_steps=final.num_timesteps,
                   checkpoints={str(path.relative_to(ROOT)): sha256(path) for path in (PHASE1, FINAL)},
                   source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (
                       Path(__file__).resolve(), ROOT/'robotics/tracking_shadow.py', ROOT/'rl/forest_tracking_residual.py',
                       ROOT/'rl/velocity_env.py', ROOT/'rl/cartesian_gait.py', ROOT/'rl/generate_model.py')},
                   holdout=dict(slopes_deg=SLOPES, commands=COMMANDS, seeds=SEEDS, horizon_s=5,
                                stress=dict(floor_sliding_friction=.6, trunk_point_payload_kg=2)),
                   results={})
    results['shadow'] = verify_replay(phase1, out)
    protocols = (('previous', {}), ('holdout', {}), ('holdout_stress', {'floor_friction': .6, 'payload_kg': 2.}))
    for profile, options in protocols:
        group = results['results'][profile] = {}
        if profile != 'previous':
            options = dict(slopes=SLOPES, cases=COMMANDS, seeds=SEEDS, **options)
        for name, model in (('rule', None), ('phase1', phase1), ('final', final)):
            report = evaluate(model, **options)
            write_json(out / f'{profile}_{name}.json', report)
            group[name] = metrics(report)
            print(json.dumps(dict(profile=profile, policy=name, **group[name])), flush=True)
            write_json(out / 'summary.json', results)


if __name__ == '__main__':
    main()
