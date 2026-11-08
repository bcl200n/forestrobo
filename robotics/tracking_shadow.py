"""Replay a Cartesian-gait residual policy with NumPy, emitting diagnostics only."""
import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA = 'forest-cartesian-tracking-residual-v1'
RESIDUAL_SCALE = np.array([.12, .08, .16])
DRIVE_LO = np.array([-.25, -.22, -.65])
DRIVE_HI = np.array([.5, .22, .65])
PERIOD_S = .02


def tracking_observation(command, filtered, gravity, integral, phase):
    return np.r_[command, filtered, gravity, integral,
                 np.sin(phase), np.cos(phase)].astype(np.float32)


def tracking_feedback(command, filtered, integral, measured):
    filtered = .96 * filtered + .04 * measured
    integral = np.clip(integral + PERIOD_S * (command - filtered), -.15, .15)
    return filtered, integral


def tracking_drive(command, filtered, integral, residual):
    drive = command + .35 * (command - filtered) + .5 * integral
    if np.linalg.norm(command) > .03:
        drive[0] += .12
    return np.clip(drive + np.clip(residual, -1, 1) * RESIDUAL_SCALE,
                   DRIVE_LO, DRIVE_HI)


class NumpyActor:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            if archive['schema'].item() != SCHEMA:
                raise ValueError('incompatible tracking schema')
            count = (len(archive.files) - 1) // 2
            if count < 1 or set(archive.files) != {'schema'} | {
                    f'{key}{i}' for i in range(count) for key in ('weight', 'bias')}:
                raise ValueError('invalid actor archive')
            self.layers = tuple((archive[f'weight{i}'], archive[f'bias{i}']) for i in range(count))
        inputs = 14
        for weight, bias in self.layers:
            if (weight.ndim != 2 or weight.shape[1] != inputs or weight.shape[0] == 0
                    or bias.shape != (weight.shape[0],)
                    or weight.dtype != np.float32 or bias.dtype != np.float32
                    or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
                raise ValueError('invalid actor parameters')
            inputs = weight.shape[0]
        if inputs != 3:
            raise ValueError('expected three residual outputs')

    def action(self, observation):
        value = np.asarray(observation, dtype=np.float32)
        if value.shape[-1:] != (14,) or not np.isfinite(value).all():
            raise ValueError('expected finite 14-element observations')
        for weight, bias in self.layers[:-1]:
            value = np.tanh(value @ weight.T + bias)
        weight, bias = self.layers[-1]
        return np.clip(value @ weight.T + bias, -1., 1.)


class TrackingShadow:
    def __init__(self, actor):
        self.actor = actor
        self.last_tick = None
        self.last_sample = None
        self.filtered = np.zeros(3)
        self.integral = np.zeros(3)

    def suppress(self, reason):
        self.filtered.fill(0)
        self.integral.fill(0)
        return {'reason': reason, 'residual': [0., 0., 0.]}

    def step(self, frame):
        try:
            tick = float(frame['tick_s'])
            stamp = float(frame['sample_s'])
            phase = float(frame['phase_rad'])
            enabled = frame['enabled']
            reference, measured, gravity = (np.asarray(frame[name], dtype=float)
                                             for name in ('reference', 'body_velocity', 'gravity'))
        except (KeyError, TypeError, ValueError, OverflowError):
            return self.suppress('invalid_input')
        if (not np.isfinite([tick, stamp, phase]).all() or not isinstance(enabled, bool)
                or any(v.shape != (3,) or not np.isfinite(v).all()
                       for v in (reference, measured, gravity))
                or not .95 <= np.linalg.norm(gravity) <= 1.05):
            return self.suppress('invalid_input')

        interval = PERIOD_S if self.last_tick is None else tick - self.last_tick
        if interval <= 0:
            return self.suppress('nonmonotonic_tick')
        self.last_tick = tick
        if not .015 <= interval <= .025:
            return self.suppress('tick_gap')
        if not 0 <= tick - stamp <= .10:
            return self.suppress('stale_or_future_sample')
        if self.last_sample is not None and stamp <= self.last_sample:
            return self.suppress('repeated_or_old_sample')
        self.last_sample = stamp
        if not enabled:
            return self.suppress('external_disable')
        if np.all(reference == 0.):
            return self.suppress('zero_reference')
        if np.any(reference < [-.12, -.14, -.3]) or np.any(reference > [.32, .14, .3]):
            return self.suppress('reference_outside_training_range')

        obs = tracking_observation(reference, self.filtered, gravity, self.integral, phase)
        try:
            action = self.actor.action(obs)
        except ValueError:
            return self.suppress('invalid_observation')
        if not np.isfinite(action).all():
            return self.suppress('invalid_action')
        self.filtered, self.integral = tracking_feedback(reference, self.filtered, self.integral, measured)
        return {'reason': 'shadow_prediction', 'residual': (action * RESIDUAL_SCALE).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actor', type=Path)
    parser.add_argument('frames', type=Path)
    args = parser.parse_args()
    shadow = TrackingShadow(NumpyActor(args.actor))
    with args.frames.open(encoding='utf-8') as frames:
        for index, line in enumerate(frames, start=1):
            try:
                result = shadow.step(json.loads(line))
            except json.JSONDecodeError:
                result = shadow.suppress('invalid_json')
            print(json.dumps({'line': index, **result}, allow_nan=False))


if __name__ == '__main__':
    main()
