"""Resolve state addresses from joint names; MJCF traversal interleaves wheels."""
import mujoco
import numpy as np


def addresses(model, suffixes):
    names = [f'{leg}_{suffix}' for leg in ('FR', 'FL', 'RR', 'RL')
             for suffix in suffixes]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if min(ids) < 0:
        raise ValueError(f'Missing joints: {names}')
    return np.asarray(model.jnt_qposadr[ids]), np.asarray(model.jnt_dofadr[ids])
