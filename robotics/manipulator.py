"""Static feasibility checks sharing the web model's proposed arm dimensions.

No collision solver, joint controller, fatigue or dynamic certification.
"""
import json
from pathlib import Path
import numpy as np

CATALOG = json.loads((Path(__file__).resolve().parents[1]/'design/configs/platforms.json').read_text(encoding='utf-8'))


def rotation(axis, angle):
    x, y, z = np.asarray(axis, float)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])


def chain(q, arm='work5'):
    spec = CATALOG['arms'][arm]
    q = np.asarray(q, float)
    if q.shape != (spec['dof'],) or not np.isfinite(q).all():
        raise ValueError('invalid joint angles')
    lengths = spec['links_m']
    # yaw, shoulder pitch, elbow pitch, optional wrist pitch, axial wrist roll.
    axes = [[0,0,1], [0,1,0], [0,1,0]] + ([[0,1,0]] if spec['dof']==5 else []) + [[1,0,0]]
    translations = [0, lengths[0], lengths[1]] + ([lengths[2]] if spec['dof']==5 else []) + [0 if spec['dof']==5 else lengths[2]]
    p, R, origins, world_axes, centers = np.zeros(3), np.eye(3), [], [], []
    for angle, axis, length in zip(q, axes, translations):
        origins.append(p.copy())
        world_axes.append(R@axis)
        R = R@rotation(axis, angle)
        if length:
            centers.append((len(origins)-1, p+R@np.array([length/2,0,0])))
        p = p+R@np.array([length,0,0])
    return p, np.array(origins), np.array(world_axes), centers


def static_check(q, force_n=(0,0,0), tool_mass_kg=0., arm='work5', base_height_m=.65):
    if tool_mass_kg < 0 or not np.isfinite(tool_mass_kg):
        raise ValueError('invalid tool mass')
    force = np.asarray(force_n, float)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError('invalid force')
    end, origins, axes, centers = chain(q, arm)
    jacobian = np.array([np.cross(axis, end-origin) for origin, axis in zip(origins, axes)]).T
    # force_n is the environment reaction on the tool; gravity acts downward.
    torque = jacobian.T@(force+np.array([0,0,-9.81*tool_mass_kg]))
    for (joint_index, center), mass in zip(centers, CATALOG['arms'][arm]['link_masses_kg']):
        for i in range(joint_index+1):
            torque[i] += np.dot(axes[i], np.cross(center-origins[i], [0,0,-9.81*mass]))
    rated = np.asarray(CATALOG['arms'][arm]['rated_torque_nm'])
    return dict(tcp_m=end.tolist(), torque_nm=torque.tolist(),
        utilization=(np.abs(torque)/rated).tolist(), within_rated=bool(np.all(np.abs(torque)<=rated)),
        theoretical_max_height_m=base_height_m+sum(CATALOG['arms'][arm]['links_m']),
        limitation='Static torque only; no joint limits, collision, gearbox life, contact stability or certification.')


def position_ik(target_m, arm='work5', initial=None, max_iterations=200, tolerance_m=.002):
    """Damped position-only IK with limits; DOES NOT guarantee tool orientation.

    A valid result still needs a collision and wrench feasibility check. The
    ±2 mm solver tolerance is not a claim of real hardware repeatability.
    """
    if not isinstance(max_iterations,int) or max_iterations<1 or not np.isfinite(tolerance_m) or tolerance_m<=0:
        raise ValueError('invalid solver limits')
    spec=CATALOG['arms'][arm]
    target=np.asarray(target_m,float)
    if target.shape!=(3,) or not np.isfinite(target).all():
        raise ValueError('invalid target')
    limits=np.radians(spec['joint_limits_deg'])
    q=np.zeros(spec['dof']) if initial is None else np.asarray(initial,float).copy()
    # Non-singular seed for a straight-chain initial guess.
    if initial is None:
        q[0]=np.arctan2(target[1],target[0])
        q[1:3]=[-.5,1.0]
    if q.shape!=(spec['dof'],) or not np.isfinite(q).all():
        raise ValueError('invalid initial joints')
    q=np.clip(q,limits[:,0],limits[:,1])
    for iteration in range(max_iterations):
        end,origins,axes,_=chain(q,arm)
        error=target-end
        if np.linalg.norm(error)<=tolerance_m:
            break
        jacobian=np.array([np.cross(axis,end-origin) for origin,axis in zip(origins,axes)]).T
        step=jacobian.T@np.linalg.solve(jacobian@jacobian.T+.0025*np.eye(3),error)
        q=np.clip(q+np.clip(step,-.15,.15),limits[:,0],limits[:,1])
    residual=float(np.linalg.norm(target-chain(q,arm)[0]))
    return dict(reachable=residual<=tolerance_m,joints_rad=q.tolist(),residual_m=residual,
        iterations=iteration+1,orientation_checked=False,collision_checked=False)


def support_check(mass_kg, com_xy, half_length_m, half_width_m, tool_position_m, reaction_n, safety_factor=1.3):
    """Horizontal support-plane center of pressure from force/moment balance.

    Assumes all external reaction is at tool_position_m and four-corner support.
    A valid CoP is necessary but does not prove feasible individual foot forces.
    """
    values = np.r_[mass_kg, com_xy, half_length_m, half_width_m, tool_position_m, reaction_n, safety_factor]
    if not np.isfinite(values).all() or min(mass_kg,half_length_m,half_width_m)<=0 or safety_factor<1:
        raise ValueError('invalid support parameters')
    reaction = np.asarray(reaction_n, float)
    normal = mass_kg*9.81-reaction[2]
    if normal <= 0:
        return dict(feasible=False, reason='ground_unloaded', normal_n=float(normal))
    moment = np.cross(tool_position_m, reaction)+np.cross([*com_xy,0], [0,0,-mass_kg*9.81])
    cop = np.array([moment[1]/normal, -moment[0]/normal])
    inside = abs(cop[0]) < half_length_m/safety_factor and abs(cop[1]) < half_width_m/safety_factor
    return dict(feasible=bool(inside), reason='inside_reduced_support' if inside else 'tipping',
                center_of_pressure_m=cop.tolist(), normal_n=float(normal))
