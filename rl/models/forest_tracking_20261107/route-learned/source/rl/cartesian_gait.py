"""Experimental Cartesian foot trajectories with analytic model-specific IK.

No pose teleportation or direct base force. Commands reach the simulator only
through the twelve joint servos. Generic model dimensions, not real hardware.
"""
import math
import numpy as np


def leg_ik(target,front=True,side=1):
    x,y,z=target;d=.07*side;a=.03 if front else -.03
    sagittal_z=-math.sqrt(max(1e-6,y*y+z*z-d*d))
    roll=math.atan2(z,y)-math.atan2(sagittal_z,d)
    roll=(roll+math.pi)%(2*math.pi)-math.pi
    l1=math.hypot(a,.30);l2=math.hypot(.02,.32)
    r2=x*x+sagittal_z*sagittal_z
    knee_eff=-math.acos(float(np.clip((r2-l1*l1-l2*l2)/(2*l1*l2),-.999,.999)))
    hip_eff=math.atan2(-x,-sagittal_z)-math.atan2(l2*math.sin(knee_eff),l1+l2*math.cos(knee_eff))
    hip=hip_eff+math.atan2(a,.30)
    knee=knee_eff-math.atan2(.02,.32)-math.atan2(a,.30)
    return [roll,hip,knee]


def targets(time,command):
    vx,vy,w=np.asarray(command,float)
    phase=time*2*math.pi*1.8+np.array([0,math.pi,math.pi,0])
    q=[]
    for leg in range(4):
        front=leg<2;side=-1 if leg in (0,2) else 1
        hx=.27 if front else -.27;hy=side*.225
        fx=vx-w*hy;fy=vy+w*hx
        moving=min(1.,math.hypot(fx,fy)/.15)
        p=phase[leg]
        x=-fx*.16*math.sin(p)
        y=.07*side-fy*.16*math.sin(p)
        z=-.43+.085*moving*max(0.,-math.cos(p))
        q.extend(leg_ik((x,y,z),front,side))
    return np.asarray(q)
