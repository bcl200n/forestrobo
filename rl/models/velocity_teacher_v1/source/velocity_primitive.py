"""Experimental command-conditioned foot primitive; not a certified controller."""
import numpy as np


def targets(time,command,q0,phase_residual=None):
    vx,vy,yaw=np.asarray(command,float)
    phases=2*np.pi*2.0*time+np.array([0,np.pi,np.pi,0])
    if phase_residual is not None:phases+=phase_residual
    out=np.asarray(q0,float).copy()
    for leg in range(4):
        side=-1 if leg in (0,2) else 1
        stride=np.clip((vx-side*.22*yaw)/.6,-1,1)
        amplitude=min(1.,abs(stride)+abs(vy)/.4)
        s,c=np.sin(phases[leg]),np.cos(phases[leg])
        # Positive hip derivative during stance moves foot backward -> body forward.
        out[3*leg+1]+=.26*stride*s
        out[3*leg+2]+=-.44*amplitude*max(0.,-c)+.08*amplitude*max(0.,c)
        out[3*leg]+=.12*np.clip(vy/.4,-1,1)*s
    return out
