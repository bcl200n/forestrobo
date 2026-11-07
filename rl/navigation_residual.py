"""Hierarchical experiment: local goal controller + two learned corrections.

This is a HYBRID controller; zero corrections already use goal tracking.
Always compare to that same zero-residual controller, not to standing still.
"""
import math
import gymnasium as gym
import numpy as np
from forest_v3 import ForestHistoryEnv
import gait

SCHEMA='forest-navigation-residual-v1'


class NavigationResidualEnv(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(ForestHistoryEnv(*args, **kwargs))
        self.action_space=gym.spaces.Box(-1,1,(2,),np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,
            (self.env.observation_space.shape[0]+2,),np.float32)
        self._observation=None

    def _phase(self, obs):
        phase=2*math.pi*gait.FREQ*self.unwrapped.data.time
        return np.r_[obs, math.sin(phase), math.cos(phase)].astype(np.float32)

    def reset(self, **kwargs):
        self._observation, info=self.env.reset(**kwargs)
        info['obs_schema']=SCHEMA
        return self._phase(self._observation), info

    def step(self, residual):
        residual=np.asarray(residual, np.float32)
        if residual.shape!=(2,) or not np.isfinite(residual).all():
            raise ValueError('Two finite residuals required')
        residual=np.clip(residual,-1,1)
        frame=self._observation[-self.env.frame_dim:]
        rawdim=self.unwrapped.observation_space.shape[0]
        ranges=frame[rawdim:rawdim+12]*6
        bearing=math.atan2(float(frame[8]),float(frame[7]))
        angles=np.linspace(-math.pi,math.pi,12,endpoint=False)
        close=np.maximum(0.,1.1-ranges)
        repulsion=np.array([-np.sum(close*np.cos(angles)), -np.sum(close*np.sin(angles))])
        desired=np.array([math.cos(bearing),math.sin(bearing)])+repulsion*.65
        bearing=math.atan2(desired[1],desired[0])
        speed=.6*max(0.,math.cos(bearing)) + float(residual[0])*.15
        speed=float(np.clip(speed,0,.7))
        turn=float(np.clip(1.5*bearing+float(residual[1])*.4,-1.,1.))
        if min(ranges[5:8]) < .45:
            speed=0.
        stale=max(float(frame[-4]),float(frame[-3]))*5 > .2
        if stale:
            speed,turn=0.,0.
        e=self.unwrapped
        if e.wheel:
            action=np.zeros(16,np.float32)
            # right wheel faster -> counterclockwise, in model's FR/FL/RR/RL order
            right=(speed+turn*.24)/gait.WHEEL_R
            left=(speed-turn*.24)/gait.WHEEL_R
            action[12:]=np.array([right,left,right,left])/30
        else:
            target=gait.cpg(e.data.time)
            scale=speed/.6
            for leg in range(4):
                side=-1 if leg in (0,2) else 1
                # Differential stride length; no claim of validated yaw tracking.
                stride=float(np.clip(scale-side*turn*.35,0,1.3))
                target[leg*3+1]=e.q0[leg*3+1]+(target[leg*3+1]-e.q0[leg*3+1])*stride
                target[leg*3+2]=e.q0[leg*3+2]+(target[leg*3+2]-e.q0[leg*3+2])*min(1.,scale+.2)
            if stale:
                target=e.q0.copy()
            action=(target-e.q0)/.35
        self._observation,reward,terminated,truncated,info=self.env.step(np.clip(action,-1,1))
        info['controller_speed_m_s']=speed
        info['controller_yaw_rad_s']=turn
        info['sensor_stop']=stale
        reward-=.02*float(np.sum(residual**2))
        return self._phase(self._observation),reward,terminated,truncated,info

    def close(self):
        self.env.close()
