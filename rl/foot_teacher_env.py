"""Velocity teacher scaffold with phase and joint-position residual actions."""
import gymnasium as gym
import numpy as np
import mujoco
from velocity_env import VelocityEnv
from velocity_primitive import targets

SCHEMA='foot-velocity-phase-teacher-v1.1'


class FootTeacherEnv(gym.Wrapper):
    def __init__(self,slope_deg=0.,max_steps=500):
        super().__init__(VelocityEnv('foot',slope_deg=slope_deg,max_steps=max_steps))
        self.action_space=gym.spaces.Box(-1,1,(16,),np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(64,),np.float32)
        e=self.unwrapped
        self.floor_id=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,'calibration_floor')
        self.foot_ids=[mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,f'{leg}_foot') for leg in ('FR','FL','RR','RL')]

    def privileged(self,obs):
        contacts=np.zeros(4);forces=np.zeros(4);wrench=np.zeros(6)
        e=self.unwrapped
        for index,c in enumerate(e.data.contact):
            if self.floor_id not in (c.geom1,c.geom2):continue
            for leg,geom in enumerate(self.foot_ids):
                if geom in (c.geom1,c.geom2):
                    mujoco.mj_contactForce(e.model,e.data,index,wrench)
                    contacts[leg]=1;forces[leg]+=abs(wrench[0])/200
        return np.r_[obs,contacts,forces].astype(np.float32)

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        info['schema']=SCHEMA
        return self.privileged(obs),info

    def step(self,action):
        a=np.asarray(action,np.float32)
        if a.shape!=(16,) or not np.isfinite(a).all():raise ValueError('Invalid teacher action')
        a=np.clip(a,-1,1)
        e=self.unwrapped
        q=targets(e.data.time,e.command,e.q0,a[:4]*.25)+a[4:]*.15
        obs,r,term,trunc,info=self.env.step(np.clip((q-e.q0)/.4,-1,1))
        r-=.02*float(np.sum(a*a))
        return self.privileged(obs),r,term,trunc,info
