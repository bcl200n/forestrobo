"""Isolated velocity-tracking teacher task, with exact planar calibration floor.

Generic model only. Privileged simulator velocities are intentional teacher
inputs, not a deployable student or a calibrated real robot.
"""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import gymnasium as gym
import mujoco
import numpy as np
from generate_model import generate
from joint_layout import addresses

SCHEMA='velocity-teacher-v1'


class VelocityEnv(gym.Env):
    def __init__(self, embodiment='foot', wheel_shape='capsule', slope_deg=0., max_steps=500):
        if embodiment not in ('foot','wheel') or wheel_shape not in ('capsule','cylinder'):
            raise ValueError('invalid model choice')
        if not np.isfinite(slope_deg) or abs(slope_deg)>20 or max_steps<1:
            raise ValueError('invalid task limits')
        cache=Path(__file__).resolve().parent/'.asset_cache';cache.mkdir(exist_ok=True)
        self.assets=tempfile.TemporaryDirectory(dir=cache,prefix='velocity_')
        self.wheel=embodiment=='wheel';self.max_steps=max_steps
        generate(seed=90,n_trees=0,n_rocks=0,n_logs=0,wheel=self.wheel,out_dir=self.assets.name,hf_n=33)
        path=Path(self.assets.name)/('forest_dog_wheel.xml' if self.wheel else 'forest_dog.xml')
        tree=ET.parse(path);world=tree.getroot().find('worldbody')
        for g in list(world.findall('geom')):world.remove(g)
        ET.SubElement(world,'geom',name='calibration_floor',type='plane',size='0 0 .1',
            euler=f'0 {np.radians(slope_deg)} 0',friction='1.0 .005 .0001')
        if self.wheel and wheel_shape=='cylinder':
            for g in world.iter('geom'):
                if g.get('name','').endswith('_wheel'):
                    g.attrib.pop('fromto',None);g.set('type','cylinder');g.set('size','.075 .032');g.set('euler','1.57079632679 0 0')
        tree.write(path,encoding='utf-8')
        self.model=mujoco.MjModel.from_xml_path(str(path));self.data=mujoco.MjData(self.model)
        self.q,self.v=addresses(self.model,('abd','hip','knee'))
        self.wq,self.wv=addresses(self.model,('wheel',)) if self.wheel else ([],[])
        self.q0=np.tile([0,.55,-1.15] if self.wheel else [0,.9,-1.72],4)
        self.command=np.zeros(3)
        self.last=np.zeros(16 if self.wheel else 12,dtype=np.float32)
        self.action_space=gym.spaces.Box(-1,1,self.last.shape,np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(64 if self.wheel else 56,),np.float32)
        self.contacts=np.zeros(4)

    def state(self):
        R=self.data.xmat[1].reshape(3,3)
        # MuJoCo free-joint linear velocity is world-frame; angular velocity is local.
        return R.T@self.data.qvel[:3],self.data.qvel[3:6].copy(),R.T@np.array([0,0,-1.])

    def _obs(self):
        linear,angular,gravity=self.state()
        phase=self.data.time*2*np.pi*2.0+np.array([0,np.pi,np.pi,0])
        return np.r_[linear,angular,gravity,self.command,self.data.qpos[self.q]-self.q0,
            self.data.qvel[self.v]*.05,self.last,np.sin(phase),np.cos(phase),
            self.data.qvel[self.wv]*.05 if self.wheel else []].astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[2]=.59 if self.wheel else .47
        self.data.qpos[self.q]=self.q0+self.np_random.normal(0,.005,12)
        self.data.ctrl[:12]=self.q0
        # Settle before measuring tracking, do not score the free-fall transient.
        for _ in range(150):mujoco.mj_step(self.model,self.data)
        self.data.time=0.;self.k=0;self.last[:]=0
        command=(options or {}).get('command')
        self.command=np.asarray(command if command is not None else [self.np_random.uniform(-.25,.45),
            0 if self.wheel else self.np_random.uniform(-.15,.15),self.np_random.uniform(-.5,.5)],float)
        if self.command.shape!=(3,) or not np.isfinite(self.command).all():raise ValueError('invalid command')
        return self._obs(),{'schema':SCHEMA}

    def step(self,action):
        a=np.asarray(action,dtype=np.float32)
        if a.shape!=self.action_space.shape or not np.isfinite(a).all():raise ValueError('invalid action')
        a=np.clip(a,-1,1);self.data.ctrl[:12]=self.q0+.4*a[:12]
        if self.wheel:self.data.ctrl[12:]=30*a[12:]
        energy=0.
        for _ in range(4):
            mujoco.mj_step(self.model,self.data)
            energy+=float(np.sum(np.abs(self.data.actuator_force*self.data.actuator_velocity)))*.005
        self.k+=1
        v,w,g=self.state();actual=np.r_[v[:2],w[2]];error=actual-self.command
        reward=float(np.exp(-np.sum(error[:2]**2)/.10)+.5*np.exp(-error[2]**2/.16))
        reward-=float(.2*np.sum(g[:2]**2)+.01*np.sum(a*a)+.03*np.sum((a-self.last)**2)+.0005*energy)
        fallen=bool(g[2]>-.65 or self.data.qpos[2]<.20)
        if fallen:reward-=5
        self.last=a.copy()
        info=dict(actual=actual.tolist(),command=self.command.tolist(),tracking_error=error.tolist(),
                  energy_j=energy,fallen=fallen)
        return self._obs(),reward,fallen,bool(self.k>=self.max_steps and not fallen),info

    def close(self):self.assets.cleanup()
