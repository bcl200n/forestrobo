"""Physical closed-loop forest benchmark with synthetic lidar and ideal odometry.

No pose teleportation, oracle obstacle coordinates in the planner, or trained
policy success claims. Flat/3-degree forest clearings, not general rough forest.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import xml.etree.ElementTree as ET
import math
from collections import deque
import numpy as np
import mujoco
from velocity_env import VelocityEnv
from cartesian_gait import targets
from robotics.local_navigation import LocalGrid,WaypointFollower


class ForestAutonomy:
    def __init__(self,scenario='corridor',seed=100,*,tracking_policy=None,geodesic=False,
                 clearance_bias=False,weighted_local=False,progress_along_route=False):
        if scenario not in ('corridor','detour','slope','dropout','dynamic'):raise ValueError('scenario')
        if progress_along_route and not (geodesic or clearance_bias or weighted_local):
            raise ValueError('route progress requires grid travel costs')
        self.env=VelocityEnv('foot',slope_deg=3 if scenario=='slope' else 0,max_steps=3500)
        self.scenario=scenario;self.seed=seed
        self.tracking_policy=tracking_policy
        self.geodesic=geodesic
        self.clearance_bias=clearance_bias
        self.weighted_local=weighted_local
        self.progress_along_route=progress_along_route
        e=self.env;path=Path(e.assets.name)/'forest_dog.xml'
        tree=ET.parse(path);world=tree.getroot().find('worldbody')
        rng=np.random.default_rng(seed)
        obstacles=[(1.8,-1.1),(1.8,1.1),(3.6,-1.1),(3.6,1.1),(5,-1.3),(5,1.3)]
        if scenario=='detour':obstacles.append((2.6,0))
        for i,(x,y) in enumerate(obstacles):
            x+=rng.uniform(-.1,.1);y+=rng.uniform(-.08,.08)
            ground=-math.tan(math.radians(3))*x if scenario=='slope' else 0.
            ET.SubElement(world,'geom',name=f'nav_tree_{i}',type='cylinder',pos=f'{x} {y} {ground+.8}',size='.16 .8',rgba='.28 .4 .18 1')
        if scenario=='detour':
            ET.SubElement(world,'geom',name='nav_log',type='capsule',fromto='3.5 -1.3 .09 4.0 -1.3 .09',size='.09',rgba='.4 .24 .12 1')
        ET.SubElement(world,'geom',name='nav_dynamic',type='cylinder',pos='2.7 0 -10',size='.18 .8',rgba='.65 .28 .1 1')
        tree.write(path,encoding='utf-8')
        e.model=mujoco.MjModel.from_xml_path(str(path));e.data=mujoco.MjData(e.model)
        e.model.geom_group[e.model.geom_bodyid>0]=1
        self.mask=np.array([1,0,0,0,0,0],np.uint8)
        self.dynamic_id=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,'nav_dynamic')
        self.goal=np.array([4.5,0.]);self.grid=LocalGrid();self.follower=WaypointFollower()
        self.path=[];self.replans=0;self.traces=[];self.integral=np.zeros(3);self.filtered=np.zeros(3)
        self.energy=0.;self.travel=0.;self.last_position=None;self.stop_steps=0
        self.motion_window=deque(maxlen=400);self.blocked=False
        self.best_distance=float('inf');self.last_progress_time=0.;self.blocked_steps=0
        e.reset(seed=seed,options={'command':[0,0,0]})
        self.progress_anchor=e.data.qpos[:2].copy()
        self.scan();self.path=self.plan_route()
        self.replans+=1

    def plan_route(self):
        return self.grid.route(self.env.data.qpos[:2],self.goal,geodesic=self.geodesic,
                               clearance_bias=self.clearance_bias,weighted_local=self.weighted_local)

    def update_route_progress(self, position, stamp):
        if not self.path:
            return
        # Compare two poses on the same map revision. Newly observed obstacles
        # can change the absolute cost without the robot making any progress.
        old_cost = self.grid.goal_costs.get(self.grid.cell(self.progress_anchor))
        new_cost = self.grid.goal_costs.get(self.grid.cell(position))
        if old_cost is not None and new_cost is not None and new_cost < old_cost-.1:
            self.last_progress_time = stamp
            self.progress_anchor = position.copy()

    def scan(self):
        e=self.env;p=e.data.qpos[:3];yaw=math.atan2(e.data.xmat[1].reshape(3,3)[1,0],e.data.xmat[1].reshape(3,3)[0,0])
        angles=np.linspace(-math.pi,math.pi,120,endpoint=False)+yaw
        all_ranges=[];all_angles=[];hits=[];geom=np.zeros(1,np.int32)
        ground=p[2]-.43
        for z in (ground+.10,ground+.27,ground+.48):
            for angle in angles:
                d=mujoco.mj_ray(e.model,e.data,np.array([p[0],p[1],z]),np.array([math.cos(angle),math.sin(angle),0.]),self.mask,1,-1,geom)
                hit=0<=d<3.2 and geom[0]!=0
                all_ranges.append(min(d,3.2) if d>=0 else 3.2);all_angles.append(angle);hits.append(hit)
        self.grid.update(p[:2],all_angles,all_ranges,hits,e.data.time)

    def step(self):
        e=self.env;t=e.data.time
        self.motion_window.append(e.data.qpos[:2].copy())
        goal_distance=float(np.linalg.norm(e.data.qpos[:2]-self.goal))
        if not self.progress_along_route and goal_distance<self.best_distance-.10:
            self.best_distance=goal_distance;self.last_progress_time=t
        if not self.progress_along_route and t-self.last_progress_time>10:self.blocked=True
        if len(self.motion_window)==400:
            window=np.asarray(self.motion_window)
            if np.linalg.norm(np.ptp(window,axis=0))<.25:self.blocked=True
        if self.scenario=='dynamic' and t>=6 and e.model.geom_pos[self.dynamic_id,2]<0:
            e.model.geom_pos[self.dynamic_id,2]=.8;mujoco.mj_forward(e.model,e.data)
        lost=self.scenario=='dropout' and 5<=t<7
        if e.k%5==0 and not lost:self.scan()
        if e.k%25==0 and not lost:
            self.path=self.plan_route();self.replans+=1
            if self.progress_along_route:
                self.update_route_progress(e.data.qpos[:2],t)
            # 空路径恢复: 逐步缩小膨胀半径直到找到可行路径
            if not self.path and not self.blocked:
                for shrink in (.42,.36,.30):
                    self.path=self.grid.route(e.data.qpos[:2],self.goal,radius=shrink)
                    if self.path:
                        self.replans+=1
                        break
                if not self.path:
                    # 仍然无路: 朝已知自由空间方向后退
                    free=self.grid.known&~self.grid.occupied
                    best_back=None;best_clear=0
                    cell=self.grid.cell(e.data.qpos[:2])
                    for dr in range(-5,6):
                        for dc in range(-5,6):
                            nr,nc=cell[0]+dr,cell[1]+dc
                            if not(0<=nr<free.shape[0] and 0<=nc<free.shape[1]) or not free[nr,nc]:
                                continue
                            clear=math.hypot(dr,dc)
                            if clear>best_clear:
                                best_clear=clear;best_back=self.grid.point((nr,nc))
                    if best_back is not None:
                        self.path=[best_back.tolist(),
                                   [e.data.qpos[0],e.data.qpos[1]]]
        if self.progress_along_route and t-self.last_progress_time>10:
            self.blocked=True
        R=e.data.xmat[1].reshape(3,3);yaw=math.atan2(R[1,0],R[0,0])
        command=self.follower.command(e.data.qpos[:2],yaw,self.path,self.goal)
        stale=t-self.grid.last_update>.20
        if stale:command[:]=0
        v,w,g=e.state();actual=np.r_[v[:2],w[2]]
        stopped=self.blocked or stale or not self.path or np.linalg.norm(e.data.qpos[:2]-self.goal)<.35
        residual=None
        if self.tracking_policy is not None and not stopped:
            from forest_tracking_residual import tracking_observation
            obs=tracking_observation(command,self.filtered,g,self.integral,t*2*np.pi*1.8)
            residual=self.tracking_policy.predict(obs,deterministic=True)[0]
        self.filtered=.96*self.filtered+.04*actual
        error=command-self.filtered
        self.integral=np.clip(self.integral+error*.02,-.15,.15) if not stale else np.zeros(3)
        drive=command+.35*error+.5*self.integral
        # Measured generic-gait fore/aft bias while swinging (see calibration).
        # Feed-forward is explicitly model-specific, not a real-robot constant.
        if np.linalg.norm(command)>.03:drive[0]+=.12
        if residual is not None:
            from forest_tracking_residual import tracking_drive
            drive=tracking_drive(command,self.filtered,self.integral,residual)
        else:
            drive=np.clip(drive,[-.25,-.22,-.65],[.5,.22,.65])
        if stopped:drive[:]=0;self.integral[:]=0
        if self.blocked:self.blocked_steps+=1
        e.command=command
        q=targets(t,drive)
        _,_,term,trunc,info=e.step(np.clip((q-e.q0)/.4,-1,1))
        self.energy+=info['energy_j'];self.stop_steps+=int(stale)
        if self.last_position is not None:self.travel+=float(np.linalg.norm(e.data.qpos[:2]-self.last_position))
        self.last_position=e.data.qpos[:2].copy()
        collision=False
        for c in e.data.contact:
            names=[mujoco.mj_id2name(e.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in (c.geom1,c.geom2)]
            if any(n.startswith('nav_') for n in names) and (e.model.geom_bodyid[c.geom1]>0 or e.model.geom_bodyid[c.geom2]>0):collision=True
        distance=float(np.linalg.norm(e.data.qpos[:2]-self.goal))
        success=distance<.35 and not term and not collision and not self.blocked
        self.traces.append(dict(t=e.data.time,position=e.data.qpos[:3].tolist(),yaw=yaw,
            command=command.tolist(),actual=actual.tolist(),stale=stale,path=self.path if e.k%25==1 else None))
        return bool(success or term or trunc or collision or self.blocked_steps>=25),dict(success=success,fallen=term,collision=collision,blocked=self.blocked,
            timeout=trunc,distance_m=distance,replans=self.replans,steps=e.k,stale_stop=stale,
            stop_steps=self.stop_steps,energy_j=self.energy,path_m=self.travel,no_route=not bool(self.path))

    def close(self):self.env.close()
