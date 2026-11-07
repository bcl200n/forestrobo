"""Local mapping/planning/tracking logic, independent of simulation or motors.

The caller supplies pose and time-stamped range beams. Executable local paths
block unknown cells. An optimistic global guide may cross unknown space only
to rank observed frontiers. This is not SLAM: pose is a separate dependency.
"""
import math
import heapq
import numpy as np
from robotics.terrain_planner import plan,inflate


class LocalGrid:
    def __init__(self,origin=(-2.,-4.),shape=(40,60),resolution=.2):
        self.origin=np.asarray(origin,float);self.resolution=resolution
        self.known=np.zeros(shape,bool);self.occupied=np.zeros(shape,bool)
        self.last_update=-float('inf')

    def cell(self,point):
        col,row=np.floor((np.asarray(point)-self.origin)/self.resolution).astype(int)
        return int(row),int(col)

    def point(self,cell):
        row,col=cell
        return self.origin+(np.array([col,row])+.5)*self.resolution

    def inside(self,cell):return 0<=cell[0]<self.known.shape[0] and 0<=cell[1]<self.known.shape[1]

    def update(self,position,angles,ranges,hits,stamp):
        if not np.isfinite(stamp) or stamp<self.last_update:raise ValueError('invalid/old stamp')
        angles,ranges,hits=np.asarray(angles),np.asarray(ranges),np.asarray(hits)
        if angles.ndim!=1 or angles.shape!=ranges.shape or ranges.shape!=hits.shape or not np.isfinite(angles).all() or not np.isfinite(position).all() or not np.isfinite(ranges).all() or (ranges<0).any():raise ValueError('invalid beams')
        marked=[]
        for angle,distance,hit in zip(angles,ranges,hits):
            direction=np.array([math.cos(angle),math.sin(angle)])
            for d in np.arange(0,max(0,distance-self.resolution*.5),self.resolution*.4):
                c=self.cell(np.asarray(position)+d*direction)
                if self.inside(c):self.known[c]=True
            c=self.cell(np.asarray(position)+distance*direction)
            if hit and self.inside(c):marked.append(c)
        for c in marked:self.known[c]=True;self.occupied[c]=True
        self.last_update=stamp

    def route(self,position,goal,radius=.48):
        start=self.cell(position)
        if not self.inside(start):return []
        # Optimistic global guide helps escape Euclidean local minima. It is NOT
        # executable: every commanded local route below stays in observed space.
        goal_cell=self.cell(goal)
        if not self.inside(goal_cell):return []
        guide=plan(np.zeros(self.known.shape),self.occupied.astype(float),start,goal_cell,
                   self.resolution,radius)
        if not guide['found']:return []
        guide_points=np.array([self.point(c) for c in guide['path']])
        lengths=np.linalg.norm(np.diff(guide_points,axis=0),axis=1)
        remaining=np.r_[np.cumsum(lengths[::-1])[::-1],0.]
        blocked=inflate(~self.known|self.occupied,self.resolution,radius)
        if blocked[start]:return []
        costs={start:0.};parent={};queue=[(0.,start)]
        while queue:
            cost,cell=heapq.heappop(queue)
            if cost>costs[cell]+1e-9:continue
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if not(dr or dc):continue
                    n=cell[0]+dr,cell[1]+dc
                    if not self.inside(n) or blocked[n]:continue
                    if dr and dc and (blocked[cell[0]+dr,cell[1]] or blocked[cell[0],cell[1]+dc]):continue
                    new=cost+self.resolution*math.hypot(dr,dc)
                    if new>1.8 or new>=costs.get(n,float('inf')):continue
                    costs[n]=new;parent[n]=cell;heapq.heappush(queue,(new,n))
        best=None;score=float('inf')
        for cell,cost in costs.items():
            if cost<.4:continue
            distance=np.linalg.norm(guide_points-self.point(cell),axis=1)
            value=float(np.min(distance*1.5+remaining))
            if value<score:best,score=cell,value
        if best is None:return []
        path=[best]
        while path[-1]!=start:path.append(parent[path[-1]])
        return [self.point(c).tolist() for c in path[::-1]]


class WaypointFollower:
    def command(self,position,yaw,path,goal):
        position=np.asarray(position,float)
        if np.linalg.norm(position-np.asarray(goal))<.35:return np.zeros(3)
        if not path:return np.zeros(3)
        points=np.asarray(path)
        nearest=int(np.argmin(np.linalg.norm(points-position,axis=1)))
        target=points[-1]
        for point in points[nearest:]:
            if np.linalg.norm(point-position)>.55:target=point;break
        error=target-position
        angle=math.atan2(error[1],error[0])
        yaw_error=(angle-yaw+math.pi)%(2*math.pi)-math.pi
        # Local frontiers are not final stop targets: very low commands caused
        # the gait to dither before a turn. Only the global arrival gate stops.
        speed=.22
        # Holonomic translation avoids relying on a sharp skid-steer turn.
        world=error/max(np.linalg.norm(error),1e-9)*speed
        body=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])@world
        return np.r_[body,float(np.clip(yaw_error*.5,-.3,.3))]
