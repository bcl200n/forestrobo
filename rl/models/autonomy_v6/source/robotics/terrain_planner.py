"""Conservative grid A* over a caller-supplied local elevation/obstacle map.

Coordinates are row/column cell centers. Unknown cells and footprint-inflated
obstacles are blocked. No mapping, localization or gait execution is provided.
"""
import heapq
import math
import numpy as np


def inflate(blocked,resolution_m,radius_m):
    """Disc plus half-cell diagonal accounts for occupied cell extent."""
    blocked=np.asarray(blocked,bool)
    distance=radius_m/resolution_m+math.sqrt(2)/2
    margin=int(math.ceil(distance))
    padded=np.pad(blocked,margin,constant_values=True)
    rows,cols=blocked.shape;result=np.zeros_like(blocked)
    for dr in range(-margin,margin+1):
        for dc in range(-margin,margin+1):
            if math.hypot(dr,dc)<=distance:
                result|=padded[margin+dr:margin+dr+rows,margin+dc:margin+dc+cols]
    return result


def plan(height_m, occupied, start, goal, resolution_m=.1, footprint_radius_m=.45,
         max_slope_deg=20., height_uncertainty_m=None, max_uncertainty_m=.08):
    heights=np.asarray(height_m,float)
    obstacles=np.asarray(occupied)
    if heights.ndim!=2 or heights.size>250000 or min(heights.shape)<2 or obstacles.shape!=heights.shape:
        raise ValueError('Expected matching 2-D maps with at most 250000 cells')
    if not (math.isfinite(resolution_m) and resolution_m>0 and math.isfinite(footprint_radius_m)
            and footprint_radius_m>=0 and 0<max_slope_deg<60):
        raise ValueError('Invalid map/robot limits')
    unknown=~np.isfinite(heights)|~np.isfinite(obstacles)
    blocked=unknown|(obstacles>0)
    if height_uncertainty_m is not None:
        uncertainty=np.asarray(height_uncertainty_m,float)
        if uncertainty.shape!=heights.shape:
            raise ValueError('Uncertainty shape mismatch')
        blocked|=~np.isfinite(uncertainty)|(uncertainty<0)|(uncertainty>max_uncertainty_m)
    # Zero-radius point-robot tests retain their original cell semantics.
    safe_blocked=inflate(blocked,resolution_m,footprint_radius_m) if footprint_radius_m else blocked.copy()
    rows,cols=heights.shape
    start,goal=tuple(start),tuple(goal)
    for cell in (start,goal):
        if len(cell)!=2 or not all(isinstance(x,(int,np.integer)) for x in cell):
            raise ValueError('Cells must be integer row/column pairs')
        if not(0<=cell[0]<rows and 0<=cell[1]<cols):
            raise ValueError('Cell outside map')
    if safe_blocked[start] or safe_blocked[goal]:
        return dict(found=False,path=[],reason='start_or_goal_blocked')
    slope_limit=math.tan(math.radians(max_slope_deg))
    costs={start:0.};parents={};queue=[(0.,0.,start)]
    visited=set()
    while queue:
        _,cost,current=heapq.heappop(queue)
        if current in visited:continue
        visited.add(current)
        if current==goal:
            path=[goal]
            while path[-1]!=start:path.append(parents[path[-1]])
            path.reverse()
            return dict(found=True,path=[list(p) for p in path],reason='planned',
                        cost_m=float(cost),visited_cells=len(visited))
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if not(dr or dc):continue
                neighbor=current[0]+dr,current[1]+dc
                nr,nc=neighbor
                if not(0<=nr<rows and 0<=nc<cols) or safe_blocked[neighbor]:continue
                if dr and dc and (safe_blocked[current[0]+dr,current[1]] or safe_blocked[current[0],current[1]+dc]):continue
                distance=resolution_m*math.hypot(dr,dc)
                rise=abs(float(heights[neighbor]-heights[current]))
                if rise/distance>slope_limit:continue
                new_cost=cost+distance+2*rise
                if new_cost>=costs.get(neighbor,float('inf')):continue
                costs[neighbor]=new_cost;parents[neighbor]=current
                heuristic=resolution_m*math.hypot(goal[0]-nr,goal[1]-nc)
                heapq.heappush(queue,(new_cost+heuristic,new_cost,neighbor))
    return dict(found=False,path=[],reason='no_traversable_route',visited_cells=len(visited))
