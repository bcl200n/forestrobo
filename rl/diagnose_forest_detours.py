"""Paired simulation-only map and route interventions for blocked runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np

from forest_autonomy import ForestAutonomy
from robotics.terrain_planner import plan
from run_forest_experiment import write_json


MODES = ('observed_map', 'truth_map', 'checked_route')
GEODESIC = 'geodesic_observed'
CLEARANCE = 'clearance_observed'
WEIGHTED = 'weighted_local_observed'
ROUTE_PROGRESS = 'weighted_route_progress'
FOOTPRINT_M = .48


def segment_distance(points, start, end):
    delta = end-start
    if np.dot(delta,delta) == 0:
        return np.linalg.norm(points-start,axis=-1)
    along = np.clip(np.sum((points-start)*delta,axis=-1)/np.dot(delta,delta),0.,1.)
    return np.linalg.norm(points-start-along[...,None]*delta,axis=-1)


class DiagnosticForest(ForestAutonomy):
    def __init__(self, scenario, seed, mode):
        self.mode = mode
        self.reference = None
        self.reference_clearance = None
        super().__init__(scenario,seed,geodesic=mode==GEODESIC,clearance_bias=mode==CLEARANCE,
                         weighted_local=mode in (WEIGHTED,ROUTE_PROGRESS),
                         progress_along_route=mode==ROUTE_PROGRESS)

    def footprints(self, include_future=False):
        physics = self.env
        for index in range(physics.model.ngeom):
            name = mujoco.mj_id2name(physics.model,mujoco.mjtObj.mjOBJ_GEOM,index) or ''
            if not name.startswith('nav_'):
                continue
            x,y,z = physics.data.geom_xpos[index]
            if z < -1 and not (include_future and self.scenario=='dynamic' and name=='nav_dynamic'):
                continue
            center = np.array([x,y])
            radius, half_length, _ = physics.model.geom_size[index]
            offset = np.zeros(2)
            if physics.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                offset = physics.data.geom_xmat[index].reshape(3,3)[:2,2]*half_length
            yield center-offset,center+offset,float(radius)

    def occupancy(self, footprints):
        rows,cols = np.indices(self.grid.known.shape)
        centers = self.grid.origin + (np.stack((cols,rows),axis=-1)+.5)*self.grid.resolution
        occupied = np.zeros(rows.shape,dtype=bool)
        for start,end,radius in footprints:
            occupied |= segment_distance(centers,start,end) <= radius+self.grid.resolution/np.sqrt(2)
        return occupied

    def scan(self):
        if self.mode != 'truth_map':
            return super().scan()
        self.grid.known.fill(True)
        self.grid.occupied[:] = self.occupancy(self.footprints())
        self.grid.last_update = self.env.data.time

    def plan_route(self):
        if self.mode != 'checked_route':
            return super().plan_route()
        if self.reference is None:
            # Fixed diagnostic fixture for tracking. Its geometry includes the
            # scheduled dynamic obstacle so that the route stays clear later.
            obstacles = list(self.footprints(include_future=True))
            grid = self.grid
            start = self.env.data.qpos[:2].copy()
            result = plan(np.zeros(grid.known.shape), self.occupancy(obstacles),
                          grid.cell(start),grid.cell(self.goal),grid.resolution,FOOTPRINT_M)
            if not result['found']:
                raise RuntimeError(f'No reference fixture for {self.scenario}/{self.seed}: {result["reason"]}')
            points = np.array([start,*[grid.point(cell) for cell in result['path']],self.goal])
            spacing = .02
            samples = np.concatenate([np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/spacing))+1))
                                      for a,b in zip(points,points[1:])])
            clearance = min(float(segment_distance(samples,a,b).min()-r) for a,b,r in obstacles)-spacing/2
            if clearance < FOOTPRINT_M:
                raise RuntimeError(f'Invalid fixture clearance for {self.scenario}/{self.seed}: {clearance:.3f} m')
            self.reference_clearance = clearance
            self.reference = points.tolist()
        return self.reference


def run_case(scenario,seed,mode,out):
    sim = DiagnosticForest(scenario,seed,mode)
    try:
        for _ in range(sim.env.max_steps):
            done,result = sim.step()
            if done:
                break
        row = dict(scenario=scenario,seed=seed,mode=mode,
                   reference_clearance_m=sim.reference_clearance,**result)
        write_json(out/f'{scenario}-{seed}-{mode}.json',dict(result=row,trace=sim.traces[::5]))
        return row
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[10110,10111,10112,10113])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--modes',nargs='+',choices=(*MODES,GEODESIC,CLEARANCE,WEIGHTED,ROUTE_PROGRESS),default=MODES)
    parser.add_argument('--scenarios',nargs='+',choices=('detour','dynamic','corridor','slope','dropout'),
                        default=('detour','dynamic'))
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or len(set(args.seeds)) != len(args.seeds):
        parser.error('require 1–8 workers and unique seeds')
    args.output.mkdir(parents=True,exist_ok=False)
    trials = [(scenario,seed,mode) for scenario in args.scenarios for seed in args.seeds for mode in args.modes]
    protocol = dict(seeds=args.seeds,scenarios=list(args.scenarios),modes=list(args.modes),
                    planned_cases=len(trials),footprint_m=FOOTPRINT_M,
                    actuator='same Cartesian gait, PI, follower, watchdog, and physical contacts',
                    truth_map='current-time simulator footprints, same local planner and scan cadence',
                    checked_route='fixed clearance-checked A* polyline including scheduled dynamic obstacle',
                    clearance_bias='soft obstacle cost in an additional 0.24 m band',
                    weighted_local='apply the same cost to the executable prefix inside the reachable set',
                    route_progress='compare anchor and current pose costs on one map revision')
    write_json(args.output/'protocol.json',protocol)
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run_case,*case,args.output) for case in trials]
            for job in as_completed(jobs):
                row = job.result()
                rows.append(row)
                print(row,flush=True)
                write_json(args.output/'summary.json',dict(status='running',episodes=rows))
        rows.sort(key=lambda row:(row['scenario'],row['seed'],args.modes.index(row['mode'])))
        scores = {}
        for mode in args.modes:
            group = [row for row in rows if row['mode']==mode]
            scores[mode] = dict(cases=len(group),arrivals=sum(row['success'] for row in group),
                                blocked=sum(row['blocked'] for row in group),
                                collisions=sum(row['collision'] for row in group),falls=sum(row['fallen'] for row in group))
        write_json(args.output/'summary.json',dict(status='complete',results=scores,episodes=rows))
        print(scores,flush=True)
    except BaseException as exc:
        write_json(args.output/'summary.json',dict(status='failed',error=repr(exc),episodes=rows))
        raise


if __name__ == '__main__':
    main()
