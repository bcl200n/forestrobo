"""Evaluate full physics, sensor-map-plan-follow loop. Not real-world autonomy."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import shutil
import numpy as np
import mujoco
import imageio.v2 as imageio
from forest_autonomy import ForestAutonomy
from run_forest_experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def run(scenario,seed,out,render=False,*,tracking_policy=None,weighted_route=False):
    sim=ForestAutonomy(scenario,seed,tracking_policy=tracking_policy,
                       weighted_local=weighted_route,progress_along_route=weighted_route)
    renderer=None;writer=None
    try:
        if render:
            renderer=mujoco.Renderer(sim.env.model,height=540,width=960)
            camera=mujoco.MjvCamera();camera.distance=3.2;camera.azimuth=115;camera.elevation=-52
            writer=imageio.get_writer(str(out/f'{scenario}_{seed}.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=2)
        for step in range(sim.env.max_steps):
            done,result=sim.step()
            if renderer and step%2==0:
                camera.lookat[:]=sim.env.data.qpos[:3]+[.45,0,-.1]
                renderer.update_scene(sim.env.data,camera=camera);frame=renderer.render();writer.append_data(frame)
                if step==200:imageio.imwrite(str(out/f'{scenario}_{seed}.jpg'),frame)
            if done:break
        trace=sim.traces[::5]
        row=dict(scenario=scenario,seed=seed,**result)
        write_json(out/f'{scenario}_{seed}.json',dict(result=row,trace=trace,goal=sim.goal.tolist(),
            known=sim.grid.known.astype(int).tolist(),occupied=sim.grid.occupied.astype(int).tolist(),
            grid_origin=sim.grid.origin.tolist(),resolution_m=sim.grid.resolution,
            assumptions=['synthetic multilevel lidar','perfect simulator odometry','flat local height prior',
                'generic quadruped without arm', 'Cartesian gait with PI and bounded learned tracking residual'
                if tracking_policy is not None else 'non-learned Cartesian gait with PI correction']))
        scene=out/'scenes'/f'{scenario}_{seed}';scene.mkdir(parents=True,exist_ok=True)
        for name in ('forest_dog.xml','terrain_hf.bin','terrain_meta.json'):
            shutil.copyfile(Path(sim.env.assets.name)/name,scene/name)
        return row
    finally:
        if writer:writer.close()
        if renderer:renderer.close()
        sim.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'rl/models/autonomy_v1'))
    p.add_argument('--scenarios',nargs='+',default=['corridor','detour','slope','dropout','dynamic'])
    p.add_argument('--seeds',type=int,nargs='+',default=[100,101]);p.add_argument('--render',action='store_true')
    p.add_argument('--tracking-policy',type=Path,help='Candidate tracking checkpoint for explicit paired evaluation')
    p.add_argument('--weighted-route',action='store_true',
                   help='Use clearance-weighted local paths and route-cost progress checks')
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [ROOT/'rl/forest_autonomy.py',ROOT/'rl/cartesian_gait.py',ROOT/'robotics/local_navigation.py',
             ROOT/'robotics/terrain_planner.py',ROOT/'rl/velocity_env.py',ROOT/'rl/generate_model.py',ROOT/'rl/terrain.py']}
    archive=out/'source';archive.mkdir(exist_ok=True)
    for name in sources:
        target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    tracking_policy=None
    if args.tracking_policy:
        from stable_baselines3 import PPO
        tracking_policy=PPO.load(args.tracking_policy,device='cpu')
        if tracking_policy.observation_space.shape!=(14,) or tracking_policy.action_space.shape!=(3,):
            p.error('checkpoint does not match the tracking observation and action spaces')
        path=ROOT/'rl/forest_tracking_residual.py'
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path,archive/'rl/forest_tracking_residual.py')
    rows=[];started=time.perf_counter()
    for scenario in args.scenarios:
        for seed in args.seeds:
            row=run(scenario,seed,out,args.render,tracking_policy=tracking_policy,weighted_route=args.weighted_route);rows.append(row);print(json.dumps(row),flush=True)
            write_json(out/'summary.json',dict(status='running',episodes=rows))
    write_json(out/'summary.json',dict(status='complete',episodes=rows,successes=sum(r['success'] for r in rows),
        total=len(rows),elapsed_s=time.perf_counter()-started,
        controller='bounded learned tracking residual' if tracking_policy is not None else 'rule controller',
        route_mode='weighted_route_progress' if args.weighted_route else 'legacy',
        checkpoint_sha256=hashlib.sha256(args.tracking_policy.read_bytes()).hexdigest() if args.tracking_policy else None,
        limitations=['not SLAM','no real hardware','flat/3 degree clearings only',
                      'obstacles avoided, not fallen-log traversal','static accumulated occupancy does not clear removed obstacles'],
        source_sha256=sources))


if __name__=='__main__':main()
