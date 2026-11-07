"""Train a provisional foot teacher; promotion is gated by held-out tracking.

Results are not comparable to navigation success rates. No student is trained
when this velocity gate fails. Old experiments remain untouched.
"""
import argparse
from pathlib import Path
import time
import hashlib
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from foot_teacher_env import FootTeacherEnv,SCHEMA
from run_forest_experiment import write_json


def evaluate(policy,out):
    rows=[]
    commands=[[0,0,0],[.3,0,0],[-.15,0,0],[0,0,.4],[0,0,-.4]]
    for slope in (-5.,0.,5.):
        env=FootTeacherEnv(slope_deg=slope,max_steps=250)
        try:
            for command in commands:
                for seed in (10001,10002):
                    obs,_=env.reset(seed=seed,options={'command':command})
                    errors=[];fallen=False;energy=0
                    for step in range(250):
                        a=np.zeros(16,np.float32) if policy is None else policy.predict(obs,deterministic=True)[0]
                        obs,_,term,trunc,info=env.step(a)
                        if step>=25:errors.append(info['tracking_error'])
                        energy+=info['energy_j'];fallen=info['fallen']
                        if term or trunc:break
                    rmse=np.sqrt(np.mean(np.asarray(errors)**2,axis=0)).tolist() if errors else [None]*3
                    passed=bool(not fallen and step==249 and errors and rmse[0]<.10 and rmse[1]<.10 and rmse[2]<.15)
                    rows.append(dict(slope_deg=slope,command=command,seed=seed,steps=step+1,
                                     fallen=fallen,rmse=rmse,passed=passed,energy_j=energy))
        finally:env.close()
    report=dict(schema=SCHEMA,scope='simulated plane/slope velocity task only',episodes=rows,
        pass_count=sum(r['passed'] for r in rows),count=len(rows),
        thresholds=dict(vx_rmse_m_s=.10,vy_rmse_m_s=.10,yaw_rmse_rad_s=.15),
        promoted=all(r['passed'] for r in rows),
        warning='Project-proposed initial gate, not a scientific or hardware safety standard.')
    write_json(out,report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=65536)
    p.add_argument('--output',default=str(Path(__file__).resolve().parent/'models/velocity_teacher_v1'))
    args=p.parse_args()
    if args.steps<1:p.error('positive steps required')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'final.zip').exists():p.error('Output already has a policy; use a new output directory')
    torch.set_num_threads(2);start=time.perf_counter()
    env=DummyVecEnv([lambda:Monitor(FootTeacherEnv()) for _ in range(4)])
    try:
        write_json(out/'status.json',dict(status='training',schema=SCHEMA))
        model=PPO('MlpPolicy',env,seed=2026,n_steps=512,batch_size=256,n_epochs=5,
            policy_kwargs=dict(net_arch=[128,128]),device='cpu',verbose=1)
        model.learn(total_timesteps=args.steps,callback=CheckpointCallback(
            save_freq=8192,save_path=str(out/'checkpoints')))
        model.save(str(out/'final'))
        write_json(out/'protocol.json',dict(schema=SCHEMA,seed=2026,actual_steps=model.num_timesteps,
            training_slopes=[0],evaluation_slopes=[-5,0,5],evaluation_seeds=[10001,10002],
            files_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                for name in ('velocity_env.py','velocity_primitive.py','foot_teacher_env.py','train_velocity_teacher.py')}))
        write_json(out/'status.json',dict(status='evaluating',schema=SCHEMA))
        baseline=evaluate(None,out/'baseline.json')
        learned=evaluate(model,out/'evaluation.json')
        write_json(out/'status.json',dict(status='complete',actual_steps=model.num_timesteps,
            baseline_pass_count=baseline['pass_count'],learned_pass_count=learned['pass_count'],
            cases=learned['count'],promoted=learned['promoted'],
            next_action='student_distillation_not_started' if not learned['promoted'] else 'further_rough_terrain_validation_required',
            elapsed_s=time.perf_counter()-start))
    except BaseException as exc:
        write_json(out/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:env.close()


if __name__=='__main__':main()
