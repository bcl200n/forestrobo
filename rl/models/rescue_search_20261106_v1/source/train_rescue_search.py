"""Auditable multi-map PPO experiment; task-level kinematics, no hardware."""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from rescue_search_env import RescueSearchEnv, SCHEMA, SCENARIOS

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def now():
    return datetime.now(timezone.utc).isoformat()


def make_env(rank,worlds):
    return Monitor(RescueSearchEnv(scenario=SCENARIOS[rank%3],
        map_seeds=range(100+rank*16,116+rank*16),seed=2026+rank))


def evaluate(model, seeds, out, label):
    rows=[]
    replay=None
    for scenario in SCENARIOS:
        for seed in seeds:
            env=RescueSearchEnv(scenario=scenario,map_seeds=[seed],seed=seed)
            obs,info=env.reset(seed=seed,options={'map_seed':seed})
            frames=[env.snapshot()]
            reward=0.
            for t in range(env.max_steps):
                action=env.baseline_action() if model is None else int(model.predict(obs,deterministic=True)[0])
                obs,r,term,trunc,info=env.step(action)
                reward+=r
                if t%4==0 or term or trunc:
                    frames.append(env.snapshot())
                if term or trunc:
                    break
            rows.append(dict(**info,episode_return=reward,terminated=term,truncated=trunc))
            if replay is None:
                replay=dict(schema=SCHEMA,controller=label,scope='2-D grid kinematics; no leg/arm dynamics',
                            scenario=scenario,map_seed=seed,frames=frames)
            env.close()
    report=dict(schema=SCHEMA,label=label,updated_at=now(),episodes=rows,total=len(rows),
                successes=sum(r['success'] for r in rows),
                mean_found=float(np.mean([r['found'] for r in rows])),
                mean_coverage=float(np.mean([r['coverage'] for r in rows])),
                collision_episodes=sum(r['collisions']>0 for r in rows),
                unsafe_motion=sum(r['unsafe_motion'] for r in rows),
                limitation='Synthetic cue search on 2-D configuration-space maps; not life detection or rubble traversal.')
    write_json(out/f'{label}.json',report)
    write_json(out/f'{label}-replay.json',replay)
    return report


class Progress(BaseCallback):
    def __init__(self,out,total,protocol,publish_status=False):
        super().__init__()
        self.out=out
        self.total=total
        self.protocol=protocol
        self.start=time.perf_counter()
        self.last=0.
        self.next_checkpoint=32768
        self.next_validation=65536
        self.best_score=-np.inf
        self.recent=[]
        self.latest_validation=None
        self.last_checkpoint=None
        self.stopped=False
        self.publish_status=publish_status

    def status(self,state='training',**extra):
        elapsed=time.perf_counter()-self.start
        recent=self.recent[-100:]
        value=dict(schema=SCHEMA,status=state,pid=os.getpid(),updated_at=now(),
            actual_steps=self.num_timesteps,target_steps=self.total,worlds=self.protocol['worlds'],
            elapsed_s=elapsed,fps=self.num_timesteps/max(elapsed,.01),
            recent_episodes=len(recent),
            recent_success_rate=float(np.mean([r['success'] for r in recent])) if recent else None,
            latest_validation=self.latest_validation,last_checkpoint=self.last_checkpoint,
            scope=self.protocol['scope'],**extra)
        write_json(self.out/'status.json',value)
        if self.publish_status:
            target=ROOT/'web/rescue-training.js'
            temp=target.with_suffix('.js.tmp')
            temp.write_text('globalThis.RESCUE_TRAINING = '+json.dumps(value,ensure_ascii=False)+';\n',encoding='utf-8')
            os.replace(temp,target)

    def _on_step(self):
        for done,info in zip(self.locals.get('dones',[]),self.locals.get('infos',[])):
            if done:
                row={k:info.get(k) for k in ('success','found','candidates','collisions','coverage','scenario','map_seed','steps')}
                row['training_step']=self.num_timesteps
                self.recent.append(row)
                with (self.out/'episodes.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(row)+'\n')
        if self.num_timesteps>=self.next_checkpoint:
            self.last_checkpoint=f'checkpoint_{self.num_timesteps}.zip'
            self.model.save(self.out/self.last_checkpoint)
            self.next_checkpoint+=32768
        if self.num_timesteps>=self.next_validation:
            report=evaluate(self.model,[5000,5001],self.out,'validation')
            self.latest_validation={k:report[k] for k in ('successes','total','mean_found','collision_episodes')}
            score=report['successes']*100+report['mean_found']*10-report['collision_episodes']
            if score>self.best_score:
                self.best_score=score
                self.model.save(self.out/'best_validation.zip')
            self.next_validation+=131072
        if time.perf_counter()-self.last>10:
            self.last=time.perf_counter()
            self.status()
            print(json.dumps(dict(steps=self.num_timesteps,worlds=self.protocol['worlds'],
                                  fps=round(self.num_timesteps/max(self.last-self.start,.01),1))),flush=True)
        if (self.out/'STOP').exists():
            self.stopped=True
            self.model.save(self.out/'stopped.zip')
            self.status('stopped')
            return False
        return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=2_000_000)
    parser.add_argument('--worlds',type=int,default=8)
    parser.add_argument('--output',required=True)
    parser.add_argument('--publish-status',action='store_true')
    args=parser.parse_args()
    if args.steps<1 or not 3<=args.worlds<=16:
        parser.error('positive steps and 3..16 worlds required')
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        parser.error('Use a fresh output directory; prior experiments are never overwritten')
    protocol=dict(schema=SCHEMA,created_at=now(),worlds=args.worlds,target_steps=args.steps,
        scope='Task-level 2-D grid search; NOT quadruped locomotion, arm control or real life detection',
        training_map_seeds=list(range(100,100+args.worlds*16)),
        scenario_by_worker=[SCENARIOS[i%3] for i in range(args.worlds)],
        validation_map_seeds=[5000,5001],test_map_seeds=[10000,10001,10002,10003],
        actor='39 observed features: rays, observed adjacency/visits/frontier, visible cue, validity and time',
        reward='new observations + frontier-following shaping + scanning synthetic cues - collision/time/revisit',
        human_role='All candidate cues await human review; no autonomous cutting or actuation',
        idealizations=['grid/compass odometry','configuration-space collision map','synthetic cue visibility',
                       'no SLAM or motor dynamics','synthetic communication/sensor dropout'],
        hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                for name in ('rescue_search_env.py','train_rescue_search.py')})
    write_json(out/'protocol.json',protocol)
    write_json(out/'status.json',dict(status='baseline_evaluation',actual_steps=0,pid=os.getpid(),updated_at=now(),**protocol))
    torch.set_num_threads(2)
    env=None
    callback=None
    try:
        baseline=evaluate(None,protocol['test_map_seeds'],out,'baseline')
        env=SubprocVecEnv([partial(make_env,i,args.worlds) for i in range(args.worlds)],start_method='spawn')
        model=PPO('MlpPolicy',env,seed=2026,device='cpu',n_steps=256,batch_size=256,n_epochs=4,
                  learning_rate=3e-4,ent_coef=.015,policy_kwargs=dict(net_arch=[64,64]),verbose=0)
        callback=Progress(out,args.steps,protocol,publish_status=args.publish_status)
        callback.status()
        model.learn(total_timesteps=args.steps,callback=callback)
        if callback.stopped:
            return
        model.save(out/'final.zip')
        callback.status('test_evaluation')
        learned=evaluate(model,protocol['test_map_seeds'],out,'evaluation')
        promoted=(learned['successes']>baseline['successes'] and
                  learned['collision_episodes']<=baseline['collision_episodes'] and learned['unsafe_motion']==0)
        callback.status('complete',baseline_successes=baseline['successes'],
                        learned_successes=learned['successes'],test_cases=learned['total'],
                        promoted=promoted,deployment_ready=False)
        print('EXPERIMENT_COMPLETE',json.dumps(dict(promoted=promoted,learned=learned['successes'],baseline=baseline['successes'])),flush=True)
    except BaseException as exc:
        write_json(out/'status.json',dict(status='failed',error=repr(exc),updated_at=now(),pid=os.getpid(),
                                        actual_steps=callback.num_timesteps if callback else 0))
        raise
    finally:
        if env is not None:
            env.close()


if __name__=='__main__':
    main()
