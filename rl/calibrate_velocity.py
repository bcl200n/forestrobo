"""Separate floor/contact geometry effects from policy learning. No auto tuning."""
import json
from pathlib import Path
import numpy as np
from velocity_env import VelocityEnv
import gait
from velocity_primitive import targets


def evaluate(env,controller,command,seed):
    env.reset(seed=seed,options={'command':command})
    actual=[];energy=0.;fallen=False
    for step in range(250):
        action=controller(env)
        _,_,term,trunc,info=env.step(action)
        if step>=25:actual.append(info['actual'])
        energy+=info['energy_j'];fallen=info['fallen']
        if term or trunc:break
    measured=np.mean(actual,axis=0) if actual else np.zeros(3)
    return dict(command=command,seed=seed,steps=step+1,mean_actual=measured.tolist(),
        rmse=np.sqrt(np.mean((np.asarray(actual)-command)**2,axis=0)).tolist() if actual else None,
        survived=not fallen and step==249,energy_j=energy)


def open_loop(env):
    if env.wheel:
        speed,_,yaw=env.command
        action=np.zeros(16,np.float32)
        action[12:]=np.array([speed+yaw*.275,speed-yaw*.275]*2)/(.075*30)
        return np.clip(action,-1,1)
    return np.clip((gait.cpg(env.data.time)-env.q0)/.4,-1,1)


def main():
    results=[]
    for kind,shape,controller in [('foot','capsule','legacy_cpg'),('foot','capsule','corrected_primitive'),
                                 ('wheel','capsule','open_loop_differential'),('wheel','cylinder','open_loop_differential')]:
        env=VelocityEnv(kind,shape)
        try:
            for command in ([0,0,0],[.3,0,0],[0,0,.4],[0,0,-.4]):
                for seed in (11,12,13):
                    policy=(lambda e:np.clip((targets(e.data.time,e.command,e.q0)-e.q0)/.4,-1,1)) if controller=='corrected_primitive' else open_loop
                    row=evaluate(env,policy,command,seed)
                    row.update(embodiment=kind,wheel_shape=shape,controller=controller)
                    results.append(row)
        finally:env.close()
    report=dict(scope='generic simulation on exact plane, no real-hardware calibration',episodes=results)
    path=Path(__file__).resolve().parents[1]/'design/velocity_calibration.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in results[::3]:print(row['embodiment'],row['wheel_shape'],row['command'],row['mean_actual'],row['survived'])


if __name__=='__main__':main()
