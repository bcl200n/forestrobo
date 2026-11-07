"""Record actual wheel forward/yaw responses before trusting a velocity interface."""
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rl'))
from forest_v3 import ForestHistoryEnv


def main():
    rows=[]
    for mode in ('forward','yaw'):
        env=ForestHistoryEnv(embodiment='wheel',scenario='nominal',randomize=False,max_steps=750)
        try:
            env.reset(seed=1)
            start=env.unwrapped.data.qpos[:2].copy()
            heading=env.unwrapped._yaw()
            speeds=[]
            for step in range(150):
                action=np.zeros(16,np.float32)
                action[12:]=.2 if mode=='forward' else np.array([.15,-.15,.15,-.15])
                obs,_,terminated,truncated,info=env.step(action)
                speeds.append(float(obs[-env.frame_dim]))
                if terminated or truncated:break
            yaw=env.unwrapped._yaw()
            rows.append(dict(mode=mode,steps=step+1,forward_speed=float(np.mean(speeds)),
                displacement=float(np.linalg.norm(env.unwrapped.data.qpos[:2]-start)),
                yaw_change=math.atan2(math.sin(yaw-heading),math.cos(yaw-heading)),failure=info['failure']))
        finally:env.close()
    target=Path(__file__).resolve().parents[1]/'design/low_level_diagnostics.json'
    target.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
