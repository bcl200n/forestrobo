"""Export actual MuJoCo playback and offline demo evidence; no fabricated motion."""
from pathlib import Path
import sys
import json
import math
import numpy as np
import imageio.v2 as imageio
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'rl')]
from velocity_env import VelocityEnv
from velocity_primitive import targets
from robotics.manipulator import static_check,CATALOG
from robotics.terrain_planner import plan
import gait


def video(name,corrected):
    env=VelocityEnv('foot',max_steps=400)
    frames=[];out=ROOT/'web/demo-assets';out.mkdir(exist_ok=True)
    renderer=None
    try:
        env.reset(seed=11,options={'command':[.3,0,0]})
        renderer=mujoco.Renderer(env.model,height=540,width=960)
        camera=mujoco.MjvCamera();camera.distance=1.7;camera.azimuth=125;camera.elevation=-18
        with imageio.get_writer(str(out/f'{name}.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=2) as writer:
            for step in range(400):
                q=targets(env.data.time,env.command,env.q0) if corrected else gait.cpg(env.data.time)
                _,_,terminated,truncated,info=env.step(np.clip((q-env.q0)/.4,-1,1))
                if step%2==0:
                    camera.lookat[:]=env.data.qpos[:3]+[0,0,-.05]
                    renderer.update_scene(env.data,camera=camera)
                    pixels=renderer.render();writer.append_data(pixels)
                    if step==100:imageio.imwrite(str(out/f'{name}.jpg'),pixels)
                frames.append(dict(t=(step+1)*.02,vx=info['actual'][0],vy=info['actual'][1],yaw=info['actual'][2]))
                if terminated or truncated:break
        steady=frames[25:]
        return dict(name=name,corrected=corrected,command=.3,seed=11,frames=frames,
            duration_s=len(frames)*.02,mean_vx=float(np.mean([x['vx'] for x in steady])),
            vx_rmse=float(np.sqrt(np.mean([(x['vx']-.3)**2 for x in steady]))),fallen=info['fallen'])
    finally:
        if renderer:renderer.close()
        env.close()


def main():
    replay=[video('legacy',False),video('corrected',True)]
    maps=[]
    for name in ('clear','blocked','uncertain'):
        h=np.zeros((36,54));obstacles=np.zeros_like(h)
        obstacles[7:24,23:28]=1;obstacles[22:28,35:40]=1
        if name=='blocked':obstacles[26:36,23:28]=1
        uncertainty=np.zeros_like(h)
        if name=='uncertain':uncertainty[1:10,20:32]=.3
        result=plan(h,obstacles,(18,6),(18,47),resolution_m=.1,footprint_radius_m=.3,height_uncertainty_m=uncertainty)
        maps.append(dict(name=name,rows=36,cols=54,occupied=obstacles.astype(int).tolist(),
            uncertain=(uncertainty>.08).astype(int).tolist(),result=result,start=[18,6],goal=[18,47]))
    arms=[]
    for arm in ('light4','work5'):
        for pose,q in [('extended',np.zeros(CATALOG['arms'][arm]['dof'])),
            ('folded',np.array([0,-1.05,1.25,0] if arm=='light4' else [0,-1.05,1.25,-.2,0]))]:
            mass=1.2 if arm=='light4' else 2.3
            arms.append(dict(arm=arm,pose=pose,mass=mass,rated=CATALOG['arms'][arm]['rated_torque_nm'],
                zero=static_check(q,tool_mass_kg=mass,arm=arm),
                force500=static_check(q,force_n=(0,0,500),tool_mass_kg=mass,arm=arm)))
    evidence=json.loads((ROOT/'rl/models/velocity_teacher_v1/height_fix_review.json').read_text())
    data=dict(replays=replay,maps=maps,arms=arms,teacher=evidence)
    (ROOT/'web/demo-data.js').write_text('globalThis.DEMO_DATA='+json.dumps(data,ensure_ascii=False)+';',encoding='utf-8')
    print('Demo assets built:',len(replay),'actual simulation recordings')


if __name__=='__main__':main()
