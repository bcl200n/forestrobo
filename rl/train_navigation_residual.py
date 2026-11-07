"""Follow-up to failed direct-joint v3 experiment; saves both hybrid baselines."""
import argparse
from pathlib import Path
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from navigation_residual import NavigationResidualEnv, SCHEMA
from run_forest_experiment import evaluate, write_json, Progress


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=65536)
    p.add_argument('--worlds',type=int,default=4)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--output',default=str(Path(__file__).resolve().parent/'models/navigation_residual'))
    a=p.parse_args()
    if a.steps<=0 or not 1<=a.worlds<=32 or a.episodes<1:p.error('invalid configuration')
    torch.set_num_threads(2)
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    try:
        for kind in ('foot','wheel'):
            directory=root/kind;directory.mkdir(exist_ok=True)
            env=DummyVecEnv([lambda i=i:Monitor(NavigationResidualEnv(
                terrain_seed=300+i,profile='smooth' if i%2==0 else 'std',embodiment=kind,
                scenario='degraded'),filename=str(directory/f'world{i}.monitor.csv')) for i in range(a.worlds)])
            try:
                model=PPO('MlpPolicy',env,seed=43,n_steps=512,batch_size=256,n_epochs=5,
                    policy_kwargs=dict(net_arch=[128,128]),verbose=1,device='cpu')
                model.learn(total_timesteps=a.steps,callback=[
                    Progress(root/'status.json',kind,'hybrid',a.steps),
                    CheckpointCallback(save_freq=max(1,32768//a.worlds),save_path=str(directory/'checkpoints'))])
                model.save(str(directory/'final'))
                write_json(directory/'protocol.json',dict(schema=SCHEMA,actual_timesteps=model.num_timesteps,
                    terrain_seeds=list(range(300,300+a.worlds)),seed=43,
                    action_dim=2,baseline='same local goal controller with zero residuals',
                    limitation='CPG/differential low-level control is heuristic; no full-body stability proof'))
            finally:env.close()
            write_json(root/'status.json',dict(status='evaluating',embodiment=kind,schema=SCHEMA))
            evaluate(None,kind,directory/'baseline.json',a.episodes,environment_class=NavigationResidualEnv,schema=SCHEMA)
            evaluate(model,kind,directory/'evaluation.json',a.episodes,environment_class=NavigationResidualEnv,schema=SCHEMA)
        write_json(root/'status.json',dict(status='complete',elapsed_s=time.perf_counter()-started,schema=SCHEMA))
    except BaseException as exc:
        write_json(root/'status.json',dict(status='failed',error=repr(exc)))
        raise


if __name__=='__main__':main()
