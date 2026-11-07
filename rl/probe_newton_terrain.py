# -*- coding: utf-8 -*-
"""Newton 地形探测：出生后逐秒打印各世界高度, 检查 hfield 是否被 GPU 端正确解析"""
import numpy as np

from newton_vecenv import ForestDogNewtonEnv

env = ForestDogNewtonEnv(num_envs=4, seed=7)
print('  heightfield_count =', env.model.heightfield_count)
try:
    hd = env.model.heightfield_data.numpy()
    print(f'  heightfield_data: size={hd.size} min={hd.min():.4f} max={hd.max():.4f}')
except Exception as e:
    print('  heightfield_data 读取失败:', e)

obs = env.reset()
print('reset 后 obs[z_rel] =', np.round(obs[:, 12], 3))
print('reset 后 obs[激光] =', np.round(obs[:, 9:12], 2))

for t in range(1, 4):
    env.step_async(np.zeros((4, 12), np.float32))
    obs, rew, done, info = env.step_wait()
    for i in np.where(done)[0]:
        term = info[i].get('terminal_observation')
        if term is None:
            continue
        grav_xy = float(np.linalg.norm(term[3:5]))
        print(f't={t}s 世界{i} 终止前: grav_xy={grav_xy:.3f} z_rel={term[12]:.3f} '
              f'laser={np.round(term[9:12], 2)} → '
              f'{"倾覆" if grav_xy > 0.8 else ""}{" 低/高" if not 0.16 < term[12] < 0.7 else ""}'
              f'{" 撞树" if term[9:12].min() < 0.12 else ""}{" 步数" if env._steps[i] >= 500 else ""}')
print('NEWTON_TERRAIN_PROBE_DONE')
