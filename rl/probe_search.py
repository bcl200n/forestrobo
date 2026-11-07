# -*- coding: utf-8 -*-
"""搜救观测/奖励正确性单元测试 (CPU, 2 世界)"""
import numpy as np

from newton_vecenv import ForestDogNewtonEnv

env = ForestDogNewtonEnv(num_envs=2, seed=11, task='search', device='cpu')
obs = env.reset()

# 1) 目标方向观测核对: 世界0 手动摆到已知位姿
surv0 = env.survivors[0]                      # 第一个幸存者
print('幸存者坐标:', np.round(surv0, 2))
# 把狗放在幸存者正西 5m, 朝向 +x (应看到目标在前方, tdir≈(+1,0))
env._jq[0, :3] = [surv0[0] - 5.0, surv0[1],
                  float(env._h(surv0[0] - 5.0, surv0[1])) + 0.46]
env._jq[0, 3:7] = [1, 0, 0, 0]               # 朝 +x
env._push_and_fk(env.st_a)
env._pull_state(env.st_a)
env._prev_pos[0] = env._jq[0, :3].copy()
env._prev_yaw[0] = 0.0                       # 单位四元数 → yaw=0
obs = env._compute_obs()
td = obs[0, 7:9]
print('观测目标方向(应≈+1,0):', np.round(td, 3))
print('观测目标距离(应≈5):', round(float(obs[0, 50]) * 25, 2))


# 2) 奖励方向性: 先记旧位置再移动, 保留速度差分
def rew_of(dx):
    env._prev_pos[0] = env._jq[0, :3].copy()
    env._jq[0, 0] += dx
    o = env._compute_obs()
    r, done, succ = env._compute_rew_done(o, np.zeros((2, 12), np.float32))
    env._jq[0, 0] -= dx                      # 还原
    return float(r[0])


r_toward = rew_of(+0.06)      # 朝幸存者 0.06m/步 → 3 m/s
r_away = rew_of(-0.06)
print(f'奖励方向性: 朝目标 {r_toward:+.2f} vs 背离 {r_away:+.2f} '
      f'(应 朝>背离, 差≈{3.5 * 3 * 2:.1f})')
print('PROBE_DONE')
