# -*- coding: utf-8 -*-
"""端到端检测链路测试: 真实 env 循环中直线冲向幸存者, 发现事件是否触发"""
import math

import numpy as np

from newton_vecenv import ForestDogNewtonEnv

env = ForestDogNewtonEnv(num_envs=2, seed=11, task='search', device='cpu')

# 世界0: 放在幸存者0 正西 2.5m, 朝向它
s = env.survivors[0]
env._jq[0, :3] = [s[0] - 2.5, s[1], float(env._h(s[0] - 2.5, s[1])) + 0.46]
env._jq[0, 3:7] = [1, 0, 0, 0]
env._jqd[0] = 0.0
env._push_and_fk(env.st_a)
env._pull_state(env.st_a)
env._prev_pos[0] = env._jq[0, :3].copy()
env._prev_yaw[0] = 0.0
env._found[0] = False
env._steps[0] = 0

print(f'幸存者0 @ ({s[0]:.2f},{s[1]:.2f}) | 狗 @ ({env._jq[0,0]:.2f},'
      f'{env._jq[0,1]:.2f}) | 初始距离 {math.hypot(s[0]-env._jq[0,0], s[1]-env._jq[0,1]):.2f} m')

r_sum = 0.0
for k in range(400):                      # 8 秒
    # 动作: 全零 → 纯 CPG 直线前进
    obs, r, done, info = env.step(np.zeros((2, 12), np.float32))
    r_sum += float(r[0])
    d = math.hypot(s[0] - env._jq[0, 0], s[1] - env._jq[0, 1])
    if k % 50 == 0 or env._found[0].any() or done[0]:
        print(f'  t={k*0.02:4.1f}s x={env._jq[0,0]:6.2f} 距离={d:5.2f} '
              f'found={env._found[0].tolist()} r={r[0]:+7.2f} 累计={r_sum:+8.1f} '
              f'done={done[0]}')
    if done[0]:
        break
print('DETECT_TEST_DONE')
