# -*- coding: utf-8 -*-
"""forest v14: 真实 DEM 地形通过性训练 (雷波巡逻路线坡度谱) — 奖励 v2 版

相对旧版的两处修正 (死因诊断驱动):
  1. 环境倾斜惩罚改为 "机体轴 vs 地形法线" 偏离 (贴坡为正确姿态), 旧版强迫水平在陡坡被掀翻
  2. 摔倒阈值 55°→60°, 给恢复学习留样本

课程 (每块 1M 步, v13 热启动):
  桥接 2M: 合成缓坡 12°/16° (从 v13 的 6° 平滑过渡到真实坡度)
  C1 2M: 20° × 2 | C2 3M: 28° × 3 | C3 3M: 36° × 3 | 回访 2M: C2/C3 各 1
"""
import json
import os
import time


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from mujoco_forest_env import ForestWheelEnv
    from train_forest_dog import EpLogCallback
    import generate_model as gm
    import terrain as terr

    bank = json.load(open('dem_patches.json', encoding='utf-8'))
    schedule = [
        ('S12', None), ('S16', None),                      # 桥接: 合成坡
        ('C1', 0), ('C1', 1),                              # 20°
        ('C2', 0), ('C2', 1), ('C2', 2),                   # 28°
        ('C3', 0), ('C3', 1), ('C3', 2),                   # 36°
        ('C2', 3), ('C3', 3),                              # 回访防遗忘
    ]
    NENV = 12
    CHUNK = 1_000_000

    os.makedirs('models/forest_v14', exist_ok=True)
    model = PPO.load('models/forest_v13/final', device='cpu')
    t0 = time.time()

    for ci, (cls, pi) in enumerate(schedule):
        if cls.startswith('S'):
            slope = float(cls[1:])
            h_fn = terr.make_h_fn(seed=300 + ci, span=44.0, amp1=0.55, amp2=0.22,
                                  amp3=0.08, rough_amp=0.03, tilt_deg=slope)
            label = '{}°'.format(slope)
        else:
            p = bank[cls][pi]
            h_fn = terr.make_dem_fn(p['lat'], p['lon'], p['rot'], seed=100 + ci)
            label = '{} {:.0f}°'.format(cls, p['slope'])
        gm.generate(seed=200 + ci, wheel=True, profile='std', h_fn=h_fn)
        env = SubprocVecEnv(
            [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False),
                                 'logs/forest_v14_{}_{}'.format(ci, i)) for i in range(NENV)],
            start_method='spawn')
        model.set_env(env)
        model.learn(total_timesteps=CHUNK, callback=EpLogCallback(print_every=120.0),
                    reset_num_timesteps=False)
        model.save('models/forest_v14/ckpt_{:02d}_{}'.format(ci, cls))
        env.close()
        print('chunk {:02d} [{}] DONE 累计 {:.1f}h'.format(
            ci, label, (time.time() - t0) / 3600), flush=True)

    model.save('models/forest_v14/final')
    print('forest_v14_DONE 总用时 %.1f 小时' % ((time.time() - t0) / 3600), flush=True)


if __name__ == '__main__':
    main()
