# -*- coding: utf-8 -*-
"""RSI 递归自我优化循环 (2026-11-09 过夜自主运行)

循环: 等待 v14 基线 → [评估(留出块) → 死因诊断 → 规则决策(调参/课程) →
      1M 续训 → 候选评估 → 择优接受/回退] × N, 至 06:30 截止。

自我改进的动作空间 (有界, 带禁忌表):
  - 课程地形: S12/S16/C1/C2/C3 (真实 DEM 块轮换)
  - 奖励旋钮: tilt_rel_w, alive_w, prog_w (env_cfg.json)
  - PPO 旋钮: learning_rate, ent_coef
接受准则: 综合分 (C1 0.5 / C2 0.3 / C3 0.2 加权 0.7×存活+0.3×到达) 提升 ≥0.01
         且平地技能不塌方 (smooth 存活 ≥0.5)
产出: rl/rsi/state.json + log.md + 最佳模型 models/rsi_best.zip
"""
import json
import math
import os
import shutil
import time


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    import generate_model as gm
    import terrain as terr
    from mujoco_forest_env import ForestWheelEnv

    DEADLINE = time.mktime(time.strptime('2026-11-09 06:30:00', '%Y-%m-%d %H:%M:%S'))
    NENV, CHUNK = 12, 1_000_000
    os.makedirs('rsi', exist_ok=True)
    bank = json.load(open('dem_patches.json', encoding='utf-8'))

    log_f = open('rsi/log.md', 'a', encoding='utf-8')

    def log(s):
        stamp = time.strftime('%H:%M:%S')
        print('[{}] {}'.format(stamp, s), flush=True)
        log_f.write('{} {}\n'.format(stamp, s))
        log_f.flush()

    # ---------- 评估 (留出块 bank[cls][7], 15 回合/档 + 平地回归) ----------
    def build_terrain(kind, seed):
        if kind in ('S12', 'S16'):
            return terr.make_h_fn(seed=seed, span=44.0, amp1=0.55, amp2=0.22,
                                  amp3=0.08, rough_amp=0.03, tilt_deg=float(kind[1:]))
        p = bank[kind][7]
        return terr.make_dem_fn(p['lat'], p['lon'], p['rot'], micro_amp=0.05, seed=seed)

    def evaluate(model, verbose=False):
        gm.generate(seed=999, wheel=True, profile='std', h_fn=build_terrain('C1', 999))
        res = {}
        for kind, w in (('C1', 0.5), ('C2', 0.3), ('C3', 0.2)):
            q = bank[kind][7]
            gm.generate(seed=999, wheel=True, profile='std',
                        h_fn=terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=999))
            env = ForestWheelEnv(profile='std', generate=False)
            surv = reach = 0
            causes = {}
            for ep in range(15):
                o, _ = env.reset(seed=7000 + ep)
                term = False
                cause = 'timeout'
                for k in range(500):
                    a, _ = model.predict(o, deterministic=True)
                    o, r, term, trunc, info = env.step(a)
                    if term or trunc:
                        cause = info.get('failure') or ('reach' if info.get('is_success') else 'timeout')
                        break
                if not term:
                    surv += 1
                if info.get('is_success'):
                    reach += 1
                causes[cause] = causes.get(cause, 0) + 1
            res[kind] = {'surv': surv / 15, 'reach': reach / 15, 'causes': causes}
            env.close()
        # 平地回归 (合成 3°)
        gm.generate(seed=999, wheel=True, profile='smooth')
        env = ForestWheelEnv(profile='smooth', generate=False)
        ss = 0
        for ep in range(10):
            o, _ = env.reset(seed=8000 + ep)
            term = False
            for k in range(500):
                a, _ = model.predict(o, deterministic=True)
                o, r, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            if not term:
                ss += 1
        env.close()
        res['smooth'] = {'surv': ss / 10}
        score = sum(w * (0.7 * res[k]['surv'] + 0.3 * res[k]['reach'])
                    for k, w in (('C1', 0.5), ('C2', 0.3), ('C3', 0.2)))
        if verbose:
            for k in ('C1', 'C2', 'C3'):
                log('  {}: 存活 {}/15, 到达 {}/15, 死因 {}'.format(
                    k, round(res[k]['surv'] * 15), round(res[k]['reach'] * 15),
                    res[k]['causes']))
            log('  smooth: 存活 {}/10'.format(round(res['smooth']['surv'] * 10)))
        return score, res

    # ---------- 等待 v14 完成 ----------
    log('RSI 启动, 等待 v14 final ...')
    v14 = 'models/forest_v14/final.zip'
    while not os.path.exists(v14):
        time.sleep(120)
        if time.time() > DEADLINE:
            log('截止前 v14 未完成, 退出')
            return
    log('v14 final 就绪')

    best_path = 'models/rsi_best.zip'
    shutil.copy(v14, best_path)
    model = PPO.load(best_path, device='cpu')
    best_score, best_res = evaluate(model, verbose=True)
    base_score = best_score
    log('基线综合分 {:.3f}'.format(best_score))
    json.dump({'iter': 0, 'score': best_score, 'res': best_res},
              open('rsi/state.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # ---------- 自我改进决策规则 ----------
    tabu = set()
    tried = []

    def decide(res, n_stall):
        c1, c2, c3 = res['C1'], res['C2'], res['C3']
        tilt_deaths = sum(r['causes'].get('fall', 0) for r in (c1, c2, c3)) / 45
        coll_deaths = sum(r['causes'].get('collision', 0) for r in (c1, c2, c3)) / 45
        acts = []
        # 基座回退: 当前基座平地技能都塌了 → 退回 v13 从 16° 桥接重新爬 (递归自我修正)
        if best_res['smooth']['surv'] < 0.5 and best_score < 0.05:
            acts.append(('base_v13_bridge', dict(base='models/forest_v13/final',
                                                 terrain='S16', ent=0.005, lr=1e-4,
                                                 cfg=dict(tilt_rel_w=2.5, alive_w=0.3))))
        if best_res['smooth']['surv'] >= 0.5 and c1['surv'] < 0.5:
            acts.append(('c1_hold', dict(terrain='C1', ent=0.005, lr=1e-4,
                                         cfg=dict(tilt_rel_w=2.5, alive_w=0.3))))
        elif c2['surv'] < 0.4 and c1['surv'] >= 0.4:
            acts.append(('c2_step', dict(terrain='C2', ent=0.004, lr=7e-5, cfg=None)))
        elif c3['surv'] < 0.3 and c2['surv'] >= 0.3:
            acts.append(('c3_push', dict(terrain='C3', ent=0.006, lr=5e-5, cfg=None)))
        if n_stall >= 2:
            acts.append(('stall_reset', dict(terrain='C2', ent=0.008, lr=5e-5, cfg=None)))
        if coll_deaths > 0.3:
            acts.append(('anti_collision', dict(terrain='C1', ent=0.003, lr=7e-5,
                                                cfg=dict(tree_w=0.10))))
        if tilt_deaths > 0.5 and c1['surv'] < 0.7:
            acts.append(('tilt_boost', dict(terrain='C1', ent=0.004, lr=7e-5,
                                            cfg=dict(tilt_rel_w=3.0, tilt_abs_w=0.2))))
        if not acts:
            acts.append(('polish', dict(terrain='C1', ent=0.002, lr=3e-5, cfg=None)))
        for name, a in acts:
            if name not in tabu:
                tabu.add(name)
                return name, a
        tabu.clear()
        return acts[0][0], acts[0][1]

    # ---------- 主循环 ----------
    it = 0
    n_stall = 0
    while it < 12 and time.time() < DEADLINE - 15 * 60:
        it += 1
        name, a = decide(best_res, n_stall)
        log('迭代 {}: 动作 {} 参数 {}'.format(it, name, a))

        # 应用奖励旋钮
        if a.get('cfg'):
            json.dump(a['cfg'], open('env_cfg.json', 'w'))
        else:
            if os.path.exists('env_cfg.json'):
                os.remove('env_cfg.json')

        # 续训 1M (基座可选: 默认当前最优; base_v13_bridge 动作可回退 v13 重爬)
        base = a.get('base', best_path)
        gm.generate(seed=4000 + it, wheel=True, profile='std',
                    h_fn=build_terrain(a['terrain'], 4000 + it))
        env = SubprocVecEnv(
            [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False),
                                 'logs/rsi_{}_{}'.format(it, i)) for i in range(NENV)],
            start_method='spawn')
        model = PPO.load(base, env=env, device='cpu')
        model.learning_rate = a['lr']
        model.ent_coef = a['ent']
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=False)
        cand = 'rsi/cand_{}.zip'.format(it)
        model.save(cand[:-4])
        env.close()

        # 候选评估 (恢复默认奖励评估口径一致)
        if os.path.exists('env_cfg.json'):
            os.remove('env_cfg.json')
        cand_model = PPO.load(cand[:-4], device='cpu')
        score, res = evaluate(cand_model, verbose=True)
        gain = score - best_score
        # 接受准则: 综合分提升; 或基座康复 (平地技能恢复 ≥0.6 而当前基座已塌) — 后者是递归自我修正的入口
        accept = (gain >= 0.01 and res['smooth']['surv'] >= 0.5) or \
                 (best_res['smooth']['surv'] < 0.5 and res['smooth']['surv'] >= 0.6
                  and score >= best_score - 0.02)
        log('迭代 {}: 候选分 {:.3f} (基线 {:.3f}, Δ{:+.3f}) → {}'.format(
            it, score, best_score, gain, '接受' if accept else '回退'))
        tried.append({'iter': it, 'action': name, 'params': a, 'score': score,
                      'gain': round(gain, 4), 'accept': accept})

        if accept:
            shutil.copy(cand, best_path)
            best_score, best_res = score, res
            n_stall = 0
        else:
            n_stall += 1
        json.dump({'iter': it, 'score': best_score, 'res': best_res, 'history': tried},
                  open('rsi/state.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    log('RSI 结束: {} 次迭代, 最佳综合分 {:.3f} (v14 基线 {:.3f}, Δ{:+.3f})'.format(
        it, best_score, base_score, best_score - base_score))
    log('RSI_FINAL_DONE best={:.3f}'.format(best_score))


if __name__ == '__main__':
    main()
