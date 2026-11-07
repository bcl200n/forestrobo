# -*- coding: utf-8 -*-
"""林下机器狗强化学习环境 —— NVIDIA Isaac Newton 批量向量化实现

- 一个 Model 复制 N 个世界，SolverMuJoCo(MuJoCo-Warp) 在 GPU 上并行步进
- 任务：机器狗在林下（树干 + 岩石 + 倒木）向随机目标点导航
- 观测含"虚拟激光雷达"：前方三扇区最近树干距离
- 兼容 Stable-Baselines3 的 VecEnv 接口

状态数组布局（跨世界扁平）:
  joint_q   (19*N,) 每世界 7(free) + 12(铰接)
  joint_qd  (18*N,) 每世界 6(free) + 12(铰接)
  body_q    (13*N,) 每世界 13 个刚体位姿 (vec7)
"""
import json
import math
import os

import numpy as np

import newton
from stable_baselines3.common.vec_env import VecEnv

import gait
import terrain as terr

OBS_DIM = 49
ACT_DIM = 12
CTRL_DT = 0.02                 # 50 Hz 控制频率
EP_LEN = 500                   # 10 s / 回合
ACTION_SCALE = 0.25            # 动作 → 关节目标偏移(弧度)
GOAL_R = 1.2                   # 到达判定半径（真实抵达）
LASER_RANGE = 6.0              # 虚拟激光雷达量程


def _quat_to_mat(q):
    """wxyz 四元数 → 旋转矩阵 (N,3,3)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - w * z)
    m[:, 0, 2] = 2 * (x * z + w * y)
    m[:, 1, 0] = 2 * (x * y + w * z)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - w * x)
    m[:, 2, 0] = 2 * (x * z - w * y)
    m[:, 2, 1] = 2 * (y * z + w * x)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


class ForestDogNewtonEnv(VecEnv):
    """Newton GPU 批量环境"""

    def __init__(self, num_envs=64, xml_path=None,
                 layout_path='forest_layout.json', device=None, seed=7,
                 wheel=False, task='navigate', goal_range=(10.0, 18.0)):
        self.rng = np.random.default_rng(seed)
        base = os.path.dirname(os.path.abspath(__file__))
        self.wheel = wheel
        self.task = task
        if task == 'search':                  # 废墟搜救: 固定足式 + 废墟场景
            wheel = False
            xml_path = 'rubble_dog.xml'
            layout_path = 'rubble_layout.json'
        if xml_path is None:
            xml_path = 'forest_dog_wheel.xml' if wheel else 'forest_dog.xml'
        xml_path = xml_path if os.path.isabs(xml_path) else os.path.join(base, xml_path)
        layout_path = layout_path if os.path.isabs(layout_path) else os.path.join(base, layout_path)

        # 设备：优先 GPU
        if device is None:
            try:
                import warp as wp
                wp.get_device('cuda:0')
                device = 'cuda:0'
            except Exception:
                device = 'cpu'
        self.device = device

        with open(layout_path, encoding='utf-8') as f:
            lay = json.load(f)
        # 地形高度函数（与 MuJoCo hfield 表面严格一致, 含平移量）
        meta_name = 'terrain_rubble_meta.json' if task == 'search' else 'terrain_meta.json'
        meta_path = os.path.join(base, meta_name)
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                self._h = terr.build_from_params(json.load(f))
        else:
            self._h = lambda x, y: np.zeros_like(np.asarray(x, dtype=float))
        if task == 'search':
            self.survivors = np.array(lay['survivors'], np.float32)   # (K,3)
            self.trees = np.array(lay['hazards'], np.float32)         # 障碍(激光)
            self._found = np.zeros((num_envs, len(self.survivors)), bool)
            self._prev_tdist = np.full(num_envs, np.nan, np.float32)
            self._prev_tgt = np.full(num_envs, -1, np.int64)
        else:
            self.trees = np.array(lay['trees'], dtype=np.float32)     # (T,3) x,y,r
        if wheel:
            import gait as _g
            self.q0 = _g.WQ0.astype(np.float32)                   # 轮足站姿 (12,)
            self.stand_z = float(_g.WSTAND_Z)
            self.wheel_r = float(_g.WHEEL_R)
            self.act_dim, self.obs_dim = 16, 57
            self.v_max = 40.0                                      # 轮速上限 rad/s
        else:
            leg0 = lay['leg_default']                              # [abd, hip, knee]
            self.q0 = np.tile(np.array(leg0, dtype=np.float32), 4)
            self.stand_z = 0.45
            self.act_dim = 12
            self.obs_dim = 53 if task == 'search' else 49
            self.v_max = 0.0
        self.ep_len = 900 if task == 'search' else EP_LEN
        self.use_cpg = (task == 'search')          # 搜救: CPG 残差动作
        self._t = 0.0                              # 步态相位时钟

        # --- Newton：单世界 → 复制 N 世界 → GPU ---
        single = newton.ModelBuilder(up_axis=newton.Axis.Z)
        single.add_mjcf(xml_path)
        batch = newton.ModelBuilder(up_axis=newton.Axis.Z)
        batch.replicate(single, num_envs)
        self.model = batch.finalize(device)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, nconmax=num_envs * 160, njmax=num_envs * 256)
        self.contacts = newton.CollisionPipeline(self.model).contacts()
        self.st_a = self.model.state()
        self.st_b = self.model.state()
        self.ctrl = self.model.control()

        self.n = num_envs
        self.goal_range = goal_range
        self.nq_w = 7 + self.act_dim                    # 轮足含 4 轮角坐标
        self.nqd_w = 6 + self.act_dim
        from gymnasium import spaces
        observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        super().__init__(num_envs, observation_space, action_space)

        # 宿主侧状态镜像（GPU 数组 .numpy() 为拷贝，写回需 assign）
        self._jq = np.zeros((self.n, self.nq_w), np.float32)
        self._jqd = np.zeros((self.n, self.nqd_w), np.float32)
        self._prev_pos = np.zeros((self.n, 3), np.float32)
        self._prev_yaw = np.zeros(self.n, np.float32)
        self._last_act = np.zeros((self.n, self.act_dim), np.float32)
        self._goal = np.zeros((self.n, 2), np.float32)
        self._steps = np.zeros(self.n, np.int64)
        self._ep_ret = np.zeros(self.n, np.float32)
        self._ep_len = np.zeros(self.n, np.int64)
        self._actions = None
        self._need_reset = np.ones(self.n, bool)

    # ---------------- 基础工具 ----------------
    def _pull_state(self, st):
        self._jq = st.joint_q.numpy().reshape(self.n, self.nq_w).copy()
        self._jqd = st.joint_qd.numpy().reshape(self.n, self.nqd_w).copy()
        bq_all = st.body_q.numpy()
        nb = bq_all.shape[0] // self.n            # 每世界刚体数(足式13/轮足17)
        self._bq = bq_all.reshape(self.n, nb, 7)[:, 0, :].copy()
        self._bqd = st.body_qd.numpy().reshape(self.n, nb, 6)[:, 0, :].copy()

    def _push_and_fk(self, st):
        st.joint_q.assign(self._jq.reshape(-1))
        st.joint_qd.assign(self._jqd.reshape(-1))
        newton.eval_fk(self.model, st.joint_q, st.joint_qd, st)

    def _reset_worlds(self, idx):
        """随机朝向/位置重置指定世界（出生区无障碍）"""
        k = len(idx)
        if k == 0:
            return
        hd = self.rng.uniform(-math.pi, math.pi, k).astype(np.float32)
        x = self.rng.uniform(0.4, 1.6, k).astype(np.float32)
        y = self.rng.uniform(-0.9, 0.9, k).astype(np.float32)
        z = np.asarray(self._h(x, y), dtype=np.float32).reshape(-1) + 0.45
        ch, sh = np.cos(hd / 2), np.sin(hd / 2)
        self._jq[idx, :7] = np.stack([x, y, np.full(k, z), ch,
                                      np.zeros(k), np.zeros(k), sh], axis=1)
        self._jq[idx, 7:] = np.concatenate(
            [np.tile(self.q0, (k, 1)) + self.rng.normal(0, 0.03, (k, 12)),
             np.zeros((k, 4), np.float32) if self.wheel else np.zeros((k, 0), np.float32)],
            axis=1)
        self._jqd[idx, :] = 0.0
        # 目标：随机方向 goal_range m
        ga = self.rng.uniform(-math.pi, math.pi, k).astype(np.float32)
        gd = self.rng.uniform(*self.goal_range, k).astype(np.float32)
        self._goal[idx, 0] = x + np.cos(ga) * gd
        self._goal[idx, 1] = y + np.sin(ga) * gd
        self._last_act[idx] = 0.0
        if self.task == 'search':
            self._found[idx] = False
            self._prev_tdist[idx] = np.nan
            self._prev_tgt[idx] = -1
        self._steps[idx] = 0
        self._ep_ret[idx] = 0.0
        self._ep_len[idx] = 0
        pos = self._jq[idx, :3].copy()
        self._prev_pos[idx] = pos
        self._prev_yaw[idx] = hd

    # ---------------- 观测 / 奖励 ----------------
    def _compute_obs(self):
        pos = self._jq[:, 0:3]
        quat = self._jq[:, 3:7]
        R = _quat_to_mat(quat)                       # 本体→世界
        Rt = np.transpose(R, (0, 2, 1))              # 世界→本体
        vel = (pos - self._prev_pos) / CTRL_DT       # 世界系线速度（数值差分）
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        yaw_rate = ((yaw - self._prev_yaw + math.pi) % (2 * math.pi) - math.pi) / CTRL_DT
        v_body = np.einsum('nij,nj->ni', Rt, vel)
        grav_body = np.einsum('nij,j->ni', Rt, np.array([0, 0, -1.0], np.float32))
        # 目标方向（本体系）
        if self.task == 'search':
            # 最近未发现幸存者方向（本体系）与距离
            rel = self.survivors[None, :, :2] - pos[:, None, :2]      # (n,K,2)
            dist = np.linalg.norm(rel, axis=2)
            dist = np.where(self._found, 1e9, dist)
            ki = dist.argmin(axis=1)
            idxr = np.arange(self.n)
            tdist = dist[idxr, ki]
            tv = rel[idxr, ki] / np.maximum(tdist[:, None], 1e-6)
            gz = np.zeros((self.n, 3), np.float32)
            gz[:, :2] = tv
            goal_body = np.einsum('nij,nj->ni', Rt, gz)[:, :2]
            self._tgt_world = self.survivors[ki, :2].copy()
            self._tgt_dist = tdist
            self._tgt_idx = ki
        else:
            gv = self._goal - pos[:, :2]
            gd = np.linalg.norm(gv, axis=1, keepdims=True)
            gun = gv / np.maximum(gd, 1e-6)
            gz = np.zeros((self.n, 3), np.float32)
            gz[:, :2] = gun
            goal_body = np.einsum('nij,nj->ni', Rt, gz)[:, :2]
        # 虚拟激光雷达：三扇区（前 ±30° / 左 / 右）最近树干净距
        laser = np.full((self.n, 3), LASER_RANGE, np.float32)
        tx, ty, tr = self.trees[:, 0], self.trees[:, 1], self.trees[:, 2]
        for i in range(self.n):
            rx, ry = tx - pos[i, 0], ty - pos[i, 1]
            d = np.hypot(rx, ry) - tr
            near = d < LASER_RANGE
            if not near.any():
                continue
            ang = np.arctan2(ry[near], rx[near])
            rel = (ang - yaw[i] + math.pi) % (2 * math.pi) - math.pi
            dn = d[near]
            f = np.abs(rel) < math.radians(30)
            l = (rel >= math.radians(30)) & (rel < math.radians(120))
            r_ = (rel <= -math.radians(30)) & (rel > -math.radians(120))
            if f.any():
                laser[i, 0] = max(0.0, dn[f].min())
            if l.any():
                laser[i, 1] = max(0.0, dn[l].min())
            if r_.any():
                laser[i, 2] = max(0.0, dn[r_].min())
        # 相对地形高度（气压计/IMU 语义, 不依赖 GNSS）
        hpos = np.asarray(self._h(pos[:, 0], pos[:, 1]),
                          dtype=np.float32).reshape(-1)
        z_rel = (pos[:, 2] - hpos).astype(np.float32)
        if self.wheel:
            leg_pos = self._jq[:, 7:19] - self.q0
            leg_vel = self._jqd[:, 6:18]
            wheel_w = self._jqd[:, 18:22]                     # rad/s
            # 打滑量: 轮缘线速度 - 机体系前向速度
            self._slip = np.abs(wheel_w.mean(axis=1) * self.wheel_r
                                - v_body[:, 0]).astype(np.float32)
            extra = (wheel_w * 0.05).astype(np.float32)
        else:
            leg_pos = self._jq[:, 7:] - self.q0
            leg_vel = self._jqd[:, 6:]
            self._slip = np.zeros(self.n, np.float32)
            extra = np.zeros((self.n, 0), np.float32)
        if self.task == 'search':
            ph = self._t * gait.FREQ * 2 * math.pi
            extra2 = np.stack([self._found.sum(axis=1) / len(self.survivors),
                               np.clip(self._tgt_dist, 0.0, 25.0) / 25.0,
                               np.sin(ph) * np.ones(self.n, np.float32),
                               np.cos(ph) * np.ones(self.n, np.float32)],
                              axis=1).astype(np.float32)
        else:
            extra2 = np.zeros((self.n, 0), np.float32)
        obs = np.concatenate([
            v_body, grav_body, yaw_rate[:, None], goal_body,
            laser, z_rel[:, None],
            leg_pos, leg_vel, self._last_act, extra, extra2,
        ], axis=1).astype(np.float32)
        self._pos, self._yaw_now, self._v_world, self._zrel = pos, yaw, vel, z_rel
        return obs

    def _compute_rew_done(self, obs, actions):
        pos = self._pos
        yaw = self._yaw_now
        vel = self._v_world
        if self.task == 'search':
            gv = self._tgt_world - pos[:, :2]
        else:
            gv = self._goal - pos[:, :2]
        gd = np.linalg.norm(gv, axis=1)
        gun = gv / np.maximum(gd[:, None], 1e-6)
        v_along = np.sum(vel[:, :2] * gun, axis=1)
        v_side = np.linalg.norm(vel[:, :2] - v_along[:, None] * gun, axis=1)
        grav_xy = np.linalg.norm(obs[:, 3:5], axis=1)
        yaw_rate = obs[:, 6]
        laser = obs[:, 9:12]
        z = self._zrel              # 相对地形高度

        r = (3.5 * v_along - 0.3 * v_side - 0.08 * np.abs(yaw_rate)
             + (0.6 if self.task == 'search' else 1.5) - 1.2 * grav_xy
             - 0.01 * np.sum(actions ** 2, axis=1)
             - 0.03 * np.sum(np.abs(actions - self._last_act), axis=1)
             - 0.5 * np.maximum(0.0, 0.32 - z))
        if self.wheel:
            # v2: 大幅减轻打滑罚(0.3→0.05)与树近罚(2.0→0.8), 防止"冻结躺平"坍缩
            r = r - 0.05 * self._slip \
                - 0.8 * np.sum(np.maximum(0.0, 0.45 - laser), axis=1)
        elif self.task == 'search':
            r = r - 0.8 * np.sum(np.maximum(0.0, 0.45 - laser), axis=1)
        else:
            r = r - 2.0 * np.sum(np.maximum(0.0, 0.55 - laser), axis=1)

        done = np.zeros(self.n, bool)
        done |= grav_xy > (0.88 if self.wheel else 0.80)   # 倾覆(轮足放宽)
        done |= (z < 0.16) | (z > 0.70)  # 塌陷 / 弹飞
        if self.task != 'search':
            done |= laser.min(axis=1) < 0.12  # 撞障碍(搜救不终止, 仅惩罚)
        if self.task == 'search':
            # 势能塑形: 距离差奖励(密集且抗速度噪声), 目标切换步不塑形
            for i in range(self.n):
                if (not np.isnan(self._prev_tdist[i])
                        and self._prev_tgt[i] == self._tgt_idx[i]):
                    r[i] += float(np.clip(2.5 * (self._prev_tdist[i]
                                                - self._tgt_dist[i]), -2.0, 2.0))
                self._prev_tdist[i] = float(self._tgt_dist[i])
                self._prev_tgt[i] = int(self._tgt_idx[i])
            # 发现判定: 接近最近未发现幸存者 <1.0 m
            near = self._tgt_dist < 1.0
            tgt_found = np.take_along_axis(self._found,
                                           self._tgt_idx[:, None], 1)[:, 0]
            newly = near & (~tgt_found)
            r[newly] += 60.0
            for i in np.where(newly)[0]:
                self._found[i, self._tgt_idx[i]] = True
            allf = self._found.all(axis=1)
            r[allf] += 20.0
            done |= allf
            success = allf
        else:
            success = gd < GOAL_R
            r[success] += 10.0
            done |= success
        done |= self._steps >= self.ep_len
        return r.astype(np.float32), done, success

    # ---------------- VecEnv 接口 ----------------
    def reset(self):
        self._reset_worlds(np.arange(self.n))
        self._push_and_fk(self.st_a)
        self.st_b.assign(self.st_a)     # 备份状态到 ping-pong 缓冲
        self._pull_state(self.st_a)
        obs = self._compute_obs()
        self._prev_pos = self._pos.copy()
        self._prev_yaw = self._yaw_now.copy()
        self._need_reset[:] = False
        return obs

    def step_async(self, actions):
        self._actions = actions.copy().reshape(self.n, self.act_dim).astype(np.float32)

    def step_wait(self):
        acts = np.clip(self._actions, -1.0, 1.0)
        tq = self.ctrl.joint_target_q.numpy().reshape(self.n, self.nq_w)
        tq[:] = 0.0
        if self.use_cpg:
            # CPG + 结构化动作: a0=转向(外摆差动), a1=速度(髋共模), a2..=微残差
            base = np.tile(gait.cpg(self._t).astype(np.float32), (self.n, 1))
            steer = (acts[:, 0:1] * 0.22).astype(np.float32)
            speed = (acts[:, 1:2] * 0.18).astype(np.float32)
            tq[:, 7:19] = base
            tq[:, 9:19] += acts[:, 2:] * 0.08   # 残差挂髋/膝 + 两个abd
            tq[:, 7:19:3] += steer          # 四腿外摆差动 → 偏航
            tq[:, 8:19:3] += speed          # 髋共模 → 前倾/速度
            self._t += CTRL_DT
        elif self.wheel:
            tq[:, 7:19] = self.q0 + acts[:, :12] * 0.35
            tqd = self.ctrl.joint_target_qd.numpy().reshape(self.n, self.nqd_w)
            tqd[:] = 0.0
            tqd[:, 18:22] = acts[:, 12:16] * self.v_max
            self.ctrl.joint_target_qd.assign(tqd.reshape(-1))
        else:
            tq[:, 7:] = self.q0 + acts * ACTION_SCALE
        self.ctrl.joint_target_q.assign(tq.reshape(-1))

        self.solver.step(self.st_a, self.st_b, self.ctrl, self.contacts, CTRL_DT)
        self.st_a, self.st_b = self.st_b, self.st_a
        self._pull_state(self.st_a)
        self._steps += 1
        self._ep_len += 1

        obs = self._compute_obs()
        rew, done, success = self._compute_rew_done(obs, acts)
        rew = rew.copy()
        self._ep_ret += rew

        infos = [{} for _ in range(self.n)]
        done_idx = np.where(done)[0]
        for i in done_idx:
            infos[i] = {'episode': {'r': float(self._ep_ret[i]),
                                    'l': int(self._ep_len[i])},
                        'success': bool(success[i]),
                        'terminal_observation': obs[i].copy()}
        # 自动重置
        self._reset_worlds(done_idx)
        if len(done_idx):
            self._push_and_fk(self.st_a)
            self._pull_state(self.st_a)
            fresh = self._compute_obs()
            obs[done_idx] = fresh[done_idx]
            self._prev_pos[done_idx] = self._pos[done_idx]
            self._prev_yaw[done_idx] = self._yaw_now[done_idx]
        else:
            self._prev_pos = self._pos.copy()
            self._prev_yaw = self._yaw_now.copy()
        self._last_act = acts
        return obs, rew, done, infos

    def close(self):
        pass

    def env_is_wrapped(self, indices, wrapper_class):
        return [False] * self.n

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.n

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self.n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)
