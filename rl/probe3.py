# -*- coding: utf-8 -*-
"""Newton 探测第三步：真实机器狗模型 × 4 世界，摸清状态/控制数组形状"""
import inspect
import numpy as np
import newton

print('eval_fk:', inspect.signature(newton.eval_fk))

single = newton.ModelBuilder(up_axis=newton.Axis.Z)
single.add_mjcf('forest_dog.xml')

batch = newton.ModelBuilder(up_axis=newton.Axis.Z)
batch.replicate(single, 4)
model = batch.finalize('cuda:0')

print('world_count:', getattr(model, 'world_count', 'N/A'),
      '| joint_count:', model.joint_count,
      '| joint_coord_count:', model.joint_coord_count,
      '| dof_count:', model.joint_dof_count,
      '| body_count:', model.body_count)

names = [x for x in dir(model) if 'name' in x.lower()]
print('name attrs:', names)
for attr in ('body_name', 'joint_name'):
    if hasattr(model, attr):
        v = getattr(model, attr)
        print(attr, '->', list(v)[:6], '...')

st = model.state()
for a in ('body_q', 'body_qd', 'joint_q', 'joint_qd'):
    arr = getattr(st, a, None)
    print('state.' + a, None if arr is None else tuple(arr.shape))

ctrl = model.control()
for a in ('joint_target_q', 'joint_target_qd'):
    arr = getattr(ctrl, a, None)
    print('control.' + a, None if arr is None else tuple(arr.shape))

for a in ('joint_q_start', 'joint_dof_start', 'joint_type', 'joint_child'):
    print('model.' + a, getattr(model, a, 'N/A'))

contacts = newton.CollisionPipeline(model).contacts()
solver = newton.solvers.SolverMuJoCo(model)
print('--- solver 创建后重新查看 control ---')
ctrl = model.control()
for a in ('joint_target_q', 'joint_target_qd'):
    arr = getattr(ctrl, a, None)
    print('control.' + a, None if arr is None else tuple(arr.shape))
st2 = model.state()

# 命令站立姿态并步进 1 秒看是否站住
LEG = np.array([0.0, 0.9, -1.8])
tq = ctrl.joint_target_q.numpy().reshape(4, 19)
tq[:] = 0.0
tq[:, 7:] = np.tile(LEG, 4)
ctrl.joint_target_q.assign(tq.reshape(-1))          # host → device

jq = st.joint_q.numpy().reshape(4, 19)
jq[:] = 0.0
jq[:, 2] = 0.42                                     # 初始高度
jq[:, 7:] = np.tile(LEG, 4)
st.joint_q.assign(jq.reshape(-1))
st.joint_qd.assign(np.zeros(72, dtype=np.float32))
newton.eval_fk(model, st.joint_q, st.joint_qd, st)  # 同步 body_q

for i in range(50):
    solver.step(st, st2, ctrl, contacts, 0.02)
    st, st2 = st2, st

jq2 = st.joint_q.numpy().reshape(4, 19)
print('步进 1s 后 高度 z =', round(float(jq2[0, 2]), 3), '(站立应 ~0.35-0.45)')
print('  世界0 关节角:', np.round(jq2[0, 7:], 2))
print('PROBE3_OK')
