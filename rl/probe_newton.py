# -*- coding: utf-8 -*-
"""Newton 1.7 API 探测：多世界批处理 / 步进接口 / GPU 可用性"""
import inspect
import newton
import warp as wp

print('=== 关键签名 ===')
print('ModelBuilder.__init__:', inspect.signature(newton.ModelBuilder.__init__))
print('Model.__init__:', inspect.signature(newton.Model.__init__))
print('State.__init__:', inspect.signature(newton.State.__init__))
print('SolverMuJoCo.step:', inspect.signature(newton.solvers.SolverMuJoCo.step))
print('add_mjcf:', inspect.signature(newton.ModelBuilder.add_mjcf))

print()
print('=== Control 属性 ===')
print([x for x in dir(newton.Control) if not x.startswith('_')])
print('=== State 属性 ===')
print([x for x in dir(newton.State) if not x.startswith('_')])

print()
print('=== GPU ===')
dev = wp.get_device()
print('warp device:', dev, '| is_cuda:', dev.is_cuda)

print()
print('=== 落体小实验：加载 MJCF 并步进 ===')
xml = """<mujoco model="probe">
  <compiler angle="radian"/>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 1"/>
    <body name="box" pos="0 0 2">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""
with open('probe.xml', 'w', encoding='utf-8') as f:
    f.write(xml)

builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
builder.add_mjcf('probe.xml')
model = newton.Model(builder)
print('Model ok | joint_count', model.joint_count, '| body_count', model.body_count,
      '| dof_count', model.dof_count)
try:
    print('model has num_worlds attr?', getattr(model, 'num_worlds', 'N/A'))
except Exception as e:
    print('num_worlds err', e)

state = newton.State(model)
control = newton.Control(model)
print('state.body_q shape:', state.body_q.shape, '| joint_q shape:', state.joint_q.shape)
print('control fields:', [x for x in dir(control) if not x.startswith('_') and hasattr(getattr(control, x), 'shape')])

solver = newton.solvers.SolverMuJoCo(model)
for i in range(20):
    solver.step(state, control, 0.005, 1)
q = state.body_q.numpy()
print('20 步后 box 高度(应为 ~0.1):', q[0, 0, 6] if q.ndim == 3 else q[0, 6])
print('PROBE_OK')
