# -*- coding: utf-8 -*-
"""Newton 1.7 探测第二步：finalize / replicate / 世界复制"""
import inspect
import newton

print('finalize:', inspect.signature(newton.ModelBuilder.finalize))
print('replicate:', inspect.signature(newton.ModelBuilder.replicate))
print('begin_world:', inspect.signature(newton.ModelBuilder.begin_world))
print('add_world:', inspect.signature(newton.ModelBuilder.add_world))
print('Control.__init__:', inspect.signature(newton.Control.__init__))
print('Contacts.__init__:', inspect.signature(newton.Contacts.__init__))
mm = [x for x in dir(newton.Model) if not x.startswith('_') and
      ('state' in x or 'control' in x or 'world' in x or 'contact' in x)]
print('Model methods:', mm)
