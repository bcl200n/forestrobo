"""Concept matching, not a procurement recommendation or capability approval."""
import math
from .manipulator import CATALOG, static_check, position_ik

SCENARIOS={
    'void':dict(name='地震 · 通路与缝隙侦察',variant='rescue_scout',tool='life_probe',
                task='停驻采集疑似生命线索，人工复核并标记',
                conditions=['通路与足端落点需现场确认','探头与臂部载荷需重新验算'],status='conditional_concept'),
    'clear':dict(name='地震 · 轻型清障',variant='rescue_clear',tool='rescue_gripper',
                 task='已确认稳定作业点的轻物夹持与递送',
                 conditions=['履带与支撑动力学未验证','夹持和切割均需人工持续使能'],status='conditional_concept'),
    'cut':dict(name='地震 · 受控轻型切割',variant='rescue_clear',tool='rescue_cut',
               task='人员确认对象与结构状态后的受控切割',
               conditions=['不能用于承重构件','切削力、夹紧与热风险需试验'],status='conditional_concept'),
    'stairs':dict(name='建筑 · 楼梯与坍塌通道',variant='rescue_scout',tool='life_probe',
                  task='先评估通行，当前只支持方案讨论',
                  conditions=['楼梯、瓦砾与边缘稳定性未验证'],status='unvalidated_mobility'),
    'hazmat':dict(name='危化 · 易燃易爆区域',variant=None,tool=None,task='当前平台不匹配',
                  conditions=['当前机型、驱动、电池和传感器无防爆验证'],status='unsupported'),
    'flood':dict(name='洪涝 · 涉水与水下搜索',variant=None,tool=None,task='当前平台不匹配',
                 conditions=['无涉水深度、密封和水下定位验证'],status='unsupported'),
}


def selection(scenario,object_mass_kg=0.):
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    if not math.isfinite(object_mass_kg) or object_mass_kg<0:
        raise ValueError('invalid object mass')
    result=dict(SCENARIOS[scenario])
    result.update(scenario=scenario,hardware_approved=False)
    if result['variant']:
        variant=CATALOG['variants'][result['variant']]
        arm=CATALOG['arms'][variant['arm']]
        tool=CATALOG['tools'][result['tool']]
        total=object_mass_kg+tool['mass_kg']
        result.update(variant_spec=variant,arm_id=variant['arm'],arm=arm,tool_spec=tool,
                      combined_payload_kg=total,payload_margin_kg=arm['payload_kg']-total,
                      within_nominal_payload=total<=arm['payload_kg'])
    return result


def arm_preview(scenario,target_m,object_mass_kg=0.,reaction_n=(0.,0.,0.)):
    match=selection(scenario,object_mass_kg)
    if not match['variant']:
        raise ValueError('scenario has no supported concept arm')
    ik=position_ik(target_m,arm=match['arm_id'])
    check=static_check(ik['joints_rad'],force_n=reaction_n,
                       tool_mass_kg=match['combined_payload_kg'],arm=match['arm_id'])
    return dict(selection=match,ik=ik,static=check,
                execution_approved=False,orientation_checked=False,collision_checked=False)
