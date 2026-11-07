"""Generate a factual local report from saved experiment evidence only."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    experiments=[]
    rows=[]
    total_steps=0
    for folder,label in [('forest_v3_experiment','直接关节策略'),('navigation_residual','分层修正策略')]:
        base=ROOT/'rl/models'/folder
        status=json.loads((base/'status.json').read_text(encoding='utf-8')) if (base/'status.json').exists() else {'status':'missing'}
        entry=dict(experiment=folder,status=status['status'],models=[])
        for kind in ('foot','wheel'):
            directory=base/kind
            if not (directory/'protocol.json').exists():continue
            protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
            total_steps+=protocol['actual_timesteps']
            model=dict(embodiment=kind,protocol=protocol)
            for file,description in [('baseline','基线'),('evaluation','学习策略')]:
                path=directory/(file+'.json')
                if not path.exists():continue
                result=json.loads(path.read_text(encoding='utf-8'))
                episodes=result['episodes']
                if not episodes:continue
                count=len(episodes);success=sum(e['is_success'] for e in episodes)
                failures=dict(Counter(e['failure'] or ('timeout' if e['truncated'] else 'success') for e in episodes))
                summary=dict(episodes=count,success=success,status=result['status'],failures=failures,
                    mean_progress_m=sum(e['progress_m'] for e in episodes)/count,
                    mean_mechanical_energy_j=sum(e['energy_j'] for e in episodes)/count)
                model[file]=summary
                rows.append(f'|{label}|{"足式" if kind=="foot" else "轮足"}|{description}|{success}/{count}|{summary["mean_progress_m"]:.3f}|{json.dumps(failures,ensure_ascii=False)}|')
            entry['models'].append(model)
        experiments.append(entry)
    report=dict(generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        total_completed_timesteps=total_steps,experiments=experiments,
        limitation='Simulation only. Baselines differ between experiment families; no valid forest deployment claim.')
    (ROOT/'design/experiment_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    text=f'''# 森林机器人训练结果报告

生成时间：{report['generated_at']}。累计已保存策略训练步数：{total_steps:,}。

|实验|底盘|对照|成功回合|平均目标接近距离 m|结束类别|
|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

正接近距离表示靠近目标，负值表示远离。能耗明细、故障分组、Wilson区间、推理延迟与单回合记录见原始JSON。本表只概括这批固定独立地形回合，不能外推所有森林类型。

直接关节策略的基线为零关节残差/零轮速，即站立；分层修正策略的基线是同一局部目标控制器和底层基元、学习修正为零。两种基线不能互换。分层策略即使成功也不能全部归功于PPO。

目前结果尚不足以证明真实林地泛化。根据低层响应诊断，应优先修正轮地接触、转向跟踪和足式底层步态，再开展更大训练。当前模型没有真实状态估计、完整移动机械臂动力学或硬件驱动，不用于实机自主切削。

## 已完成的配套工作

- 三种任务机型、两种臂和可换工具，266个带名称/规格/来源的模块零件；WebGL离线爆炸图、三个GLB和三张预览。
- 感知失效与断联指令门控、局部高程图规划、机械臂IK/静态力矩/支撑检查、样品离线证据日志。
- 修改说明和验证边界见DEVELOPMENT.md；运行说明见README.md及rl/README_FOREST_V3.md。

## 定时汇报状态

用户要求2026-11-07早上汇报，本轮尝试创建北京时间08:00一次性晨报。工具因“需要审批，当前审批策略不允许”拒绝创建，未成功设置自动推送。此文件保存可复查汇报结果，不代表定时任务已创建。
'''
    if any(e['status']!='complete' for e in experiments):
        text+='\n注意：仍有实验未完成，以上是当前已保存证据的阶段报告。\n'
    (ROOT/'TRAINING_REPORT.md').write_text(text,encoding='utf-8')
    print('Wrote TRAINING_REPORT.md; total steps:',total_steps)


if __name__=='__main__':main()
