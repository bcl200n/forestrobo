"""Write reviewable static examples; never sends hardware commands."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robotics.manipulator import static_check, support_check
from robotics.navigation import CommandGate, Health
from dataclasses import asdict, replace

root=Path(__file__).resolve().parents[1]
health=Health(.02,.02,.03,.8,0.)
gate=CommandGate()
report={
    'arm_no_load':static_check(np.zeros(5)),
    'arm_500n_vertical_reaction':static_check(np.zeros(5),force_n=(0,0,500),tool_mass_kg=2.3),
    'unanchored_30kg_base_500n':support_check(30,[0,0],.35,.25,[.6,0,.2],[0,0,500]),
    'navigation_onboard_without_link':asdict(gate.filter(.4,0,replace(health,communication_age_s=10))),
    'navigation_stale_sensor':asdict(gate.filter(.4,0,replace(health,range_age_s=.5))),
    'limitation':'Illustrative configurations, not measured hardware performance or deployable control.'
}
(root/'design/planning_checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
