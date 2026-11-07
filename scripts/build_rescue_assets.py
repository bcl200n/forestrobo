"""Generate offline selection data from the same catalog used by the API."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from robotics.rescue_selection import SCENARIOS
from robotics.manipulator import CATALOG

if __name__=='__main__':
    data=dict(scenarios=SCENARIOS,catalog=CATALOG)
    (ROOT/'web/rescue-catalog.js').write_text('globalThis.RESCUE_CATALOG = '+
        json.dumps(data,ensure_ascii=False)+';\n',encoding='utf-8')
    print('Generated web/rescue-catalog.js')
