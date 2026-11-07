"""Bundle only the offline demo and its public-facing local assets."""
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
files=['demo.html','demo.js','demo-data.js','forest-evidence.js','index.html','viewer.js','robot-data.js']
files += [str(p.relative_to(root/'web')).replace('\\','/') for p in (root/'web/demo-assets').glob('*')]
files += [str(p.relative_to(root/'web')).replace('\\','/') for p in (root/'web/forest-autonomy-assets-v6').glob('*') if p.suffix in ('.mp4','.jpg')]
with zipfile.ZipFile(root/'ForestRobo-Demo.zip','w',compression=zipfile.ZIP_DEFLATED) as out:
    for name in files:out.write(root/'web'/name,'ForestRobo-Demo/'+name)
    out.writestr('ForestRobo-Demo/使用说明.txt',
        '解压后双击 demo.html。无需安装依赖或联网。\n含森林、地震、城市更新七种配置和森林闭环验证回放。\n'
        '使用Chrome或Edge并启用WebGL。请保留目录结构。\n'
        '回放使用修正基元，不是已通过验收的学习策略。规划圆点为示意，不是闭环机器人跟踪。\n')
print(root/'ForestRobo-Demo.zip')
