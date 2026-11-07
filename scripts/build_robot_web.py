"""Build file:// compatible data and standalone HTML after model export."""
import json
from pathlib import Path


def build():
    web = Path(__file__).resolve().parents[1] / 'web'
    data = json.loads((web / 'robot.json').read_text(encoding='utf-8'))
    bundle = 'globalThis.FOREST_ROBOT = ' + json.dumps(data, ensure_ascii=False) + ';'
    (web / 'robot-data.js').write_text(bundle, encoding='utf-8')
    html = (web / 'index.html').read_text(encoding='utf-8')
    for name, source in [('robot-data.js', bundle),
                         ('viewer.js', (web / 'viewer.js').read_text(encoding='utf-8'))]:
        html = html.replace(f'<script src="{name}"></script>',
                            '<script>' + source.replace('</script', '<\\/script') + '</script>')
    (web / 'robot-offline.html').write_text(html, encoding='utf-8')
    print(f'Built browser-local data and offline page: {len(data["parts"])} parts')


if __name__ == '__main__':
    build()
