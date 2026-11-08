# -*- coding: utf-8 -*-
"""下载雷波林场区域真实 DEM (Mapzen/AWS terrarium 瓦片, ~30m/px @z12) 并拼接为 numpy 栅格。

范围: 103.30-103.72E, 28.28-28.66N (覆盖 211/212/213/214 全部林班)
输出: data/leibo/dem/dem_z12.npy + dem_meta.json
"""
import io
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

Z = 12
LON0, LON1 = 103.30, 103.72
LAT0, LAT1 = 28.28, 28.66  # 南, 北

OUT_DIR = r'E:\forestrobo\data\leibo\dem'
os.makedirs(OUT_DIR, exist_ok=True)


def tile_x(lon, z):
    return (lon + 180.0) / 360.0 * (2 ** z)


def tile_y(lat, z):
    rad = math.radians(lat)
    return (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * (2 ** z)


x0, x1 = int(tile_x(LON0, Z)), int(tile_x(LON1, Z))
y0, y1 = int(tile_y(LAT1, Z)), int(tile_y(LAT0, Z))  # y 向南增
print('瓦片范围: x {}-{}, y {}-{} ({} 张)'.format(x0, x1, y0, y1, (x1 - x0 + 1) * (y1 - y0 + 1)))

cols = (x1 - x0 + 1) * 256
rows = (y1 - y0 + 1) * 256
dem = np.zeros((rows, cols), dtype=np.float32)
got = 0
for tx in range(x0, x1 + 1):
    for ty in range(y0, y1 + 1):
        url = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{}/{}/{}.png'.format(Z, tx, ty)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    img = np.asarray(Image.open(io.BytesIO(r.read())).convert('RGB'), dtype=np.float32)
                elev = img[:, :, 0] * 256.0 + img[:, :, 1] + img[:, :, 2] / 256.0 - 32768.0
                r0 = (ty - y0) * 256
                c0 = (tx - x0) * 256
                dem[r0:r0 + 256, c0:c0 + 256] = elev
                got += 1
                break
            except Exception as e:
                if attempt == 2:
                    print('失败 {}: {}'.format(url, e))
                time.sleep(1.5)

print('下载 {}/{} 张'.format(got, (x1 - x0 + 1) * (y1 - y0 + 1)))

# 栅格地理参考: 左上角像素中心的经纬度
n = 2 ** Z
res_lon = 360.0 / n / 256.0
top_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y0 + 0.0) / n))))
meta = {
    'z': Z, 'x0': x0, 'y0': y0, 'rows': rows, 'cols': cols,
    'res_deg': res_lon, 'top_lat': top_lat,
    'source': 'Mapzen terrarium (SRTM/ETOPO blend) via AWS Open Data',
}
np.save(os.path.join(OUT_DIR, 'dem_z12.npy'), dem)
with open(os.path.join(OUT_DIR, 'dem_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

print('高程统计: min {:.0f} m, max {:.0f} m, mean {:.0f} m'.format(dem.min(), dem.max(), dem.mean()))
print('保存: {} ({}x{})'.format(OUT_DIR, rows, cols))
