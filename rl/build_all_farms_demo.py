# -*- coding: utf-8 -*-
"""用 forest_compartments.geojson 的真实边界构建 4 个林场全部 271 林班 WebGIS 演示页。

多边形用 Douglas-Peucker 简化以控制 HTML 体积，属性保留 面积/周长/名称。
输出: forest_all_farms_demo.html
"""
import io
import json
import math
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SRC = r'E:\forestrobo\林场范围哈\forest_compartments.geojson'
FEAT_SRC = r'E:\forestrobo\林场范围哈\forest_features.geojson'
OUT = r'E:\forestrobo\rl\forest_all_farms_demo.html'

TOL_M = 8.0  # 简化容差 (米)


def perp_dist(pt, a, b):
    """点 pt 到线段 ab 的垂直距离 (近似平面, 度转米)。"""
    x0, y0 = pt
    x1, y1 = a
    x2, y2 = b
    dx, dy = (x2 - x1) * math.cos(math.radians(y1)) * 111320.0, (y2 - y1) * 110574.0
    px, py = (x0 - x1) * math.cos(math.radians(y1)) * 111320.0, (y0 - y1) * 110574.0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px, py)
    t = max(0.0, min(1.0, (px * dx + py * dy) / L2))
    return math.hypot(px - t * dx, py - t * dy)


def simplify(pts, tol=TOL_M):
    """Douglas-Peucker, 输入/输出 [[lon,lat],...] 闭合环。"""
    if len(pts) <= 8:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        dmax, imax = -1.0, i
        for k in range(i + 1, j):
            d = perp_dist(pts[k], pts[i], pts[j])
            if d > dmax:
                dmax, imax = d, k
        if dmax > tol:
            keep[imax] = True
            stack.append((i, imax))
            stack.append((imax, j))
    out = [p for p, k in zip(pts, keep) if k]
    return out if len(out) >= 4 else pts


def ring_area_ha(ring):
    """球面近似面积 (公顷)。"""
    if len(ring) < 4:
        return 0.0
    s = 0.0
    for k in range(len(ring) - 1):
        x1, y1 = ring[k][0], ring[k][1]
        x2, y2 = ring[k + 1][0], ring[k + 1][1]
        s += math.radians(x2 - x1) * (2 + math.sin(math.radians(y1)) + math.sin(math.radians(y2)))
    return abs(s) * 6378137.0 * 6378137.0 / 2.0 / 10000.0


def centroid(ring):
    xs = [p[0] for p in ring[:-1]] or [p[0] for p in ring]
    ys = [p[1] for p in ring[:-1]] or [p[1] for p in ring]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


with open(SRC, encoding='utf-8') as f:
    gj = json.load(f)

comps = []  # {farm, zone, num, name, area_mu, perim, pts(简化后 [lat,lon]), cen}
stat = {}
nverts_raw = nverts_simp = 0
for ft in gj.get('features', []):
    p = ft.get('properties', {}) or {}
    geom = ft.get('geometry', {}) or {}
    farm = str(p.get('场', '?'))
    zone = str(p.get('作业区', '?'))
    num = str(p.get('林班', '?'))
    name = p.get('名称') or '{}林班'.format(num)
    area_mu = float(p.get('面积') or 0)
    perim = float(p.get('Shape_Leng') or 0)

    coords = geom.get('coordinates')
    polys = [coords] if geom.get('type') == 'Polygon' and coords else (coords or [])

    best_ring, best_ha = None, 0.0
    for poly in polys:
        for ring in poly:
            nverts_raw += len(ring)
            r = simplify([[float(x) for x in pt[:2]] for pt in ring])
            nverts_simp += len(r)
            ha = ring_area_ha(r)
            if ha > best_ha:
                best_ha, best_ring = ha, r
    if not best_ring:
        continue

    # 面积字段优先; 若缺失用几何面积换算 (1 ha = 15 亩)
    if area_mu <= 0:
        area_mu = best_ha * 15.0

    # Leaflet 用 [lat, lon]
    pts_ll = [[pt[1], pt[0]] for pt in best_ring]
    comps.append({
        'farm': farm, 'zone': zone, 'num': num, 'name': name,
        'area_mu': round(area_mu, 1), 'perim': round(perim, 0),
        'ha': round(best_ha, 1),
        'pts': [[round(a, 6), round(b, 6)] for a, b in pts_ll],
        'cen': [round(c, 6) for c in (lambda c: [c[1], c[0]])(centroid(best_ring))],
    })
    s = stat.setdefault(farm, {'n': 0, 'mu': 0.0})
    s['n'] += 1
    s['mu'] += area_mu

# KMZ 地块 (林下经济 + 直服用地) 叠加层
feat_layers = []
try:
    with open(FEAT_SRC, encoding='utf-8') as f:
        fg = json.load(f)
    for ft in fg.get('features', []):
        p = ft.get('properties', {}) or {}
        geom = ft.get('geometry', {}) or {}
        name = p.get('name') or p.get('Name') or '地块'
        coords = geom.get('coordinates')
        if not coords:
            continue
        polys = [coords] if geom.get('type') == 'Polygon' else (coords if geom.get('type') == 'MultiPolygon' else [])
        rings = []
        for poly in polys:
            ring = poly[0] if isinstance(poly[0][0][0], (int, float)) else poly
            rings.append([[round(pt[1], 6), round(pt[0], 6)] for pt in ring])
        if rings:
            feat_layers.append({'name': name, 'rings': rings})
except Exception as e:
    print('KMZ 地块加载失败(忽略): {}'.format(e))

print('林班: {} 个, 顶点 {} -> {} (简化 {:.0f}%)'.format(
    len(comps), nverts_raw, nverts_simp, 100 * (1 - nverts_simp / max(1, nverts_raw))))
print('叠加地块: {} 个'.format(len(feat_layers)))

FARM_STYLE = {
    '211': {'color': '#e74c3c', 'label': '211林场 (27林班)'},
    '212': {'color': '#3498db', 'label': '212林场 (105林班)'},
    '213': {'color': '#2ecc71', 'label': '213林场 (62林班)'},
    '214': {'color': '#9b59b6', 'label': '214林场 (77林班)'},
}

rows = []
for i, c in enumerate(comps):
    rows.append(
        '{{f:"{f}",z:"{z}",n:"{n}",nm:"{nm}",a:{a},ha:{ha},p:{p},c:[{cx},{cy}],pts:[{pts}]}}'.format(
            f=c['farm'], z=c['zone'], n=c['num'], nm=c['name'],
            a=c['area_mu'], ha=c['ha'], p=c['perim'],
            cx=c['cen'][0], cy=c['cen'][1],
            pts=','.join('[{},{:f}]'.format(pt[0], pt[1]) for pt in c['pts'])))

feat_rows = []
for fl in feat_layers:
    rr = []
    for ring in fl['rings']:
        rr.append('[{}]'.format(','.join('[{},{:f}]'.format(pt[0], pt[1]) for pt in ring)))
    feat_rows.append('{{nm:"{}",rings:[{}]}}'.format(fl['name'], ','.join(rr)))

stat_rows = ','.join(
    '"{}":{{n:{},mu:{:.0f}}}'.format(f, v['n'], v['mu']) for f, v in sorted(stat.items()))

all_cen = [c['cen'] for c in comps]
mid = [sum(p[0] for p in all_cen) / len(all_cen), sum(p[1] for p in all_cen) / len(all_cen)]

html = '''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>雷波林业局 · 211/212/213/214林场 全部林班 (真实边界)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
body{margin:0;font-family:system-ui,"Microsoft YaHei",sans-serif}
#map{width:100vw;height:100vh}
.panel{position:absolute;top:10px;left:10px;background:rgba(13,17,23,.92);color:#e6edf3;padding:14px;border-radius:10px;z-index:1000;width:250px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.5);max-height:calc(100vh - 40px);overflow-y:auto}
.panel h3{margin:0 0 4px;font-size:14px}
.panel .info{color:#8b949e;font-size:11px;margin-bottom:8px;line-height:1.5}
.farm-row{display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:6px;cursor:pointer;margin:2px 0}
.farm-row:hover{background:#21262d}
.farm-row .sw{width:12px;height:12px;border-radius:3px;flex:none}
.farm-row .cnt{margin-left:auto;color:#8b949e}
.stat{margin-top:8px;border-top:1px solid #30363d;padding-top:8px;color:#8b949e;line-height:1.7}
.legend-bar{position:absolute;bottom:10px;left:10px;background:rgba(13,17,23,.92);color:#e6edf3;padding:8px 12px;border-radius:8px;font-size:11px;z-index:1000}
.leaflet-popup-content-wrapper{background:#21262d;color:#e6edf3;border-radius:8px;font-size:12px}
.leaflet-popup-tip{background:#21262d}
.coord{position:absolute;bottom:10px;right:10px;background:rgba(13,17,23,.92);color:#8b949e;padding:4px 10px;border-radius:6px;font-size:11px;z-index:1000;font-family:Consolas,monospace}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
<h3>雷波林业局 9120 · 林班分布</h3>
<div class="info">数据源: forest_compartments.geojson (真实边界)<br>4个林场 · __NCOMP__个林班 · 面积字段(亩)<br>坐标系: WGS84 / CGCS2000</div>
<div id="farmList"></div>
<div class="stat" id="stat"></div>
</div>
<div class="legend-bar">透明度 = 面积大小 · 点击林班查看详情 · 勾选林场开关显示</div>
<div class="coord" id="coord">--</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
'use strict';
var COMPS=[__COMPS__];
var FEATS=[__FEATS__];
var STAT=__STAT__;
var STYLE={
 '211':{color:'#e74c3c',label:'211林场'},
 '212':{color:'#3498db',label:'212林场'},
 '213':{color:'#2ecc71',label:'213林场'},
 '214':{color:'#9b59b6',label:'214林场'}
};

var map=L.map('map').setView([__MIDLAT__,__MIDLON__],12);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{
  maxZoom:19,attribution:'Esri, Maxar'
}).addTo(map);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',{
  maxZoom:19,opacity:.55
}).addTo(map);

var maxA=0;COMPS.forEach(function(c){if(c.a>maxA)maxA=c.a;});

var layers={};var allBounds=L.latLngBounds();
COMPS.forEach(function(c){
  var st=STYLE[c.f]||{color:'#888',label:c.f};
  var t=c.a/maxA;
  var poly=L.polygon(c.pts,{
    color:st.color,weight:1.2,opacity:.9,
    fillColor:st.color,fillOpacity:.14+.42*t
  });
  poly.bindPopup(
    '<b>'+c.nm+'</b><br>'+
    '所属: 9120局 → '+c.f+'场 → '+c.z+'作业区<br>'+
    '面积: '+c.a.toFixed(1)+' 亩 ('+(c.a/15).toFixed(1)+' 公顷)<br>'+
    '几何面积: '+c.ha.toFixed(1)+' 公顷<br>'+
    '周长: '+c.p.toFixed(0)+' m<br>'+
    '中心: '+c.c[1].toFixed(5)+'°E, '+c.c[0].toFixed(5)+'°N'
  );
  poly.on('mouseover',function(e){e.target.setStyle({weight:3});});
  poly.on('mouseout',function(e){e.target.setStyle({weight:1.2});});
  (layers[c.f]=layers[c.f]||[]).push(poly);
  allBounds.extend(c.pts);
});

// KMZ 地块叠加 (虚线黄框)
var featLayer=L.layerGroup();
FEATS.forEach(function(f){
  f.rings.forEach(function(r){
    L.polygon(r,{color:'#f1c40f',weight:2,dashArray:'6 4',fillOpacity:.08})
      .bindPopup('<b>'+f.nm+'</b><br>来源: KMZ 地块')
      .addTo(featLayer);
  });
});
featLayer.addTo(map);

var farmGroups={};
Object.keys(layers).forEach(function(f){
  farmGroups[f]=L.layerGroup(layers[f]).addTo(map);
});

// 标注 (缩放到一定级别显示)
var labelGroup=L.layerGroup();
COMPS.forEach(function(c){
  if(+c.num===+c.num){
    L.marker(c.c,{icon:L.divIcon({className:'',iconSize:[24,14],iconAnchor:[12,7],
      html:'<div style="color:#fff;font-size:10px;font-weight:600;text-shadow:0 0 3px #000,-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000">'+c.f+'-'+c.num+'</div>'}),
      interactive:false}).addTo(labelGroup);
  }
});
labelGroup.addTo(map);
map.on('zoomend',function(){
  if(map.getZoom()>=14)map.addLayer(labelGroup);else map.removeLayer(labelGroup);
});
if(map.getZoom()<14)map.removeLayer(labelGroup);

map.fitBounds(allBounds,{padding:[15,15]});
L.control.scale({imperial:false,position:'bottomleft'}).addTo(map);
map.on('mousemove',function(e){
  document.getElementById('coord').textContent=
    e.latlng.lat.toFixed(6)+'°N, '+e.latlng.lng.toFixed(6)+'°E';
});

// 面板
var list=document.getElementById('farmList');
Object.keys(STYLE).forEach(function(f){
  var st=STYLE[f],s=STAT[f]||{n:0,mu:0};
  var row=document.createElement('div');row.className='farm-row';
  row.innerHTML='<span class="sw" style="background:'+st.color+'"></span>'+
    '<span>'+st.label+'</span><span class="cnt">'+s.n+'班<br>'+Math.round(s.mu/10000*10)/10+'万亩</span>';
  row.onclick=function(){
    if(map.hasLayer(farmGroups[f])){map.removeLayer(farmGroups[f]);row.style.opacity=.35;}
    else{map.addLayer(farmGroups[f]);row.style.opacity=1;}
    var b=L.latLngBounds();
    COMPS.forEach(function(c){if(c.f===f)b.extend(c.pts);});
    if(b.isValid())map.fitBounds(b,{padding:[20,20]});
  };
  list.appendChild(row);
});
var fr=document.createElement('div');fr.className='farm-row';
fr.innerHTML='<span class="sw" style="background:#f1c40f"></span><span>KMZ 地块 (石斛/黄莲等)</span><span class="cnt">'+FEATS.length+'</span>';
fr.onclick=function(){
  if(map.hasLayer(featLayer)){map.removeLayer(featLayer);fr.style.opacity=.35;}
  else{map.addLayer(featLayer);fr.style.opacity=1;}
};
list.appendChild(fr);

var totalMu=0,nTot=0;
Object.keys(STAT).forEach(function(f){totalMu+=STAT[f].mu;nTot+=STAT[f].n;});
document.getElementById('stat').innerHTML=
  '合计: '+nTot+' 个林班<br>总面积: '+(totalMu/10000).toFixed(1)+' 万亩 ('+(totalMu/15/10000).toFixed(2)+' 万公顷)';
</script>
</body>
</html>
'''

html = (html
        .replace('__COMPS__', ',\n'.join(rows))
        .replace('__FEATS__', ',\n'.join(feat_rows) if feat_rows else '')
        .replace('__STAT__', '{' + stat_rows + '}')
        .replace('__NCOMP__', str(len(comps)))
        .replace('__MIDLAT__', '{:.5f}'.format(mid[0]))
        .replace('__MIDLON__', '{:.5f}'.format(mid[1])))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print('已生成 {} ({:.1f} MB)'.format(OUT, len(html.encode('utf-8')) / 1048576))
