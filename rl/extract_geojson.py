"""Extract the complete forest parcel geometries and provenance from local KMZ files."""

import argparse
import hashlib
from html import unescape
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / '林场范围哈'
WEB_DATA = ROOT / 'rl' / 'forest_features.js'


def tag(element):
    return element.tag.rsplit('}', 1)[-1]


def child_text(element, name):
    return next(((child.text or '').strip() for child in element if tag(child) == name), '')


def table_value(description, label):
    cell = r'<td(?:\s[^>]*)?>'
    match = re.search(cell + r'\s*' + re.escape(label) + r'\s*</td>\s*'
                      + cell + r'([^<]*)</td>', description, re.IGNORECASE)
    return unescape(match.group(1)).strip() if match else ''


def ring_from_boundary(boundary):
    coordinates = next((el.text for el in boundary.iter() if tag(el) == 'coordinates'), None)
    ring = [list(map(float, vertex.split(',')[:2])) for vertex in (coordinates or '').split()]
    if len(ring) < 3 or any(len(vertex) != 2 for vertex in ring):
        raise ValueError('A KML polygon ring requires at least three longitude/latitude pairs')
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def extract(directory):
    features = []
    sources = []
    for path in sorted(directory.glob('*.kmz')):
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith('.kml')]
            if not members:
                raise ValueError(f'{path.name} does not contain KML')
            source_start = len(features)
            for member in members:
                document = ET.fromstring(archive.read(member))
                for placemark in (el for el in document.iter() if tag(el) == 'Placemark'):
                    description = child_text(placemark, 'description')
                    zone = table_value(description, '功能分区')
                    source_area = table_value(description, '面积')
                    name = child_text(placemark, 'name') or zone or path.stem
                    for polygon in (el for el in placemark.iter() if tag(el) == 'Polygon'):
                        outer = next(el for el in polygon if tag(el) == 'outerBoundaryIs')
                        holes = [el for el in polygon if tag(el) == 'innerBoundaryIs']
                        rings = [ring_from_boundary(boundary) for boundary in (outer, *holes)]
                        features.append({
                            'type': 'Feature',
                            'id': f'{path.stem}-{len(features) - source_start + 1}',
                            'properties': {
                                'name': name,
                                'category': path.stem,
                                'functionZone': zone or None,
                                'reportedArea': source_area or None,
                                'source': path.name,
                                'sourceKml': member,
                                'altitudeMode': child_text(polygon, 'altitudeMode') or 'clampToGround',
                                'points': sum(map(len, rings)),
                            },
                            'geometry': {'type': 'Polygon', 'coordinates': rings},
                        })
        sources.append({
            'name': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'parcels': len(features) - source_start,
        })
    if not features:
        raise ValueError(f'No KMZ parcel polygons found in {directory}')
    return {'type': 'FeatureCollection', 'features': features, 'sources': sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE_DIR)
    parser.add_argument('--web-output', type=Path, default=WEB_DATA)
    args = parser.parse_args()
    data = extract(args.source)
    geojson = args.source / 'forest_features.geojson'
    geojson.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    web_payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    args.web_output.write_text('window.FOREST_FEATURES = ' + web_payload + ';\n', encoding='utf-8')
    print(f'{len(data["features"])} parcels, {sum(f["properties"]["points"] for f in data["features"])} vertices')
    print(geojson)
    print(args.web_output)


if __name__ == '__main__':
    main()
