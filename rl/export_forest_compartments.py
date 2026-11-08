"""Decode the locally verified OviO forest-boundary variant without Ovi software.

This is a strict decoder for the observed file structure, not a general OVOBJ
reader. Geographic coordinate units are established by record and geometry
checks; the datum is NOT established. Outputs require alignment verification
before use alongside WGS84 / CGCS2000 / GCJ-02 data. No network is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import struct


class AttributeCells(HTMLParser):
    """Read nested HTML table cells as text, without executing the HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.cells = []

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.stack.append([])

    def handle_data(self, data):
        if self.stack:
            self.stack[-1].append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.stack:
            self.cells.append(" ".join("".join(self.stack.pop()).split()))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signed_area(ring):
    # Shift the origin to avoid cancellation on geographic coordinates.
    ox, oy = ring[0]
    return sum((a[0] - ox) * (b[1] - oy) - (b[0] - ox) * (a[1] - oy)
               for a, b in zip(ring, ring[1:])) / 2


def contains(point, ring):
    x, y = point
    inside = False
    for a, b in zip(ring, ring[1:]):
        if (a[1] > y) != (b[1] > y):
            crossing = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < crossing:
                inside = not inside
    return inside


def approximate_metrics(rings):
    """Local WGS84 ellipsoid scaling as an independent sanity check only.

    Agreement with stored area/perimeter cannot distinguish WGS84, CGCS2000,
    or an offset coordinate system. It is not a datum determination.
    """
    points = [point for ring in rings for point in ring]
    lat0 = sum(p[1] for p in points) / len(points)
    lon0 = points[0][0]
    phi = math.radians(lat0)
    a, e2 = 6378137.0, 6.6943799901413165e-3
    root = math.sqrt(1 - e2 * math.sin(phi) ** 2)
    xs = math.pi / 180 * a / root * math.cos(phi)
    ys = math.pi / 180 * a * (1 - e2) / root ** 3
    area_ha = perimeter_m = 0.0
    for ring in rings:
        xy = [((x - lon0) * xs, (y - lat0) * ys) for x, y in ring]
        area_ha += abs(signed_area(xy)) / 10000
        perimeter_m += sum(math.hypot(a[0] - b[0], a[1] - b[1])
                           for a, b in zip(xy, xy[1:]))
    return area_ha, perimeter_m


def decode_geometry(data, offset, end, fid):
    require(offset + 24 <= end, f"FID {fid}: missing geometry header")
    count, latitude, longitude = struct.unpack_from("<QQQ", data, offset)
    require(4 <= count <= 1_000_000, f"FID {fid}: unsupported point count")
    offset += 24
    integer_points = [(longitude, latitude)]
    for _ in range(count - 1):
        require(offset < end, f"FID {fid}: truncated delta header")
        tag = data[offset]
        payload_bytes = (tag >> 2) & 15
        require(not tag & 2 and 1 <= payload_bytes <= 8,
                f"FID {fid}: unsupported delta tag 0x{tag:02x}")
        require(offset + 1 + payload_bytes <= end, f"FID {fid}: truncated delta")
        bits = 4 * payload_bytes
        payload = int.from_bytes(data[offset + 1:offset + 1 + payload_bytes], "big")
        offset += 1 + payload_bytes
        latitude_bits = (payload >> bits) | ((tag & 1) << bits)
        require(latitude_bits % 2 == 0, f"FID {fid}: unsupported latitude flag")
        delta_lat = latitude_bits >> 1
        delta_lon = payload & ((1 << bits) - 1)
        latitude += -delta_lat if tag & 128 else delta_lat
        longitude += -delta_lon if tag & 64 else delta_lon
        integer_points.append((longitude, latitude))

    if offset == end:
        starts = [0]
    else:
        # The observed multipart record carries three reserved zero bytes
        # followed by one flag per point: 3 at the first point, 1 at new rings.
        tail = data[offset:end]
        require(len(tail) == count + 3 and tail[:3] == b"\0" * 3,
                f"FID {fid}: unsupported trailing geometry data")
        flags = tail[3:]
        require(flags[0] == 3 and set(flags[1:]) <= {0, 1},
                f"FID {fid}: unsupported ring flags")
        starts = [i for i, flag in enumerate(flags) if flag]

    rings = []
    for start, stop in zip(starts, starts[1:] + [count]):
        integer_ring = integer_points[start:stop]
        require(len(integer_ring) >= 4 and integer_ring[0] == integer_ring[-1],
                f"FID {fid}: unclosed or degenerate ring")
        ring = [[lon / 1e8, lat / 1e8] for lon, lat in integer_ring]
        require(all(102 < lon < 105 and 27 < lat < 30 for lon, lat in ring),
                f"FID {fid}: coordinates outside the expected Leibo region")
        require(signed_area(ring) != 0, f"FID {fid}: zero-area ring")
        rings.append(ring)

    # The sole multipart object in this source contains two exterior rings.
    # Fail on nested rings rather than silently exporting a hole as an island.
    for i, ring in enumerate(rings):
        for j, other in enumerate(rings):
            if i != j:
                require(not contains(ring[0], other),
                        f"FID {fid}: nested rings need separate interpretation")
    rings = [ring if signed_area(ring) > 0 else list(reversed(ring)) for ring in rings]
    return rings, count


def decode(source):
    data = source.read_bytes()
    require(len(data) >= 264 and data[:4] == b"OviO", "Not an OviO object file")
    require(struct.unpack_from("<I", data, 12)[0] == 105, "Unsupported OviO header code")
    require(struct.unpack_from("<Q", data, 24)[0] == len(data), "File length header mismatch")
    offset = 264
    features, types, fids, groups = [], Counter(), set(), Counter()
    vertex_count = ring_count = 0
    area_ratios, perimeter_ratios, multipart = [], [], []
    while offset < len(data):
        require(offset + 32 <= len(data), "Truncated record envelope")
        size = struct.unpack_from("<I", data, offset)[0]
        end = offset + size + 8
        require(size >= 24 and end <= len(data), f"Bad record length at {offset}")
        kind = struct.unpack_from("<I", data, offset + 28)[0]
        types[kind] += 1
        require(kind in (13, 30), f"Unsupported record type {kind}")
        if kind == 13:
            blocks = list(re.finditer(rb"<!\[CDATA\[([\s\S]*?)\]\]>", data[offset:end]))
            require(len(blocks) == 1, f"Expected one attribute block at {offset}")
            block = blocks[0]
            parser = AttributeCells()
            parser.feed(block.group(1).decode("utf-8"))
            cells = parser.cells
            require(len(cells) == 18 and cells[-1] == "", "Unexpected attribute table")
            properties = dict(zip(cells[1:-1:2], cells[2:-1:2]))
            require(set(properties) == {"FID", "局", "场", "作业区", "林班", "Shape_Leng", "名称", "面积"},
                    "Unexpected source attribute keys")
            fid = properties["FID"]
            require(fid not in fids, f"Duplicate FID {fid}")
            fids.add(fid)
            rings, count = decode_geometry(data, offset + block.end(), end, fid)
            area, perimeter = approximate_metrics(rings)
            require(float(properties["面积"]) > 0 and float(properties["Shape_Leng"]) > 0,
                    f"FID {fid}: invalid source metrics")
            area_ratio = area / float(properties["面积"])
            perimeter_ratio = perimeter / float(properties["Shape_Leng"])
            require(abs(area_ratio - 1) < .005 and abs(perimeter_ratio - 1) < .005,
                    f"FID {fid}: decoded geometry disagrees with source area/perimeter")
            area_ratios.append(area_ratio)
            perimeter_ratios.append(perimeter_ratio)
            groups[properties["场"]] += 1
            vertex_count += count
            ring_count += len(rings)
            if len(rings) > 1:
                multipart.append({"FID": fid, "ringVertexCounts": list(map(len, rings))})
            geometry = {"type": "Polygon", "coordinates": rings} if len(rings) == 1 else {
                "type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}
            features.append({"type": "Feature", "id": f"ovobj-fid-{fid}",
                             "properties": properties, "geometry": geometry})
        offset = end
    require(offset == len(data), "Records do not reach EOF exactly")
    require(features, "No boundary records")
    points = [point for feature in features
              for polygon in ([feature["geometry"]["coordinates"]] if feature["geometry"]["type"] == "Polygon"
                              else feature["geometry"]["coordinates"])
              for ring in polygon for point in ring]
    report = {
        "records": sum(types.values()), "recordTypes": dict(types), "features": len(features),
        "rings": ring_count, "verticesIncludingClosures": vertex_count,
        "fieldCounts": dict(sorted(groups.items())), "multipart": multipart,
        "allRecordLengthsValid": True, "allRingsClosedExactlyInIntegerCoordinates": True,
        "allCoordinatesInLeiboBounds": True,
        "approxAreaToStoredRatioRange": [min(area_ratios), max(area_ratios)],
        "approxPerimeterToStoredRatioRange": [min(perimeter_ratios), max(perimeter_ratios)],
        "metricValidation": "Local ellipsoid approximation; shape check only, not CRS validation",
    }
    return {
        "type": "FeatureCollection",
        "name": "雷波林班 · OVOBJ 提取边界（坐标基准待核验）",
        "bbox": [min(p[0] for p in points), min(p[1] for p in points),
                 max(p[0] for p in points), max(p[1] for p in points)],
        "metadata": {
            "source": source.name, "sourceSha256": hashlib.sha256(data).hexdigest(),
            "sourceBytes": len(data), "decoder": "observed OviO header code 105",
            "coordinateStatus": {
                "units": "decimal degrees", "order": "longitude, latitude",
                "datum": "unverified", "epsg": None, "defaultVisible": False,
                "requiresAlignmentVerification": True,
                "warning": "已验证二进制坐标结构及边界闭合；未由奥维官方导出核验 WGS84、CGCS2000 或 GCJ-02 偏移。不能仅凭数值范围或面积一致判定坐标基准。",
                "interchangeNote": "诊断用途 GeoJSON 数据；坐标基准确认前，不按 RFC 7946 的 WGS84 保证用于跨系统叠加。",
            },
            "usage": "仅林班管理边界；不包含 DEM、道路通行性或机器人训练结果。",
            "attributeNote": "保留原属性字符串；场号与林班号是不同层级。面积近似匹配公顷、Shape_Leng 近似匹配米，源表未声明单位。",
            "validation": report,
        },
        "features": features,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "林场范围哈" / "雷波林班(1).ovobj")
    parser.add_argument("--geojson", type=Path, default=root / "林场范围哈" / "forest_compartments.geojson")
    parser.add_argument("--javascript", type=Path, default=root / "rl" / "forest_compartments.js")
    args = parser.parse_args()
    source = args.input.resolve()
    require(source not in (args.geojson.resolve(), args.javascript.resolve()), "Outputs cannot overwrite source")
    collection = decode(source)
    serialized = json.dumps(collection, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    args.geojson.write_text(serialized + "\n", encoding="utf-8")
    args.javascript.write_text("// Local source boundaries. Datum unverified; do not enable overlay by default.\n"
                               + "window.FOREST_COMPARTMENTS=" + serialized + ";\n", encoding="utf-8")
    print(json.dumps({"geojson": str(args.geojson), "javascript": str(args.javascript),
                      "validation": collection["metadata"]["validation"],
                      "coordinateDatum": "unverified", "bbox": collection["bbox"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
