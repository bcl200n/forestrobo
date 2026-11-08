"""Export the Jianchang street case and existing replays onto one georeferenced image.

Run with rl/.venv/Scripts/python.exe. Optional GIS image dependencies are
installed with the same interpreter using: -m pip install pyproj rasterio
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'rl/cases/xichang-jianchang-core.geojson'
RUNS = ROOT / 'rl/models/xichang_interior_20261107_final'
METADATA = ROOT / 'data/jianchang/local_metadata'
IMAGE = ROOT / 'data/jianchang/orthophoto/DOM_DOM.tif'
OUTPUT = ROOT / 'web/jianchang-assets'
CRS = 'EPSG:4543'


def file_hash(path):
    digest = sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def projection():
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError('Install GIS dependencies with rl/.venv/Scripts/python.exe -m pip install pyproj rasterio') from exc
    return Transformer.from_crs('EPSG:4326', CRS, always_xy=True)


def case_data(case_path, run_directory):
    case = load_json(case_path)
    case_hash = file_hash(case_path)
    meta = case['case']
    size = int(meta['grid_size'])
    cell = float(meta['cell_m'])
    origin = np.asarray(meta['southwest_offset_m'], dtype=float)
    reference = np.asarray(meta['reference_lonlat'], dtype=float)
    meters_per_degree = np.array([111320 * np.cos(np.deg2rad(reference[1])), 111320])
    transform = projection()

    def project(lonlat):
        lonlat = np.asarray(lonlat, dtype=float)
        east, north = transform.transform(lonlat[..., 0], lonlat[..., 1], errcheck=True)
        return np.stack((east, north), axis=-1)

    def from_offsets(offsets):
        return project(reference + np.asarray(offsets) / meters_per_degree)

    side = size * cell
    t = np.linspace(0., side, 65)
    zeros = np.zeros_like(t)
    full = np.full_like(t, side)
    boundary = origin + np.concatenate((np.column_stack((t, zeros)), np.column_stack((full, t)),
                                       np.column_stack((t, full)), np.column_stack((zeros, t))))
    projected_boundary = from_offsets(boundary)
    lower = projected_boundary.min(axis=0)
    upper = projected_boundary.max(axis=0)
    extent = upper - lower

    def normalized(points):
        return np.round((np.asarray(points) - lower) / extent, 7).tolist()

    features, labels = [], []
    named = set()
    for feature in case['features']:
        properties = feature['properties']
        coordinates = feature['geometry']['coordinates']
        if feature['geometry']['type'] == 'Polygon':
            if len(coordinates) != 1:
                raise ValueError('Case exporter requires explicit handling for polygon holes')
            coordinates = coordinates[0]
        points = normalized(project(coordinates))
        name = properties.get('name', '')
        features.append(dict(kind=properties['kind'], name=name, osmWay=properties['osm_way'], points=points))
        in_view = [point for point in points if 0.03 <= point[0] <= .97 and .03 <= point[1] <= .97]
        if name and name not in named and in_view:
            labels.append(dict(text=name, point=in_view[len(in_view) // 2]))
            named.add(name)

    runs = []
    for scenario in ('corridor', 'rubble', 'dropout'):
        for mode in ('cues_only', 'street_sweep'):
            run_id = f'{scenario}-{mode}'
            path = run_directory / f'{run_id}.json'
            report = load_json(path)
            if report['source_sha256'] != case_hash:
                raise ValueError(f'{path.name} was generated from a different source map')
            grid = np.asarray(report['path_grid'], dtype=float)
            if grid.shape != (report['steps'] + 1, 2) or not np.isfinite(grid).all():
                raise ValueError(f'{path.name} has an invalid path sequence')
            if np.any(grid < 0) or np.any(grid >= size):
                raise ValueError(f'{path.name} has coordinates outside the simulation grid')
            points = normalized(from_offsets(origin + (grid + .5) * cell))
            if np.any(np.asarray(points) < 0) or np.any(np.asarray(points) > 1):
                raise ValueError(f'{path.name} falls outside the image extent')
            runs.append(dict(id=run_id, scenario=scenario, mode=mode, seed=report['map_seed'], path=points,
                firstAllCuesStep=report['first_all_cues_step'], steps=report['steps'], success=report['success'],
                found=report['found'], reachableCoverage=report['reachable_cue_coverage'],
                requiredCoverage=report['required_coverage'], distanceM=report['distance_m'],
                collisions=report.get('collisions'), unsafeMotion=report.get('unsafe_motion'),
                sourceHash=file_hash(path)))

    old_extent = np.array([[-280, -244], [232, -244], [232, 268], [-280, 268], [-280, -244]])
    return dict(schema='jianchang-georeferenced-replay-v1',
        meta=dict(name=meta['name'], gridSize=size, cellM=cell, caseHash=case_hash,
                  referenceLonLat=reference.tolist(), seed=38000,
                  projection='WGS84 coordinates transformed to CGCS2000 Gauss-Kruger CM 102E; always_xy',
                  projectionOperation=transform.description, projectionAccuracyM=transform.accuracy,
                  coordinateConvention='u eastward, v northward, origin southwest; screen y = 1-v'),
        extent=dict(crs=CRS, min=lower.tolist(), max=upper.tolist(), widthM=float(extent[0]), heightM=float(extent[1])),
        features=features, runs=runs, labels=labels, previousExtent=normalized(from_offsets(old_extent)),
        attribution=dict(text='© OpenStreetMap contributors', url='https://www.openstreetmap.org/copyright',
                         license='ODbL 1.0', imagery='User-provided orthophoto, source CRS and world file retained'),
        limitations=[
            '轨迹来自已有二维仿真，路宽、瓦砾、线索和探测距离为仿真假设。',
            '正射影像用于对照位置。影像中的空地、院落和屋顶没有自动转成已验证通行区。',
            'OSM道路与测绘影像的现场配准精度尚未用控制点核验。',
            '原始正射影像包含白色背景，白色区域的地表内容未知。',
            '探测覆盖率和确认数为回合最终指标，轨迹点保留原始决策步顺序。'])


def export_image(path, world_file, crs_file, extent, destination, max_pixels):
    try:
        import rasterio
        from affine import Affine
        from PIL import Image
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError('Install GIS dependencies with rl/.venv/Scripts/python.exe -m pip install pyproj rasterio') from exc

    values = [float(line) for line in world_file.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(values) != 6 or not np.isfinite(values).all():
        raise ValueError('World file must contain six finite numbers in A,D,B,E,C,F order')
    a, d, b, e, center_x, center_y = values
    expected = Affine(a, b, center_x - (a + b) / 2, d, e, center_y - (d + e) / 2)
    expected_crs = rasterio.crs.CRS.from_wkt(crs_file.read_text(encoding='utf-8-sig'))
    if expected_crs.to_epsg() != 4543:
        raise ValueError('The supplied projection file must identify EPSG:4543')
    width = max_pixels
    height = max(1, round(width * extent['heightM'] / extent['widthM']))
    if height > max_pixels:
        height = max_pixels
        width = max(1, round(height * extent['widthM'] / extent['heightM']))
    with rasterio.open(path) as source:
        if source.crs is not None and source.crs != expected_crs:
            raise ValueError(f'GeoTIFF CRS {source.crs} differs from the supplied projection file')
        embedded = source.transform
        if embedded.is_identity:
            transform = expected
            transform_source = 'world file, pixel centers converted to corner transform'
        else:
            if not np.allclose(tuple(embedded)[:6], tuple(expected)[:6], atol=1e-5, rtol=0):
                raise ValueError('GeoTIFF geotransform disagrees with world file pixel centers')
            transform = embedded
            transform_source = 'Dataset transform agrees with supplied world-file pixel centers; may use a sidecar'
        if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
            raise ValueError('This exporter requires an east-positive, north-up orthophoto')
        bounds = (*extent['min'], *extent['max'])
        window = from_bounds(*bounds, transform=transform)
        if window.col_off >= source.width or window.row_off >= source.height or window.col_off + window.width <= 0 or window.row_off + window.height <= 0:
            raise ValueError('The orthophoto does not overlap the requested map extent')
        if source.count < 3:
            raise ValueError('Expected an orthophoto with at least three color bands')
        colors = list(source.colorinterp)
        names = [color.name for color in colors]
        bands = [names.index(name)+1 for name in ('red', 'green', 'blue')] if all(name in names for name in ('red', 'green', 'blue')) else [1, 2, 3]
        pixels = source.read(bands, window=window, out_shape=(3, height, width), boundless=True,
                             fill_value=0, resampling=Resampling.bilinear)
        valid = source.read_masks(bands, window=window, out_shape=(3, height, width),
                                  boundless=True, resampling=Resampling.nearest).min(axis=0) > 0
        valid &= np.isfinite(pixels).all(axis=0)
        if not valid.any():
            raise ValueError('No valid image pixels within the map extent')
        stretch = None
        if pixels.dtype != np.uint8:
            stretch = []
            rendered = np.zeros_like(pixels, dtype=np.uint8)
            for channel in range(3):
                low, high = np.percentile(pixels[channel][valid], [1, 99])
                if high <= low:
                    high = low + 1
                stretch.append([float(low), float(high)])
                rendered[channel] = np.clip((pixels[channel] - low) * 255. / (high - low), 0, 255).astype(np.uint8)
            pixels = rendered
        pixels[:, ~valid] = np.array([234, 232, 225])[:, None]
        rgb = np.moveaxis(pixels, 0, -1)
        temporary = destination.with_suffix('.jpg.tmp')
        Image.fromarray(rgb).save(temporary, format='JPEG', quality=92, optimize=True)
        temporary.replace(destination)
        metadata = dict(available=True, url='jianchang-assets/orthophoto.jpg', width=width, height=height,
            coverageFraction=float(valid.mean()), sourceSize=[source.width, source.height],
            coverageDefinition='Declared raster valid-data mask only; source white background is not classified',
            sourceCrs=str(source.crs or expected_crs), transformSource=transform_source,
            worldFileHash=file_hash(world_file), projectionFileHash=file_hash(crs_file),
            pixelSizeM=[float(transform.a), float(-transform.e)], stretchPercentiles=stretch)
    metadata['sourceHash'] = file_hash(path)
    metadata['imageHash'] = file_hash(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-only', action='store_true', help='Export map and recorded replays without opening the orthophoto')
    parser.add_argument('--image', type=Path, default=IMAGE)
    parser.add_argument('--world-file', type=Path, default=METADATA / 'DOM_DOM.tfw')
    parser.add_argument('--crs-file', type=Path, default=METADATA / 'DOM_DOM.prj')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--max-pixels', type=int, default=2048)
    args = parser.parse_args()
    if not 256 <= args.max_pixels <= 4096:
        parser.error('--max-pixels must be between 256 and 4096')
    data = case_data(CASE, RUNS)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.map_only or not args.image.exists():
        reason = 'Explicit map-only export' if args.map_only else 'Orthophoto file has not arrived'
        data['image'] = dict(available=False, url=None, reason=reason)
    else:
        data['image'] = export_image(args.image, args.world_file, args.crs_file, data['extent'],
                                     args.output / 'orthophoto.jpg', args.max_pixels)
    target = args.output / 'data.js'
    temporary = target.with_suffix('.js.tmp')
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    temporary.write_text('window.JIANCHANG_DATA = ' + payload + ';\n', encoding='utf-8')
    temporary.replace(target)
    print(json.dumps(dict(output=str(target), runs=len(data['runs']), features=len(data['features']),
                          image=data['image']), ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Export failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
