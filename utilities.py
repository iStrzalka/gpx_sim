# https://pl.wikipedia.org/wiki/Układ_współrzędnych_2000
# EPSG_CODE = 2178 # Warsaw
# https://pl.wikipedia.org/wiki/Układ_współrzędnych_1992
EPSG_CODE = 2180 # Lemko

MIN_POL = 2176
MAX_POL = 2180

# https://latitudelongitude.org/pl/
MIN_LAT = 49.29899
MAX_LAT = 54.79086
MIN_LON = 14.24712
MAX_LON = 23.89251

from pyproj import Proj
from math import pi, cos, floor
import os, re
from gpxpy import parse

R_EARTH = 6378137

FAILURE = 0
SUCCESS = 1


def test_against_location(xll, yll):
    for i in range(MIN_POL, MAX_POL + 1):
        lat, lon = convert_xll_to_latlon(xll, yll, i)
        if lat > MIN_LAT and lat < MAX_LAT and lon > MIN_LON and lon < MAX_LON:
            return i
    return None


def convert_xll_to_latlon(xll, yll, epsg_code = EPSG_CODE):
    projection = Proj(f'epsg:{epsg_code}')
    lon, lat = projection(xll, yll, inverse=True)
    return lat, lon


def convert_latlon_to_xll(lat, lon):
    projection = Proj(f'epsg:{EPSG_CODE}')
    xll, yll = projection(lon, lat)
    return xll, yll


def add_to_lattitude(lat, lon, dy, dx):
    new_lat = lat + (dy / R_EARTH) * (180 / pi)
    new_lon = lon + (dx / R_EARTH) * (180 / pi) / cos(lat * pi/180)

    return new_lat, new_lon


def database_covers(database_path):
    ret = []

    for file_name in os.listdir(database_path):
        if file_name.endswith('.asc'):
            with open(database_path + file_name, 'r') as input:
                lines = [input.readline().strip() for _ in range(4)]
                ncols = int(lines[0].split(' ')[-1])
                nrows = int(lines[1].split(' ')[-1])
                xllcenter = int(floor(float(lines[2].split(' ')[-1])))
                yllcenter = int(floor(float(lines[3].split(' ')[-1])))

                ret.append(((xllcenter, yllcenter), (xllcenter + ncols - 1, yllcenter + nrows - 1), database_path + file_name))

    return ret


def get_data(file_name):
    data = []
    with open(file_name, 'r') as input:    
        lines = input.readlines()
        lines = [line.strip() for line in lines]

        data = lines[6:]
        data = [re.split('\s+', line) for line in data]
        data = [[float(x) for x in line] for line in data]

    return data


def parse_gpx_to_points(file_name):
    gpx = parse(open(file_name, 'r'))

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(((point.longitude, point.latitude), point.elevation))

    return points


def create_new_gpx(file_name, new_elevations):
    gpx = parse(open(file_name, 'r'))

    for track in gpx.tracks:
        for segment in track.segments:
            for point, elevation in zip(segment.points, new_elevations):
                point.elevation = elevation

    with open('out.gpx', 'w') as output:
        output.write(gpx.to_xml())

cur_node = ((-1, -1), (-1, -1), [])

def find_new_node(dataset, x, y):
    for node in dataset:
        (x1, y1), (x2, y2), _ = node
        if x1 <= x <= x2 and y1 <= y <= y2:
            data = get_data(node[2])
            return (SUCCESS, ((x1, y1), (x2, y2), data))
    return (FAILURE, f"Node for ({x}, {y}) not found in dataset.")


def find_elevation(dataset, lat, lon):
    x, y = convert_latlon_to_xll(lat, lon)
    x, y = int(round(x, 1)), int(round(y, 1))

    global cur_node
    try:
        (x1, y1), (x2, y2), data = cur_node
    except:
        (result, cur_node) = find_new_node(dataset, x, y)
        if result == FAILURE:
            return (FAILURE, cur_node)
        (x1, y1), (x2, y2), data = cur_node
    if x1 <= x <= x2 and y1 <= y <= y2:
        return (SUCCESS, data[y2 - y][x - x1])
    else:
        (result, cur_node) = find_new_node(dataset, x, y)
        if result == FAILURE:
            return (FAILURE, cur_node)
        (x1, _), (_, y2), data = cur_node
        return (SUCCESS, data[y2 - y][x - x1])
    

def get_elevation_for_points(dataset, points):
    ret = []
    for (lon, lat), _ in points:
        (result, elevation) = find_elevation(dataset, lat, lon)
        if result == FAILURE:
            return (FAILURE, (lat, lon))
        ret.append(elevation)
    return (SUCCESS, ret)