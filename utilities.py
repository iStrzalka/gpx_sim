# https://pl.wikipedia.org/wiki/Układ_współrzędnych_2000
# EPSG_CODE = 2178 # Warsaw
# https://pl.wikipedia.org/wiki/Układ_współrzędnych_1992
# EPSG_CODE = 2180 # Lemko
MIN_EPSG = 2176
MAX_EPSG = 2180


MIN_POL = 2176
MAX_POL = 2180

# https://latitudelongitude.org/pl/
MIN_LAT = 49.29899
MAX_LAT = 54.79086
MIN_LON = 14.24712
MAX_LON = 23.89251

from pyproj import Proj
from math import pi, cos, floor, sqrt
import os, re
from gpxpy import parse
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
import numpy as np
import haversine

R_EARTH = 6378137

FAILURE = 0
SUCCESS = 1

converters = {
    2176: Proj(f'epsg:2176'),
    2177: Proj(f'epsg:2177'),
    2178: Proj(f'epsg:2178'),
    2179: Proj(f'epsg:2179'),
    2180: Proj(f'epsg:2180'),
}


# Testing xll, yll against all EPSG codes in Poland (2176 - 2180)
def test_against_location(xll, yll):
    for i in range(MIN_POL, MAX_POL + 1):
        lat, lon = convert_xll_to_latlon(xll, yll, i)
        if lat > MIN_LAT and lat < MAX_LAT and lon > MIN_LON and lon < MAX_LON:
            return i
    return None


# Converts xll, yll values to lattitude and longitude values using EPSG code
def convert_xll_to_latlon(xll, yll, epsg_code = 2178):
    lon, lat = converters[epsg_code](xll, yll, inverse=True)
    return lat, lon


# Converts lattitude and longitude values to xll, yll values using EPSG code
def convert_latlon_to_xll(lat, lon, epsg_code = 2178):
    xll, yll = converters[epsg_code](lon, lat)
    return int(xll), int(yll)


# Adds dy and dx to lattitude and longitude values and returns new values
def add_to_lattitude(lat, lon, dy, dx):
    new_lat = lat + (dy / R_EARTH) * (180 / pi)
    new_lon = lon + (dx / R_EARTH) * (180 / pi) / cos(lat * pi/180)

    return new_lat, new_lon


# Given database path, returns what the database covers in xll, yll values
# with the path to the file as well as its EPSG code [TODO] 
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

                epsg_code = test_against_location(xllcenter, yllcenter)

                ret.append(((xllcenter, yllcenter), (xllcenter + ncols - 1, yllcenter + nrows - 1), database_path + file_name))

    return ret


# Given path to file, returns its estimated elevation data
def get_data(file_name):
    data = []
    with open(file_name, 'r') as input:    
        lines = input.readlines()
        lines = [line.strip() for line in lines]

        data = lines[6:]
        data = [re.split('\s+', line) for line in data]
        data = [[float(x) for x in line] for line in data]

    return data


# Given path to gpx file, return list of points in format ((lon, lat), elevation)
def parse_gpx_to_points(file_name):
    gpx = parse(open(file_name, 'r'))

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(((point.longitude, point.latitude), point.elevation))

    return points


# Given path to gpx file as well as new elevations, creates new gpx file with new elevations.
def create_new_gpx(file_name, new_elevations, new_file_name):
    gpx = parse(open(file_name, 'r'))

    for track in gpx.tracks:
        for segment in track.segments:
            for point, elevation in zip(segment.points, new_elevations):
                point.elevation = elevation

    with open(new_file_name, 'w') as output:
        output.write(gpx.to_xml())

cur_node = ((-1, -1), (-1, -1), [])


# Given dataset and xll, yll values finds new database file that contains those values
# returns it in format ((xll1, yll1), (xll2, yll2), data)
def find_new_node(dataset, x, y):
    for node in dataset:
        (x1, y1), (x2, y2), _ = node
        for i in range(MIN_EPSG, MAX_EPSG + 1):
            xll, yll = convert_latlon_to_xll(x, y, i)
            if x1 <= xll <= x2 and y1 <= yll <= y2:
                data = get_data(node[2])
                return (SUCCESS, ((x1, y1), (x2, y2), data))
    return (FAILURE, f"Node for ({x}, {y}) not found in dataset.")


# Given dataset and lattitude and longitude values, returns elevation for those values
def find_elevation(dataset, lat, lon):
    x, y = convert_latlon_to_xll(lat, lon)
    x, y = int(round(x, 1)), int(round(y, 1))

    global cur_node
    try:
        (x1, y1), (x2, y2), data = cur_node
    except:
        (result, cur_node) = find_new_node(dataset, lat, lon)
        if result == FAILURE:
            return (FAILURE, cur_node)
        (x1, y1), (x2, y2), data = cur_node
    
    if x1 <= x <= x2 and y1 <= y <= y2:
        return (SUCCESS, data[y2 - y][x - x1])
    else:
        (result, cur_node) = find_new_node(dataset, lat, lon)
        if result == FAILURE:
            return (FAILURE, cur_node)
        (x1, _), (_, y2), data = cur_node
        return (SUCCESS, data[y2 - y][x - x1])
    

# Given dataset and list of points, returns list of elevations for those points
def get_elevation_for_points(dataset, points):
    ret = []
    for (lon, lat), _ in points:
        (result, elevation) = find_elevation(dataset, lat, lon)
        if result == FAILURE:
            return (FAILURE, (lat, lon))
        ret.append(elevation)
    return (SUCCESS, ret)


# Given dataset and list of points, cut elevation change to threshold
def cut_data_to_threshold(data, threshold):
    result = [data[0]]
    temp = 0
    for i in range(1, len(data)):
        temp += data[i][1] - data[i - 1][1]
        if temp > threshold or temp < -threshold:
            result.append(data[i])
            temp = 0
    result.append(data[-1])

    return result


# Given data find directional change based on trading algorithm
def directional_change(data, d=0.015):     
    p = pd.DataFrame({
    "Price": data
    })
    p["Event"] = ''
    run = "upward" # initial run
    ph = p['Price'][0] # highest price
    pl = ph # lowest price
    pl_i = ph_i = 0

    for t in range(0, len(p)):
        pt = p["Price"][t]
        if run == "downward":
            if pt < pl:
                pl = pt
                pl_i = t
            if pt >= pl * (1 + d):
                p.at[pl_i, 'Event'] = "start upturn event"
                run = "upward"
                ph = pt
                ph_i = t
                # print(">> {} - Upward! : {}%, value {}".format(pl_i, round((pt - pl)/pl, 2), round(pt - pl,2)))
        elif run == "upward":
            if pt > ph:
                ph = pt
                ph_i = t
            if pt <= ph * (1 - d):
                p.at[ph_i, 'Event'] = "start downturn event"
                run = "downward"
                pl = pt
                pl_i = t
                # print(">> {} - Downward! : {}%, value {}".format(ph_i, round((ph - pt)/ph, 2), round(ph - pt,2)))
    
    ids_change = p[p['Event'] != ''].index.tolist()

    return p, ids_change


# Perform cutoff on data and plot directional change and return as base64
def plot_directional_change(data, d=0.015, distances = [], cutoff = 0.2):
    original_data = data
    if cutoff != 0:
        data = cut_data_to_threshold(data, cutoff)
    _, ids_change = directional_change([ele for _, ele in data], d)

    plt.figure(figsize=(16, 8))
    
    # print(original_data[:5])
    # distances = [0]
    # for i in range(1, len(original_data)):
    #     distances.append(calculate_distance(original_data[i - 1], original_data[i]))
    # distances = np.cumsum(distances)

    last_id = 0
    green = True
    for i in ids_change:
        X, Y = distances[last_id:data[i][0] + 1], [y for _, y in original_data[last_id:data[i][0] + 1]]
        if green:
            plt.plot(X, Y, color='green')
        else:
            plt.plot(X, Y, color='red')
        green = not green
        last_id = data[i][0]
    X, Y = distances[last_id:], [y for _, y in original_data[last_id:]]
    if green:
        plt.plot(X, Y, color='green')
    else:
        plt.plot(X, Y, color='red')
    
    plt.xticks(np.arange(0, max(distances), 0.5))

    iobytes = io.BytesIO()
    plt.savefig(iobytes, format='png')
    iobytes.seek(0)
    graph = base64.b64encode(iobytes.read()).decode('utf-8')
    plt.close()

    return graph, [data[i][0] for i in ids_change]


# Calculate distance between two points
def calculate_distance(point1, point2):
    return haversine.haversine(point1, point2)


# Given original data, ids of change, and cutoff, return table of elevation gain and loss
def table(original_data, ids_change, cutoff = 0.2):
    cut_data = cut_data_to_threshold(original_data, cutoff)
    y = [0 for i in range(len(original_data))]
    for i in range(1, len(cut_data)):
        y[cut_data[i][0]] = cut_data[i][1] - cut_data[i - 1][1]
    
    y = np.array(y)
    y_up = np.cumsum((y >= 0) * y)
    y_down = np.cumsum((y < 0) * y)

    llist = []
    last_id = 0
    for i in ids_change:
        llist.append([last_id, i, y_up[i] - y_up[last_id], y_down[i] - y_down[last_id]])
        last_id = i
    llist.append([last_id, len(y), y_up[-1] - y_up[last_id], y_down[-1] - y_down[last_id]])

    return pd.DataFrame(llist, columns=['start', 'end', 'elevation_gain', 'elevation_loss'])


def convert(dataset, gpx_file, converted_file):
    df = database_covers(dataset)
    points = parse_gpx_to_points(gpx_file)
    result, elevation_map = get_elevation_for_points(df, points)

    if result is FAILURE:
        return FAILURE, elevation_map

    create_new_gpx(gpx_file, elevation_map, converted_file)


# Given segment in (xll, yll) format and list of points in (lat, lon) format
# returns list of traces that match the segment with at least [PERCENTAGE]% accuracy
def find_points_for_segment(segment, points):
    HOW_MANY = 10
    DISTANCE_THRESHOLD_METERS = 7
    INDEX_THRESHOLD = 3
    PERCENTAGE = 0.95

    def distance_point(point, point2):
        (x1, y1), (x2, y2) = point, point2
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

    converted = [convert_latlon_to_xll(lat, lon) for (lat, lon) in points]

    index_tab = []
    for point in segment:
        distances = [(distance_point(point, point2), i) for i, point2 in enumerate(converted)]
        distances.sort()
        # cut to 3meters distance
        distances = [(d, i) for d, i in distances if d < DISTANCE_THRESHOLD_METERS]
        indexes = [i for _, i in distances[:HOW_MANY]]
        indexes.sort()
        index_tab.append(indexes)

    do_not_continue = [[False for _ in range(HOW_MANY)] for _ in range(len(index_tab))]

    def dfs(last, i, dnc):
        if i == len(index_tab):
            return (True, [])
        for j in range(len(index_tab[i])):
            if do_not_continue[i][j]:
                continue
            # print(index_tab[i][j], last)
            if 0 <= index_tab[i][j] - last and index_tab[i][j] - last <= INDEX_THRESHOLD:
                result, data = dfs(index_tab[i][j], i + 1, (i, j))
                if result:
                    return (True, data + [(i, j)]) 
        

        do_not_continue[dnc[0]][dnc[1]] = True
        return (False, i)

    found_traces = []
    for i in range(len(index_tab[0])):
        result, data = dfs(index_tab[0][i], 1, (0, i))
        if result:
            data = [(0, i)] + data[::-1]
            data = [index_tab[i][j] for i, j in data]
            found_traces.append(data)
            #  print(result, data)
        else:
            # print(result, data)
            pass

    # get unique traces that are unique in 95% of points
    unique_traces = []
    for trace in found_traces:
        for trace2 in unique_traces:
            percentage = sum(trace[i] == trace2[i] for i in range(len(trace))) / len(trace)
            if percentage > PERCENTAGE:
                break
        else:
            unique_traces.append(trace)

    return unique_traces