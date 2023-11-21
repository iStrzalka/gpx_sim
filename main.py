from flask import Flask, render_template, request, redirect, url_for, session, flash
from utilities import test_against_location, convert_xll_to_latlon
import folium
import os

import pandas as pd
from math import floor
import gpxpy

import matplotlib.pyplot as plt
import numpy as np
import io, base64

from utilities import *

app = Flask(__name__)

DATASET = 'map_data/temp/'
GPXDATA = 'gpx_data/'

def process_gpx_to_df(file_name : str) -> tuple[pd.DataFrame, list]:
    gpx = gpxpy.parse(open(file_name)) 
    
    # Make DataFrame
    track = gpx.tracks[0]
    segment = track.segments[0]

    data, points = [], []
    for point_idx, point in enumerate(segment.points):
        data.append([point.longitude, point.latitude,point.elevation, point.time, segment.get_speed(point_idx)])
        points.append(tuple([point.latitude, point.longitude]))
    columns = ['Longitude', 'Latitude', 'Altitude', 'Time', 'Speed']
    gpx_df = pd.DataFrame(data, columns=columns)    
    
    return (gpx_df, points)

colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']


def add_to_lattitude(lat, lon, dy, dx):
    r_earth = 6378137
    from math import pi, cos

    new_lat = lat + (dy / r_earth) * (180 / pi)
    new_lon = lon + (dx / r_earth) * (180 / pi) / cos(lat * pi/180)

    return new_lat, new_lon


def convert_to_latlon(xll, yll):
    code = test_against_location(xll, yll)
    return convert_xll_to_latlon(xll, yll, code)


def database_covers_map(database_path):
    ret = []

    for file_name in os.listdir(database_path):
        if file_name.endswith('.asc'):
            with open(database_path + '/' + file_name, 'r') as input:
                lines = [input.readline().strip() for _ in range(4)]
                ncols = int(lines[0].split(' ')[-1])
                nrows = int(lines[1].split(' ')[-1])
                xllcenter = int(floor(float(lines[2].split(' ')[-1])))
                yllcenter = int(floor(float(lines[3].split(' ')[-1])))

                lat, lon = convert_to_latlon(xllcenter, yllcenter)
                lat2, lon2 = add_to_lattitude(lat, lon, nrows, ncols)

                ret.append(((lat, lon), (lat2, lon2)))
    
    return ret

def get_plot_for(filename):
    dataset = database_covers(DATASET)
    points = parse_gpx_to_points(filename)
    result, elevation_map = get_elevation_for_points(dataset, points)
    if result is FAILURE:
        return (result, elevation_map)
    elevation_map = np.array(elevation_map)
    elevation_gpx = np.array([ele for _, ele in points])

    # try to fill missing values with average of positive values to the left and right
    for i in range(len(elevation_map)):
        if elevation_map[i] < 0:
            left_i = i - 1
            while left_i >= 0 and elevation_map[left_i] < 0:
                left_i -= 1
            right_i = i + 1
            while right_i < len(elevation_map) and elevation_map[right_i] < 0:
                right_i += 1
            elevation_map[i] = (elevation_map[left_i] + elevation_map[right_i]) / 2

    plt.plot(elevation_gpx, label='gpx')
    plt.plot(elevation_map, label='map')
    plt.legend()

    iobytes = io.BytesIO()
    plt.savefig(iobytes, format='png')
    iobytes.seek(0)
    graph1 = base64.b64encode(iobytes.read()).decode('utf-8')

    plt.close()
    plt.plot(elevation_gpx - elevation_map, label='difference')
    
    iobytes = io.BytesIO()
    plt.savefig(iobytes, format='png')
    iobytes.seek(0)
    graph2 = base64.b64encode(iobytes.read()).decode('utf-8')
    plt.close()

    return (SUCCESS, (graph1, graph2))


@app.route('/myroutes', methods=['GET'])
def my_routes():
    return redirect(url_for('index'))


@app.route('/', defaults={'filename': None}, methods=['GET'])
@app.route('/<filename>', methods=['GET'])
def index(filename):
    if filename:
        df, points = process_gpx_to_df(GPXDATA + filename)
        
        mymap = folium.Map(location=[ df.Latitude.mean(), df.Longitude.mean() ], zoom_start=15)
        folium.TileLayer('http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg', attr='terrain-bcg', name='Terrain Map').add_to(mymap)

        html_start = "Start of the track {}".format("Lemko 2018")
        start_popup = folium.Popup(html_start, max_width=400)
        end_popup = folium.Popup("End of the track {}".format("Lemko 2018"), max_width=400)
        
        folium.Marker(location=[points[0][0], points[0][1]], popup=start_popup, tooltip='Start', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:green;opacity:0.5"></div>')).add_to(mymap)
        folium.Marker(location=[points[-1][0], points[-1][1]], popup=end_popup, tooltip='End', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:red;opacity:0.5"></div>')).add_to(mymap)

        folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(mymap)

        data = database_covers_map(DATASET)
        for (lat, lon), (lat2, lon2) in data:
            folium.Rectangle(bounds=[(lat, lon), (lat2, lon2)], color='blue', fill=True, fill_opacity=0.2).add_to(mymap)

        map_html = mymap._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')
        filenames = list(sorted(os.listdir(GPXDATA)))
        result, data = get_plot_for(GPXDATA + filename)
        print(result)
        if result is SUCCESS:
            return render_template('index.html', graph1=data[0], graph2=data[1], map=map_html, filenames=filenames)
        else:
            return render_template('index.html', map=map_html, filenames=filenames)
    else:
        mymap = folium.Map(location=[ 51.9537505,19.1343786 ], zoom_start=7)
        folium.TileLayer('http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg', attr='terrain-bcg', name='Terrain Map').add_to(mymap)
        
        data = database_covers_map(DATASET)
        for (lat, lon), (lat2, lon2) in data:
            folium.Rectangle(bounds=[(lat, lon), (lat2, lon2)], color='blue', fill=True, fill_opacity=0.2).add_to(mymap)

        map_html = mymap._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')
        filenames = list(sorted(os.listdir(GPXDATA)))
        return render_template('index.html', map=map_html, filenames=filenames)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
