from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from utilities import *
import folium
import os
from io import BytesIO
import base64
from subprocess import Popen, PIPE
import time
import re

import haversine
import pandas as pd
from geopy import distance
from math import sqrt
import gpxpy

app = Flask(__name__)

def process_gpx_to_df(file_name : str) -> tuple[pd.DataFrame, list]:
    gpx = gpxpy.parse(open(file_name)) 
    
    # Make DataFrame
    track = gpx.tracks[0]
    segment = track.segments[0]

    # Load the data into a Pandas dataframe (by way of a list)
    data = []
    segment_length = segment.length_3d()
    for point_idx, point in enumerate(segment.points):
        data.append([point.longitude, point.latitude,point.elevation, point.time, segment.get_speed(point_idx)])
    columns = ['Longitude', 'Latitude', 'Altitude', 'Time', 'Speed']
    gpx_df = pd.DataFrame(data, columns=columns)
    
    points = []
    for track in gpx.tracks:
        for segment in track.segments: 
            for point in segment.points:
                points.append(tuple([point.latitude, point.longitude]))
    
    return (gpx_df, points)

colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

def convert_to_latlon(xllcenter, yllcenter, epsg_code = 2180):
    import pyproj

    # Define the projection using the EPSG code
    # https://pl.wikipedia.org/wiki/Układ_współrzędnych_2000
    projection = pyproj.Proj(f'epsg:{epsg_code}')

    # Calculate the latitude and longitude of the lower-left corner
    lon, lat = projection(xllcenter, yllcenter, inverse=True)

    return lat, lon

def add_to_lattitude(lat, lon, dy, dx):
    r_earth = 6378137
    from math import pi, cos

    new_lat = lat + (dy / r_earth) * (180 / pi)
    new_lon = lon + (dx / r_earth) * (180 / pi) / cos(lat * pi/180)

    return new_lat, new_lon

def database_covers():
    ret = []

    for file_name in os.listdir('map_data'):
        if file_name.endswith('.asc'):
            with open('map_data/' + file_name, 'r') as input:
                lines = input.readlines()
                lines = [line.strip() for line in lines]
                ncols = int(lines[0].split(' ')[-1])
                nrows = int(lines[1].split(' ')[-1])
                xllcenter = float(lines[2].split(' ')[-1])
                yllcenter = float(lines[3].split(' ')[-1])

                lat, lon = convert_to_latlon(xllcenter, yllcenter)
                lat2, lon2 = add_to_lattitude(lat, lon, nrows, ncols)
                
                # data = lines[6:]
                # data = [line.split(' ') for line in data]
                # data = [[float(x) for x in line] for line in data]

                ret.append(((lat, lon), (lat2, lon2), (xllcenter, yllcenter), (xllcenter + nrows - 1, yllcenter + ncols - 1), 1))
    
    return ret


@app.route('/')
def index():
    df, points = process_gpx_to_df('gpx_data/lemko.gpx')
    
    colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    mymap = folium.Map(location=[ df.Latitude.mean(), df.Longitude.mean() ], zoom_start=10)
    # folium.TileLayer('openstreetmap', name='OpenStreet Map').add_to(mymap)
    # folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}', attr='Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC', name='Nat Geo Map').add_to(mymap)
    folium.TileLayer('http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg', attr='terrain-bcg', name='Terrain Map').add_to(mymap)

    html_start = "Start of the track {}".format("Lemko 2018")
    start_popup = folium.Popup(html_start, max_width=400)
    end_popup = folium.Popup("End of the track {}".format("Lemko 2018"), max_width=400)
    
    folium.Marker(location=[points[0][0], points[0][1]], popup=start_popup, tooltip='Start', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:green;opacity:0.5"></div>')).add_to(mymap)
    folium.Marker(location=[points[-1][0], points[-1][1]], popup=end_popup, tooltip='End', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:red;opacity:0.5"></div>')).add_to(mymap)

    folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(mymap)

    data = database_covers()
    for (lat, lon), (lat2, lon2), _, _, _ in data:
        folium.Rectangle(bounds=[(lat, lon), (lat2, lon2)], color='blue', fill=True, fill_opacity=0.2).add_to(mymap)

    map_html = mymap._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')
    return render_template('index.html', map=map_html)

if __name__ == '__main__':
    app.run(debug=True)
