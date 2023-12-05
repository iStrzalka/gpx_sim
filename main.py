from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from utilities import test_against_location, convert_xll_to_latlon
import folium
import os

import pandas as pd
from math import floor
import gpxpy

import matplotlib.pyplot as plt
import numpy as np
import io, base64

from folium.plugins import BeautifyIcon

from utilities import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_data'


db_name = 'gpx_viewer'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_name}.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


DATASET = 'map_data/'
GPXDATA = 'gpx_data/converted/'


class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    email = db.Column(db.String(100))
    password = db.Column(db.String(100))

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password


class Track(db.Model):
    __tablename__ = 'track'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    file_name = db.Column(db.String(100))
    file_path = db.Column(db.String(100))
    converted_file_path = db.Column(db.String(100))
    coverted = db.Column(db.Boolean, default=False)

    def __init__(self, user_id, file_name, file_path, converted_file_path):
        self.user_id = user_id
        self.file_name = file_name
        self.file_path = file_path
        self.converted_file_path = converted_file_path
        self.converted = False


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    print(request.method)
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        return redirect(url_for(''))
    else:
        return render_template('upload.html')


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

def get_plot_and_breakpoints(filename, a = None, b = None):
    points = parse_gpx_to_points(filename)
    print(points[0])
    if a:
        points = points[a:b]
    x = [0]
    for i in range(1, len(points)):
        point1 = (points[i - 1][0][1], points[i - 1][0][0])
        point2 = (points[i][0][1], points[i][0][0])
        x.append(calculate_distance(point1, point2))
    x = np.cumsum(x)
    x = [round(ele, 2) for ele in x]
    print(x)
    plot, ids_change = plot_directional_change([(i, ele) for i, (_, ele) in enumerate(points)], 0.015, x)

    return (SUCCESS, (plot, ids_change))


@app.route('/myroutes', methods=['GET'])
def my_routes():
    return redirect(url_for('index'))

@app.route('/peek', defaults={'filename': None}, methods=['GET'])
@app.route('/peek/<filename>', methods=['GET'])
def peek(filename = None):
    if filename:
        df, points = process_gpx_to_df('gpx_data/original/' + filename)
        
        mymap = folium.Map(location=[ df.Latitude.mean(), df.Longitude.mean() ], zoom_start=15)
        folium.TileLayer('http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg', attr='terrain-bcg', name='Terrain Map').add_to(mymap)

        html_start = "Start of the track {}".format("Lemko 2018")
        start_popup = folium.Popup(html_start, max_width=400)
        end_popup = folium.Popup("End of the track {}".format("Lemko 2018"), max_width=400)
        
        folium.Marker(location=[points[0][0], points[0][1]], popup=start_popup, tooltip='Start', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:green;opacity:0.5"></div>')).add_to(mymap)
        folium.Marker(location=[points[-1][0], points[-1][1]], popup=end_popup, tooltip='End', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:red;opacity:0.5"></div>')).add_to(mymap)
        
        folium.PolyLine(points, color="blue", weight=2.5, opacity=1).add_to(mymap)
        
        data = database_covers_map(DATASET)
        for (lat, lon), (lat2, lon2) in data:
            folium.Rectangle(bounds=[(lat, lon), (lat2, lon2)], color='blue', fill=True, fill_opacity=0.2).add_to(mymap)

        map_html = mymap._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')

        filenames = list(sorted(os.listdir('gpx_data/original')))
        return render_template('peek.html', map=map_html, filenames=filenames)
    else:
        mymap = folium.Map(location=[ 51.9537505,19.1343786 ], zoom_start=7)
        folium.TileLayer('http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg', attr='terrain-bcg', name='Terrain Map').add_to(mymap)
        
        data = database_covers_map(DATASET)
        for (lat, lon), (lat2, lon2) in data:
            folium.Rectangle(bounds=[(lat, lon), (lat2, lon2)], color='blue', fill=True, fill_opacity=0.2).add_to(mymap)

        map_html = mymap._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')
        filenames = list(sorted(os.listdir('gpx_data/original')))
        return render_template('peek.html', map=map_html, filenames=filenames)


@app.route('/', defaults={'filename': None, 'a' : None, 'b' : None}, methods=['GET'])
@app.route('/<filename>', defaults={'a' : None, 'b' : None}, methods=['GET'])
@app.route('/<filename>/<a>/<b>', methods=['GET'])
def index(filename, a, b):
    if filename:
        df, points = process_gpx_to_df(GPXDATA + filename)
        
        if not a:
            a = 0
            b = len(points) - 1
        else:
            a = int(a)
            b = int(b)

        mymap = folium.Map(location=[ df[a:b].Latitude.mean(), df[a:b].Longitude.mean() ], zoom_start=15)
        folium.TileLayer('http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg', attr='terrain-bcg', name='Terrain Map').add_to(mymap)

        html_start = "Start of the track {}".format("Lemko 2018")
        start_popup = folium.Popup(html_start, max_width=400)
        end_popup = folium.Popup("End of the track {}".format("Lemko 2018"), max_width=400)
        

        folium.Marker(location=points[a], popup=start_popup, tooltip='Start', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:green;opacity:0.5"></div>')).add_to(mymap)
        folium.Marker(location=points[b], popup=end_popup, tooltip='End', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:red;opacity:0.5"></div>')).add_to(mymap)

        # data = database_covers_map(DATASET)
        # for (lat, lon), (lat2, lon2) in data:
        #     folium.Rectangle(bounds=[(lat, lon), (lat2, lon2)], color='blue', fill=True, fill_opacity=0.2).add_to(mymap)

        x = [0]
        for i in range(1, len(points)):
            x.append(calculate_distance(points[i - 1], points[i]))
        x = np.cumsum(x)
        print(x[-1], 'km')
        print(len(points))

        # map_html = mymap._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')
        filenames = list(sorted(os.listdir(GPXDATA)))
        elevation_points = parse_gpx_to_points(GPXDATA + filename)
        elevation_points = elevation_points[a:b]
        result, data = get_plot_and_breakpoints(GPXDATA + filename, a, b)
        ele = [(i, ele) for i, (_, ele) in enumerate(elevation_points)]
        
        # cutoff_x = [x for x, y in cut_data_to_threshold(ele, 0.2)]
        # for x in cutoff_x:
        #     icon_circle = BeautifyIcon(
        #         icon_shape='circle-dot', 
        #         border_color='green', 
        #         border_width=10,
        #     )
        #     folium.Marker(location=points[x], tooltip='circle', icon=icon_circle).add_to(mymap)
        #     # popup = folium.Popup("Elevation change {}".format(x), max_width=400)
        #     # folium.Marker(location=points[x], popup=popup, tooltip=f'Point {x}', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:10px;height:10px;border-radius:5px;background:blue;opacity:0.75"></div>')).add_to(mymap)
        
        g2 = table(ele, data[1])
        
        

        # diff = 0
        # DIFFERENCE = 0.5 # 0.1km
        # for i in range(a + 1, b + 1):
        #     diff += x[i] - x[i - 1]
        #     if diff > DIFFERENCE:
        #         icon_circle = BeautifyIcon(
        #             icon_shape='circle-dot', 
        #             border_color='blue', 
        #             border_width=6,
        #             inner_icon_style='opacity:0.3'
        #         )
        #         folium.Marker(location=points[i], tooltip=f'point {round(x[i] - x[a], 2)}km', icon=icon_circle).add_to(mymap)
        #         diff -= DIFFERENCE
                # popup = folium.Popup("Elevation change {}".format(i), max_width=400)
                # folium.Marker(location=points[i], popup=popup, tooltip=f'Point {i}', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:10px;height:10px;border-radius:5px;background:blue;opacity:0.75"></div>')).add_to(mymap)
        
        for index in data[1]:
            icon_circle = BeautifyIcon(
                icon_shape='circle-dot', 
                border_color='blue', 
                border_width=6,
                inner_icon_style='opacity:0.3'
            )
            folium.Marker(location=points[index + a], tooltip=f'Point {round(x[index], 2)}km', icon=icon_circle).add_to(mymap)


        g2['start_id'] = g2['start']
        g2['end_id'] = g2['end']
        g2['start'] = g2['start'].apply(lambda a: str(round(x[a], 2)) + ' km')
        g2['end'] = g2['end'].apply(lambda a: str(round(x[a], 2)) + ' km')
        g2['elevation_gain'] = g2['elevation_gain'].apply(lambda a: str(round(a, 2)) + ' m')
        g2['elevation_loss'] = g2['elevation_loss'].apply(lambda a: str(round(a, 2)) + ' m')

        # dataframe to list
        tab = g2.to_numpy().tolist()

        green = True
        last_i = 0
        points = points[a:b]
        for i in data[1] + [len(ele)]:
            if green:
                folium.PolyLine(points[last_i:i + 1], color="green", weight=2.5, opacity=1).add_to(mymap)
            else:
                folium.PolyLine(points[last_i:i + 1], color="red", weight=2.5, opacity=1).add_to(mymap)
            green = not green
            last_i = i

        # folium.PolyLine(locations=[(52.23502,21.18470), (52.229255, 21.186420)], color="black", opacity=1).add_to(mymap)

        # def calculate_line_slope(line):
        #     (x1, y1), (x2, y2) = line
        #     return (y2 - y1) / (x2 - x1)

        # def calculate_line_intercept(line):
        #     (x1, y1), (x2, y2) = line
        #     return y1 - calculate_line_slope(line) * x1

        # def calculate_line_y(line, x):
        #     return calculate_line_slope(line) * x + calculate_line_intercept(line)

        # line = [(52.23502,21.18470), (52.229255, 21.186420)]
        # converted_line = [convert_latlon_to_xll(lat, lon) for (lat, lon) in line]

        # tab_ = []
        # for x in range(converted_line[0][0], converted_line[1][0] + 1):
        #     y = int(calculate_line_y(converted_line, x))
        #     (lat, lon) = convert_xll_to_latlon(x, y)
        #     folium.CircleMarker(location=[lat, lon], radius=2, color='black', fill=True, fill_color='black', fill_opacity=1).add_to(mymap)

        # print(points[:5])
        # converted_file = [convert_latlon_to_xll(lat, lon) for (lon, lat) in points]
        # converted_file = [convert_xll_to_latlon(x, y) for x, y in converted_file]
        # for (x, y) in converted_file:
        #     folium.CircleMarker(location=[y, x], radius=2, color='black', fill=True, fill_color='black', fill_opacity=1).add_to(mymap)

        # Compare segment /1.gpx to current segment
        _, segment_points = process_gpx_to_df(GPXDATA + '/1.gpx') 
        print(segment_points[2090:2332])
        print(convert_latlon_to_xll(segment_points[2090][0], segment_points[2090][1]))

        llist = find_points_for_segment([convert_latlon_to_xll(lat, lon) for (lat, lon) in segment_points[2090:2332]], points)
        print(len(llist))

        for (x, y) in segment_points[2090:2332]:
            folium.CircleMarker(location=[x, y], radius=2, color='black', fill=True, fill_color='black', fill_opacity=1).add_to(mymap)

        for i, track in enumerate(llist):
            track_points = [points[i] for i in track]
            track_color = colors[2 + i]
            folium.PolyLine(track_points, color=track_color, weight=2.5, opacity=1).add_to(mymap)

        map_html = mymap._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')
        # folium.PolyLine(points, color="blue", weight=2.5, opacity=1).add_to(mymap)

        # print(g2.to_html())
        # table_html = g2.to_html()
        # table_html = table_html.replace('$', '</a>')
        # table_html = table_html.replace('#', '>')
        # table_html = table_html.replace('a href=', '<a href=')
        # print(table_html)

        if result is SUCCESS:
            return render_template('index.html', graph1=data[0], table=tab, map=map_html, filename=filename, filenames=filenames, a=a)
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
