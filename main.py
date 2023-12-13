from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import folium
import os

import pandas as pd
from math import floor, ceil
import gpxpy

import numpy as np
from time import time

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from folium.plugins import BeautifyIcon

from common.utilities import *

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER')
app.config['DATABASE_FOLDER'] = os.environ.get('DATABASE_FOLDER')
app.config['ORIGINAL_GPX_FOLDER'] = os.environ.get('ORIGINAL_GPX_FOLDER')
app.config['CONVERTED_GPX_FOLDER'] = os.environ.get('CONVERTED_GPX_FOLDER')

db_name = os.environ.get('DB_NAME')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.environ.get('SQLALCHEMY_TRACK_MODIFICATIONS')

colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

DATASET = 'map_data/'
GPXDATA = 'gpx_data/converted/'


def is_user_admin():
    return True


def read_database_file(filename):
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as f:
            lines = [f.readline().strip() for _ in range(6)]
            
            ncols = int(lines[0].split(' ')[-1])
            nrows = int(lines[1].split(' ')[-1])
            xllcenter = int(floor(float(lines[2].split(' ')[-1])))
            yllcenter = int(floor(float(lines[3].split(' ')[-1])))
            cellsize = int(lines[4].split(' ')[-1])
            epsg_code = test_against_location(xllcenter, yllcenter)
            if epsg_code == None:
                return FAILURE, 'File does not cover Poland'
            return SUCCESS, (filename, epsg_code, xllcenter, yllcenter, nrows, ncols, cellsize)
    except Exception as e:
        return FAILURE, e


def get_base_map(center = (51.9537505,19.1343786), zoom = 7):
    folium_map = folium.Map(location=center, zoom_start=zoom)
    folium.TileLayer('http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg', attr='terrain-bcg', name='Terrain Map').add_to(folium_map)

    return folium_map


def add_lines_to_map(folium_map, lines, line_colors = None):
    if line_colors is None:
        line_colors = colors[1:len(lines) + 1]
    for i, line in enumerate(lines):
        folium.PolyLine(line, color=line_colors[i], weight=2.5, opacity=1).add_to(folium_map)
    return folium_map


def add_database_coverage_to_map(folium_map, coverage):
    for (lat, lon), (lat2, lon2) in coverage:
        folium.Rectangle(bounds=[(lat, lon), (lat2, lon2)], color='blue', fill=True, fill_opacity=0.2).add_to(folium_map)
    return folium_map


def html_representation_of_map(folium_map):
    return folium_map._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')


@app.route('/test')
def test():
    coverage = get_map_coverage(db)
    folium_map = get_base_map()
    folium_map = add_database_coverage_to_map(folium_map, coverage)
    map_html = html_representation_of_map(folium_map)

    filenames = list(sorted(os.listdir(GPXDATA)))
    return render_template('index.html', map=map_html, filenames=filenames)


@app.route('/add_to_database', methods=['POST', 'GET'])
def add_to_database():
    # TODO : check if user is admin
    # TODO : check if file is already in database
    # TODO : check if file is valid
    # TODO : check if file is not too big
    # TODO : if successful, give user feedback
    if not is_user_admin():
        return redirect(url_for('index'))

    if request.method == 'POST':
        f = request.files.getlist('file')
        for file in f:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            result, data = read_database_file(file.filename)
            if result == SUCCESS:
                map_data = MapData(*data)
                db.session.add(map_data)
                db.session.commit()
                os.rename(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), 
                          os.path.join(app.config['DATABASE_FOLDER'], file.filename))
            else:
                flash(data)

        return redirect(url_for('add_to_database'))
    else:
        return render_template('add_to_database.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    user = None
    
    if request.method == 'POST':
        f = request.files['file']
        if not f.filename.endswith('.gpx'):
            flash('File must be .gpx')
            return redirect(url_for('upload'))
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        
        # TODO: asyncio convertion? 
        
        last_id = len(db.session.query(Track).all())

        coverage = get_database_coverage(db)
        result, error_message = convert(coverage, 
                os.path.join(app.config['UPLOAD_FOLDER'], f.filename),
                os.path.join(app.config['CONVERTED_GPX_FOLDER'], f"{last_id + 1}.pkl"))                                   

        print(result, error_message)
        if result == SUCCESS:
            flash('File uploaded successfully')
        else:
            flash(error_message)
            return redirect(url_for('upload'))

        os.rename(os.path.join(app.config['UPLOAD_FOLDER'], f.filename),
                  os.path.join(app.config['ORIGINAL_GPX_FOLDER'], f"{last_id + 1}.gpx"))

        track = Track(
            user_id=1,
            file_name=f.filename,
            file_path=os.path.join(app.config['ORIGINAL_GPX_FOLDER'], f"{last_id + 1}.gpx"),
            converted_file_path=os.path.join(app.config['CONVERTED_GPX_FOLDER'], f"{last_id + 1}.pkl")
        )

        db.session.add(track)
        db.session.commit()

        return redirect(url_for('track', track_id=track.id))
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
    # print(points[0])
    if a:
        points = points[a:b]
    x = [0]
    for i in range(1, len(points)):
        point1 = (points[i - 1][0][1], points[i - 1][0][0])
        point2 = (points[i][0][1], points[i][0][0])
        x.append(calculate_distance(point1, point2))
    x = np.cumsum(x)
    x = [round(ele, 2) for ele in x]
    # print(x)
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


@app.route('/compare/<filename>/<start>/<end>')
def compare(filename, start, end):
    df, points = process_gpx_to_df(GPXDATA + filename)
    start = int(start)
    end = int(end)
    segment_points = points[start:end]

    mymap = folium.Map(location=[ df[start:end].Latitude.mean(), df[start:end].Longitude.mean() ], zoom_start=15)
    folium.TileLayer('http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg', attr='terrain-bcg', name='Terrain Map').add_to(mymap)

    html_start = "Start of the track {}".format("Lemko 2018")
    start_popup = folium.Popup(html_start, max_width=400)
    end_popup = folium.Popup("End of the track {}".format("Lemko 2018"), max_width=400)
    
    folium.Marker(location=segment_points[0], popup=start_popup, tooltip='Start', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:green;opacity:0.5"></div>')).add_to(mymap)
    folium.Marker(location=segment_points[-1], popup=end_popup, tooltip='End', icon=folium.DivIcon(html='<div style="margin-top:-50%;margin-left:-50%;width:25px;height:25px;border-radius:12.5px;background:red;opacity:0.5"></div>')).add_to(mymap)

    folium.PolyLine(segment_points, color="blue", weight=2.5, opacity=1).add_to(mymap)

    tab = []
    count = 0
    for file in os.listdir(GPXDATA):
        if file.endswith('.gpx'):
            _, points = process_gpx_to_df(GPXDATA + file)
            
            start = time()
            llist = find_points_for_segment([convert_latlon_to_xll(lat, lon) for (lat, lon) in segment_points], points)
            print(file, len(llist), time() - start)
            if len(llist) > 0:
                for j, track in enumerate(llist):
                    track_points = [points[i] for i in track]
                    folium.PolyLine(track_points, color=colors[2+count], weight=2.5, opacity=1).add_to(mymap)
                    tab.append((colors[2+count], file, track[0], track[-1], 'NA'))
                    count += 1
    print(tab)

    map_html = mymap._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')
    return render_template('compare.html', map=map_html, table=tab)


def get_directional_change_data(track):
    points = track['gpx']

    distances = [0]
    for i in range(1, len(points)):
        distances.append(calculate_distance(points[i - 1], points[i]))
    distances = np.cumsum(distances)

    enumerated_elevations = [(i, ele) for i, ele in enumerate(track['elevations'])]
    plot, ids_change = plot_directional_change(enumerated_elevations, 0.015, distances)

    g2 = table(enumerated_elevations, ids_change)
    
    g2['start_id'] = g2['start']
    g2['end_id'] = g2['end']
    g2['start'] = g2['start'].apply(lambda a: str(round(distances[a], 2)) + ' km')
    g2['end'] = g2['end'].apply(lambda a: str(round(distances[a], 2)) + ' km')
    g2['elevation_gain'] = g2['elevation_gain'].apply(lambda a: str(round(a, 2)) + ' m')
    g2['elevation_loss'] = g2['elevation_loss'].apply(lambda a: str(round(a, 2)) + ' m')

    # dataframe to list 
    tab = g2.to_numpy().tolist()

    return SUCCESS, (plot, tab, ids_change)


def add_directional_change_lines(folium_map, points, ids_change):
    ids_change = [0] + ids_change + [len(points)]
    
    lines = [points[ids_change[i]:ids_change[i + 1] + 1] for i in range(len(ids_change) - 1)]
    folium_map = add_lines_to_map(folium_map, lines, ['green', 'red'] * ceil(len(lines) / 2))

    return folium_map


def set_optimal_position_and_zoom(folium_map, points):
    points = np.array(points)
    folium_map.location = list(points.mean(axis=0))
    folium_map.fit_bounds([list(points.min(axis=0)), list(points.max(axis=0))])
    return folium_map


@app.route('/track/<track_id>')
def track(track_id):
    print(folium.__version__)
    # TODO : check if user is privileged to see this track
    track = Track.query.filter_by(id=track_id).first()
    if not track:
        return redirect(url_for('index'))
    track_pkl = pickle.load(open(track.converted_file_path, 'rb'))

    folium_map = get_base_map()
    folium_map = set_optimal_position_and_zoom(folium_map, track_pkl['gpx'])
    _, (plot, tab, ids_change) = get_directional_change_data(track_pkl)
    folium_map = add_directional_change_lines(folium_map, track_pkl['gpx'], ids_change)

    map_html = html_representation_of_map(folium_map)
    return render_template('index.html', map=map_html, graph1=plot, table=tab, a = 0)



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

        start = time()
        x = [0]
        for i in range(1, len(points)):
            x.append(calculate_distance(points[i - 1], points[i]))
        x = np.cumsum(x)

        filenames = list(sorted(os.listdir(GPXDATA)))
        elevation_points = parse_gpx_to_points(GPXDATA + filename)
        elevation_points = elevation_points[a:b]
        result, data = get_plot_and_breakpoints(GPXDATA + filename, a, b)
        ele = [(i, ele) for i, (_, ele) in enumerate(elevation_points)]
        print("Time taken", time() - start)

        g2 = table(ele, data[1])
        
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

        # take segment from 1.gpx and compare it against other tracks.
        # _, segment_points = process_gpx_to_df(GPXDATA + '/1.gpx')
        # start = time()
        # llist = find_points_for_segment([convert_latlon_to_xll(lat, lon) for (lat, lon) in segment_points[2090:2332]], points)
        # print("Matched", len(llist), "times")
        # print("Time taken", time() - start)

        # for (x, y) in segment_points[2090:2332]:
        #     folium.CircleMarker(location=[x, y], radius=2, color='black', fill=True, fill_color='black', fill_opacity=1).add_to(mymap)

        # for i, track in enumerate(llist):
        #     track_points = [points[i] for i in track]
        #     track_color = colors[2 + i]
        #     folium.PolyLine(track_points, color=track_color, weight=2.5, opacity=1).add_to(mymap)

        map_html = mymap._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')

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
    from models import *
    from common.database_utility import *

    with app.test_request_context():
        db.create_all()

    app.run(debug=True, port=5001)
