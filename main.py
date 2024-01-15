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

from folium.plugins import BeautifyIcon, TimestampedGeoJson

from common.utilities import *
import sys
sys.setrecursionlimit(2001)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER')
app.config['DATABASE_FOLDER'] = os.environ.get('DATABASE_FOLDER')
app.config['ORIGINAL_GPX_FOLDER'] = os.environ.get('ORIGINAL_GPX_FOLDER')
app.config['CONVERTED_GPX_FOLDER'] = os.environ.get('CONVERTED_GPX_FOLDER')

db_name = os.environ.get('DB_NAME')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.environ.get('SQLALCHEMY_TRACK_MODIFICATIONS')

colors = "Blue Red Black Grey Orange White Brown Pink Yellow Green Purple Maroon Turquoise Cyan Navy blue Gold Tomato Teal Lime Cyan Wheat Salmon Olive Aqua Violet".split(' ')

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
    map_html = folium_map._repr_html_().replace('height:0', 'height:calc(100vh - 86px)')
    map_html = map_html.replace('&lt;script&gt;', """&lt;script&gt;
            document.body.onkeyup = function(event) {
                // console.log(event.code);
                if (event.code == 'Space') {
                    document.getElementsByClassName('timecontrol-play')[0].click()
                }
            }
                                """
                                )

    return map_html


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



@app.route('/')
def index():
    return redirect(url_for('tracklist'))
    # coverage = get_map_coverage()
    # folium_map = get_base_map()
    # folium_map = add_database_coverage_to_map(folium_map, coverage)
    # map_html = html_representation_of_map(folium_map)

    # # filenames = list(sorted(os.listdir(GPXDATA)))
    # return render_template('index.html', map=map_html)#, filenames=filenames)


@app.route('/tracklist')
def tracklist():
    tracks = Track.query.all()
    
    lines = []
    ids = []
    for track in tracks:
        track_pkl = pickle.load(open(track.converted_file_path, 'rb'))
        lines.append(track_pkl['gpx'])
        ids.append((track.file_name, track.id))
       
    folium_map = get_base_map()
    folium_map = set_optimal_position_and_zoom(folium_map, [x for line in lines for x in line])
    folium_map = add_lines_to_map(folium_map, lines)

    map_html = html_representation_of_map(folium_map)
    return render_template('tracklist.html', map=map_html, ids=ids)
    
    


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

        coverage = get_database_coverage()
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


def add_blue_point(folium_map, location, tooltip):
    icon_circle = BeautifyIcon(
        icon_shape='circle-dot', 
        border_color='blue', 
        border_width=6,
        inner_icon_style='opacity:0.3'
    )
    folium.Marker(location=location, tooltip=tooltip, icon=icon_circle).add_to(folium_map)
    return folium_map


def add_timed_points(folium_map, track, times, distances, color, startstop = False):
    def point(lat, lon, time, distance, tooltip, popup):
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [lon, lat]
            },
            'properties': {
                'time': time.isoformat(),
                'popup' : popup,
                'tooltip': tooltip,
                'icon': 'circle',
                'iconstyle': {
                    'color': color,
                    'fillOpacity': 0.6,
                    'tooltip': f'Point {round(distance, 2)}km'
                }
            }
        }

    data = []
    for i, (lat, lon) in enumerate(track):
        popup = f'Point {round(distances[i], 2)}km'
        if startstop:
            popup = f"""<button class="Set as start" onclick="parent.set_start({i})">Set as start</button><br>
                        <button class="Set as end" onclick="parent.set_end({i})">Set as end</button>"""
        
        data.append(point(lat, lon, times[i], distances[i], f'Point {round(distances[i], 2)}km', popup))

    TimestampedGeoJson({
        'type': 'FeatureCollection',
        'features': data,
    }, period='PT1S', 
    duration='PT1S',
    transition_time=10,
    auto_play=True,
    add_last_point=True,
    max_speed=100).add_to(folium_map)

    return folium_map


@app.route('/compare/<track_id>/<start>/<end>')
def compare(track_id, start, end):
    track = Track.query.filter_by(id=track_id).first()
    if not track:
        return redirect(url_for('index'))
    track_pkl = pickle.load(open(track.converted_file_path, 'rb'))

    start = int(start)
    end = int(end)

    for key in track_pkl.keys():
        track_pkl[key] = track_pkl[key][start:end]

    
    all_tracks = Track.query.all()
    all_tracks.remove(track)
    result = []
    result_timed = []
    tab = []
    tab.append(('blue', track.file_name, start, end, track_pkl['time'][-1] - track_pkl['time'][0] if track_pkl['time'][0] else 'NA', track.id))
    color_tag = 1
    for track in all_tracks:
        track_pkl2 = pickle.load(open(track.converted_file_path, 'rb'))
        traces = find_points_for_segment(track_pkl[2178], track_pkl2[2178])
        for trace in traces:
            result.append([track_pkl2['gpx'][i] for i in trace])

            # diff between start and end
            diff = 'NA'
            if track_pkl2['time'][0] != None:
                diff = track_pkl2['time'][trace[-1]] - track_pkl2['time'][trace[0]]
                result_timed.append([track_pkl2['time'][i] for i in trace])
            else:
                result_timed.append(None)

            tab.append((colors[color_tag], track.file_name, trace[0], trace[-1], diff, track.id))

            color_tag += 1

    folium_map = get_base_map()
    folium_map = set_optimal_position_and_zoom(folium_map, track_pkl['gpx'])
    folium_map = add_lines_to_map(folium_map, result)

    distances = [0]
    for i in range(1, len(track_pkl['gpx'])):
        distances.append(calculate_distance(track_pkl['gpx'][i - 1], track_pkl['gpx'][i]))
    distances = np.cumsum(distances)
    
    print(tab)

    folium_map = add_lines_to_map(folium_map, [track_pkl['gpx']], ['blue'])

    if track_pkl['time'][0] != None:
        folium_map = add_timed_points(folium_map, track_pkl['gpx'], track_pkl['time'], distances, 'blue')
        for i in range(len(result_timed)):
            if result_timed[i] != None:
                start_time = track_pkl['time'][0]
                s_time = result_timed[i][0]
                result_timed[i] = [s_time] + [start_time + (x - s_time) for x in result_timed[i][1:]]
                folium_map = add_timed_points(folium_map, result[i], result_timed[i], distances, colors[i + 1])

    map_html = html_representation_of_map(folium_map)
    return render_template('compare.html', map=map_html, table=tab)

@app.route('/track/<track_id>', methods=['GET'], defaults={'a' : None, 'b' : None})
@app.route('/track/<track_id>/<a>/<b>', methods=['GET'])
def track(track_id, a, b):
    # TODO : check if user is privileged to see this track
    track = Track.query.filter_by(id=track_id).first()
    if not track:
        return redirect(url_for('index'))
    track_pkl = pickle.load(open(track.converted_file_path, 'rb'))

    if a:
        a = int(a)
        b = int(b) + 1
    else:
        a = 0
        b = len(track_pkl['gpx'])
    
    for key in track_pkl.keys():
        track_pkl[key] = track_pkl[key][a:b]

    distances = [0]
    for i in range(1, len(track_pkl['gpx'])):
        distances.append(calculate_distance(track_pkl['gpx'][i - 1], track_pkl['gpx'][i]))
    distances = np.cumsum(distances)

    folium_map = get_base_map()
    folium_map = set_optimal_position_and_zoom(folium_map, track_pkl['gpx'])
    _, (plot, tab, ids_change) = get_directional_change_data(track_pkl)
    folium_map = add_directional_change_lines(folium_map, track_pkl['gpx'], ids_change)

    if track_pkl['time'][0] != None:
        add_timed_points(folium_map, track_pkl['gpx'], track_pkl['time'], distances, 'blue', True)

    for i, val in enumerate(ids_change):
        tooltip = f'Going {"down" if i % 2 == 0 else "up"}.\nPoint {round(distances[val], 2)}km'
        folium_map = add_blue_point(folium_map, track_pkl['gpx'][val], tooltip)

    map_html = html_representation_of_map(folium_map)
    return render_template('track.html', map=map_html, graph1=plot, table=tab, a = a, b = b, my_id = track_id)


@app.route('/peek')
def peek():
    base_map = get_base_map()
    base_map = add_database_coverage_to_map(base_map, get_map_coverage())

    filenames = list(sorted(os.listdir(app.config['ORIGINAL_GPX_FOLDER'])))
    
    for i, filename in enumerate(filenames):
        track = parse(open(os.path.join(app.config['ORIGINAL_GPX_FOLDER'], filename)), 'r')
        points = []
        for segment in track.tracks[0].segments:
            for point in segment.points:
                points.append((point.latitude, point.longitude))
        
        fg = folium.FeatureGroup(name=filename, show=True)
        folium.PolyLine(points, color=colors[i + 1], weight=2.5, opacity=1).add_to(fg)

        base_map.add_child(fg)

    folium.LayerControl().add_to(base_map)

    map_html = html_representation_of_map(base_map)
    return render_template('peek.html', map=map_html, filenames=filenames)        


if __name__ == '__main__':
    from models import *
    from common.database_utility import *

    with app.test_request_context():
        db.create_all()

    app.run(debug=True, port=5001)
