from common.utilities import convert_xll_to_latlon, MIN_EPSG, MAX_EPSG
from models import User, Track, MapData

def get_database_coverage(db):
    """
    Returns a list of all xll, yll for the map data in the database.
    """
    query = db.session.execute("SELECT * FROM map_data").fetchall()
    result = {}
    for epsg_code in range(MIN_EPSG, MAX_EPSG + 1):
        result[epsg_code] = []
    for row in query:
        xur = row['xll'] + (row['ncols'] * row['cellsize']) - 1
        yur = row['yll'] + (row['nrows'] * row['cellsize']) - 1
        result[row['epsg_code']].append(((row['xll'], row['yll']), (xur, yur), row['file_name']))
    return result


def get_map_coverage(db):
    """
    Returns a list of coordinates that represent the coverage of all the maps in the database.
    """
    query = db.session.execute("SELECT * FROM map_data").fetchall()
    result = []
    for row in query:
        xur = row['xll'] + (row['ncols'] * row['cellsize'])
        yur = row['yll'] + (row['nrows'] * row['cellsize'])
        lat, lon = convert_xll_to_latlon(row['xll'], row['yll'], row['epsg_code'])
        lat2, lon2 = convert_xll_to_latlon(xur, yur, row['epsg_code'])
        result.append(((lat, lon), (lat2, lon2)))
    return result


def get_list_of_tracks(db):
    """
    Returns a list of all tracks in the database.
    """
    query = db.session.execute("SELECT * FROM track").fetchall()
    result = []
    for row in query:
        result.append((row['file_name'], row['file_path'], row['converted_file_path']))
    return result