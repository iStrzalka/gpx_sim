import datetime
from flask_sqlalchemy import SQLAlchemy

from __main__ import app

db = SQLAlchemy(app)

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

    def __init__(self, user_id, file_name, file_path, converted_file_path):
        self.user_id = user_id
        self.file_name = file_name
        self.file_path = file_path
        self.converted_file_path = converted_file_path


class MapData(db.Model):
    __tablename__ = 'map_data'
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(100))
    added_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    added_date = db.Column(db.DateTime)
    epsg_code = db.Column(db.Integer)
    xll = db.Column(db.Integer)
    yll = db.Column(db.Integer)
    nrows = db.Column(db.Integer)
    ncols = db.Column(db.Integer)
    cellsize = db.Column(db.Integer)
    
    def __init__(self, file_name, epsg_code, xll, yll, nrows, ncols, cellsize):
        self.file_name = file_name
        self.epsg_code = epsg_code
        self.xll = xll
        self.yll = yll
        self.nrows = nrows
        self.ncols = ncols
        self.cellsize = cellsize
        self.added_date = datetime.datetime.now()
        self.added_by = None