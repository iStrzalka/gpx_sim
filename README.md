### Flask application for viewing and comparing gpx files.
## About
Application that gets users gpx files and fixes elevation values against known elevation data and shows analysis on said track. User can also compare against other tracks within the database (for now). 

## Pages
- /track - Given id of track returns map with track as well as data about elevation change
- /peek - Given id of track returns map with just the original track 
- /compare - Compares track against other tracks 
- /tracklist - Shows map with all tracks in the database
- /add_to_database - Page for uploading elevation data from https://mapy.geoportal.gov.pl/imap/Imgp_2.html
- /upload - Uploads a track to database, and transforms elevation data against database's elevation data (since often gpx data provided by watches are usually incorrect)

## Plans
[ ] Change algorithm for comparison since current one is too slow or 
[ ] try to use osmnx against current map to try to ease out comparison as well as elevation data
[ ] Maybe diffrenciate users and add some admin properties? 