import os
import sys
import fiona
import numpy as np
import pyproj
import pathlib
from shapely.geometry import Point, mapping, shape
# from shapely.prepared import prep

def set_fcp_alert_matrix(**kwargs):

    fcp_dic  = kwargs.get('fcp', None)
    Config   = kwargs.get('cfg', None)
    eve_dic  = kwargs.get('event_parameters', None)
    ptf      = kwargs.get('ptf_status', None)

    local_distance    = Config.getfloat('matrix', 'local_distance')
    regional_distance = Config.getfloat('matrix', 'regional_distance')
    # basin_distance    = Config.getfloat('matrix', 'basin_distance')

    eve_dict = define_level_alerts_decision_matrix(event_parameters = eve_dic, cfg = Config)

    # qui da rifare per aggiornare matrix
    for i in range(len(fcp_dic['data'])):

        if(fcp_dic['data'][i]['distance_km'] <= local_distance):
            fcp_dic['data'][i]['matrix_fcp_alert_type'] = eve_dic['alert_local']

        elif(fcp_dic['data'][i]['distance_km'] > local_distance and
             fcp_dic['data'][i]['distance_km'] <= regional_distance):
             fcp_dic['data'][i]['matrix_fcp_alert_type'] = eve_dic['alert_regional']

        else:
            fcp_dic['data'][i]['matrix_fcp_alert_type'] = eve_dic['alert_basin']

        if(fcp_dic['data'][i]['matrix_fcp_alert_type'] == 3):
            fcp_dic['data'][i]['matrix_fcp_alert_level'] = 'watch'
        elif(fcp_dic['data'][i]['matrix_fcp_alert_type'] == 2):
            fcp_dic['data'][i]['matrix_fcp_alert_level'] = 'advisory'
        elif(fcp_dic['data'][i]['matrix_fcp_alert_type'] == 1):
            fcp_dic['data'][i]['matrix_fcp_alert_level'] = 'information'
        else:
            fcp_dic['data'][i]['matrix_fcp_alert_level'] = 'unset'

        fcp_dic['data'][i]['matrix_state_alert_level'] = fcp_dic['data'][i]['matrix_fcp_alert_level']
        fcp_dic['data'][i]['matrix_state_alert_type']  = fcp_dic['data'][i]['matrix_fcp_alert_type']

    return fcp_dic

def set_points_alert_level(**kwargs):

    distances = kwargs.get('distances', None)
    eve_dic   = kwargs.get('event_parameters', None)
    Config    = kwargs.get('cfg', None)

    local_distance    = Config.getfloat('matrix', 'local_distance')
    regional_distance = Config.getfloat('matrix', 'regional_distance')
    # basin_distance    = Config.getfloat('matrix', 'basin_distance')

    levels = np.ones((len(distances)))

    for i in range(len(distances)):


        if(distances[i] <= local_distance):
            levels[i] = eve_dic['alert_local']

        elif(distances[i] > local_distance and distances[i] <= regional_distance):
            levels[i] = eve_dic['alert_regional']

        else:
            levels[i] = eve_dic['alert_basin']


    return levels


def define_decision_matrix():

    dcm = {
      'III0' : [(0,    100,    -1,  40,   5.499,  6.001,   'information', 'information', 'information')],
      'AII1' : [(0,    100,    -1,  40,   6.000,  6.501,   'advisory',    'information', 'information')],
      'III2' : [(0,    100,    40, 100,   5.499,  6.501,   'information', 'information', 'information')],
      'WAI3' : [(0,    100,    -1, 100,   6.500,  7.001,   'watch',       'advisory',    'information')],
      'WWA4' : [(0,    100,    -1, 100,   7.000,  7.501,   'watch',       'watch',       'advisory'   )],
      'WWW5' : [(0,    100,    -1, 100,   7.500, 19.900,   'watch',       'watch',       'watch'      )],
      'III6' : [(100, 2000,    -1, 100,   5.499, 19.900,   'information', 'information', 'information')]
    }

    return dcm

def define_decision_matrix_number(**kwargs):

    Config = kwargs.get('cfg', None)

    min_mag = float(Config.get('matrix', 'min_mag_for_message'))
    """
    dcm = {
      'III0' : [(0,    100,    -1,  40,   5.499,  6.001,   1, 1, 1)],
      'AII1' : [(0,    100,    -1,  40,   6.000,  6.501,   2, 1, 1)],
      'III2' : [(0,    100,    40, 100,   5.499,  6.501,   1, 1, 1)],
      'WAI3' : [(0,    100,    -1, 100,   6.500,  7.001,   3, 2, 1)],
      'WWA4' : [(0,    100,    -1, 100,   7.000,  7.501,   3, 3, 2)],
      'WWW5' : [(0,    100,    -1, 100,   7.500, 19.900,   3, 3, 3)],
      'III6' : [(100, 2000,    -1, 100,   5.499, 19.900,   1, 1, 1)]
    }
    """

    dcm = {
      'III0' : [(0,    100,    -1,  40, min_mag,  6.001,   1, 1, 1)],
      'AII1' : [(0,    100,    -1,  40,   6.000,  6.501,   2, 1, 1)],
      'III2' : [(0,    100,    40, 100,   5.499,  6.501,   1, 1, 1)],
      'WAI3' : [(0,    100,    -1, 100,   6.500,  7.001,   3, 2, 1)],
      'WWA4' : [(0,    100,    -1, 100,   7.000,  7.501,   3, 3, 2)],
      'WWW5' : [(0,    100,    -1, 100,   7.500, 19.900,   3, 3, 3)],
      'III6' : [(100, 2000,    -1, 100,   5.499, 19.900,   1, 1, 1)]
    }

    return dcm

def get_distance_point_to_Ring(**kwargs):

    ee         = kwargs.get('event_parameters', None)
    geoms      = kwargs.get('land_geometry', None)
    lat        =  float(ee['lat'])
    lon        =  float(ee['lon'])

    if not geoms:
        return 0

    ellipsoid  = pyproj.Geod(ellps='WGS84')


    d_min  = sys.float_info.max
    la_min = 0
    lo_min = 0
    #point  = sgeom.Point(x,y)

    #print(list(geoms['coordinates'][0]))

    #ing   = sgeom.LinearRing(list(geoms.exterior.coords))
    #arr    = list(geoms.exterior.coords)
    arr    = list(geoms['coordinates'][0])
    for i in range(len(arr)):
        a,b,d = ellipsoid.inv(lon, lat, arr[i][0], arr[i][1])
        if (d <= d_min):
            d_min = d
            lo_min = arr[i][0]
            la_min = arr[i][1]

    d_min=d_min/1000
    
    return (d_min)

def check_if_point_is_land(**kwargs):

    ee         = kwargs.get('event_parameters', None)
    Config     = kwargs.get('cfg', None)

    # shp_path   = Config.get('pyptf','path_to_coast_shapefiles')
    shapefile   = Config.get('pyptf', 'coast_shapefile')
    lat        = float(ee['lat'])
    lon        = float(ee['lon'])

    # default value
    ee['epicenter_is_in_land'] = False

    # Create Point
    #point       = Point(x,y)
    point_shp   = {'properties': {'LocationID': '0', 'Latitude': lat, 'Longitude': lon },'geometry': mapping(Point(lon,lat))}
    point       = shape(point_shp['geometry'])

    # Shape file
    # shapefile  = shp_path + os.sep + shp_file
    source     = fiona.open(shapefile)

    #Loop over shapes of shapefile
    for f in source:
        a = point.within(shape(f['geometry']))
        if (a == True):
           ee['epicenter_is_in_land'] =  True
           selected_polygon = f['geometry']
           return ee, selected_polygon

    return ee, False

def define_level_alerts_decision_matrix(**kwargs):


    dct      = kwargs.get('event_parameters', None)
    Config   = kwargs.get('cfg', None)

    #dcm  = define_decision_matrix()
    dcm  = define_decision_matrix_number(cfg = Config)

    #result = find_level_for_magnitude(dcm, dct['mag'], 4, 5)
    result = find_level_for_magnitude(dcm, dct['mag_percentiles']['p50'], 4, 5)

    if(len(result) >= 1):
        result = find_level_for_magnitude(result, dct['depth'], 0, 1)

    if(len(result) >= 1):
        result = find_level_for_magnitude(result, dct['epicentral_distance_from_coast'], 2, 3)

    dct['alert_local']        = 'None'
    dct['alert_regional']     = 'None'
    dct['alert_basin']        = 'None'
    dct['in_decision_matrix'] = False


    if(len(result) == 1):
        temp_list                 = list(result.items())[0][1]
        dct['alert_local']        = list(temp_list[0])[6]
        dct['alert_regional']     = list(temp_list[0])[7]
        dct['alert_basin']        = list(temp_list[0])[8]
        dct['in_decision_matrix'] = True

    else:
        dct['alert_local']     = 'Event out of DM'
        dct['alert_regional']  = 'Event out of DM'
        dct['alert_basin']     = 'Event out of DM'

    return dct

def find_level_for_magnitude(dictionary, val, i1, i2):

    matches = {}
    for key, record_list in dictionary.items():
        for record in record_list:
            #value = record[1]
            if val > record[i1] and val <= record[i2]:
                if key in matches:
                    matches[key].append(record)
                else:
                    matches[key] = [record]
    return matches
