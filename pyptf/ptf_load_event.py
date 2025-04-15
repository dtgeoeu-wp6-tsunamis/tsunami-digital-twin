import os
import sys
import utm
import math
import shutil
import reverse_geocoder as rg
import numpy as np
from tornado          import escape
from time             import gmtime, strftime
from obspy.core       import UTCDateTime
#from scipy.stats      import norm
#from scipy.stats      import distributions
from pyptf.ptf_matrix         import check_if_point_is_land
from pyptf.ptf_matrix         import get_distance_point_to_Ring
from pyptf.ptf_mix_utilities  import find_epicentral_area
from pyptf.ptf_scaling_laws   import correct_BS_horizontal_position
from pyptf.ptf_scaling_laws   import correct_PS_horizontal_position
from pyptf.pyptf_exceptions   import PyPTFException


def create_event_dict(**kwargs):

    Config        = kwargs.get('cfg', None)
    workflow_dict = kwargs.get('workflow_dict', None)
    jsn_object    = kwargs.get('jsn_object', None)
    geocode_area  = kwargs.get('geocode_area', None)
    logger        = kwargs.get('logger', None)

    event_parameters = int_quake_cat2dict(json          = jsn_object, 
                                          geocode_area  = geocode_area,
                                          mag_sigma_val = workflow_dict['mag_sigma_val'],
                                          logger        = logger)

    event_parameters['sigma'] = workflow_dict['sigma']
    event_parameters = compute_position_sigma_lat_lon(event = event_parameters,
                                                      cfg   = Config)

    event_parameters['area'] = find_epicentral_area(lon=event_parameters['lon'], lat=event_parameters['lat'], cfg=Config)
    # logger.info(f" --> Event within the area: {event_parameters['area']}")

    if event_parameters['area'] == 'cat_area':
        logger.info(' --> Event within the CAT area')
    else:
        if workflow_dict['ptf_version'] == 'neam':
            #raise Exception('If the event is out of the CAT area ptf_version must be set to global.')
            logger.info('If the event is out of the CAT area ptf_version must be set to global.')

    # check if event inland with respect neam guidelines
    event_parameters, geometry_land_with_point = check_if_point_is_land(event_parameters=event_parameters, cfg=Config)
    event_parameters['epicentral_distance_from_coast'] = get_distance_point_to_Ring(land_geometry=geometry_land_with_point, 
                                                                                    event_parameters = event_parameters)

    if (event_parameters['epicentral_distance_from_coast'] > 100):
        logger.info(' --> Epicenter INLAND > 100 km (%.3f [km])' % (event_parameters['epicentral_distance_from_coast'] ))
        # raise PyPTFException('PTF workflow not executed because the epicenter is inland (>100km)')

    else:
        logger.info(' --> Epicenter INLAND < 100 km or OFFSHORE (%.3f [km])' % (event_parameters['epicentral_distance_from_coast']))

    return event_parameters


def read_event_parameters(**kwargs):
    """
    This function loads the json file (or any other supported format like xml
    or csv) which contains the event informations.

    If the file format is json like, the json file reflect the rabbit-mq_ingv-ont
    file format. See documentations in docs folder

    Return the event dictionary and a ptf dictionary containing the main
    properties of the event
    """
    event_file    = kwargs.get('event_file', None)
    event_format  = kwargs.get('event_format', None)

    if (event_file != None and event_format == 'json'):
        s = open(event_file, 'r').read()
        jsn_object = eval(s)
        jsn_object['event_file'] = event_file
    else:
        sys.exit("Event defined uncorrectly!")

    return jsn_object


def compute_position_sigma_lat_lon(**kwargs):
    """
    REFERENCE LAT = YY
    REFERENCE LON = XX
    """

    event_parameters = kwargs.get('event', None)
    Config           = kwargs.get('cfg', None)

    bs_mag_max          = float(Config.get('Settings', 'Mag_BS_Max'))

    sigma               = event_parameters['sigma']
    event_mag           = event_parameters['mag_percentiles']['p50']
    event_mag_max       = event_parameters['mag_percentiles']['p50'] + \
                          event_parameters['MagSigma'] * sigma
    event_mag_sigma     = event_parameters['MagSigma']

    event_cov_xx        = event_parameters['pos_Sigma']['XX']
    event_cov_xy        = event_parameters['pos_Sigma']['XY']
    event_cov_yy        = event_parameters['pos_Sigma']['YY']


    mag_to_correct      = min(bs_mag_max, event_mag_max)

    delta_position_BS_h = correct_BS_horizontal_position(mag=mag_to_correct)
    delta_position_PS_h = correct_PS_horizontal_position(mag=event_mag + sigma * event_mag_sigma)

    position_BS_sigma_yy  = math.sqrt(abs(event_cov_yy)) + delta_position_BS_h
    position_BS_sigma_xx  = math.sqrt(abs(event_cov_xx)) + delta_position_BS_h
    position_BS_sigma_xy  = math.sqrt(abs(event_cov_xy)) + delta_position_BS_h

    event_parameters['position_BS_sigma_yy'] = position_BS_sigma_yy
    event_parameters['position_BS_sigma_xx'] = position_BS_sigma_xx
    event_parameters['position_BS_sigma_xy'] = position_BS_sigma_xy

    position_PS_sigma_yy  = math.sqrt(abs(event_cov_yy)) + delta_position_PS_h
    position_PS_sigma_xx  = math.sqrt(abs(event_cov_xx)) + delta_position_PS_h
    position_PS_sigma_xy  = math.sqrt(abs(event_cov_xy)) + delta_position_PS_h

    event_parameters['position_PS_sigma_yy'] = position_PS_sigma_yy
    event_parameters['position_PS_sigma_xx'] = position_PS_sigma_xx
    event_parameters['position_PS_sigma_xy'] = position_PS_sigma_xy

    return event_parameters


def print_event_parameters(**kwargs):

    d      = kwargs.get('event_dict', None)
    logger = kwargs.get('logger', None)

    logger.info(" --> eventid:                    %s" % (d['eventid']))
    logger.info(" --> originid:                   %s" % (d['originid']))
    logger.info(" --> version:                    %s" % (d['version']))
    logger.info(" --> author:                     %s" % (d['author']))
#    logger.info(" --> routing_key:                %s" % (d['routing_key']))
    logger.info(" --> OT and Epicenter:           ot: %s Lat: %.3f Lon: %.3f Depth: %.2f" % (d['ot'], d['lat'], d['lon'], d['depth']))
    logger.info(" --> Location Covariant Matrix:  XX: %.5f XY: %.5f XZ: %.5f YY: %.5f YZ: %.5f ZZ: %.5f " % \
                                            (d['cov_matrix']['XX'], d['cov_matrix']['XY'],d['cov_matrix']['XZ'], \
                                            d['cov_matrix']['YY'],d['cov_matrix']['YZ'],d['cov_matrix']['ZZ'])) # XY: %.3f XZ: %.3f YY: %.3f YZ: %.3f ZZ: %.3f
    logger.info(" --> Position Sigma Matrix:      XX: %.5f XY: %.5f XZ: %.5f YY: %.5f YZ: %.5f ZZ: %.5f " % \
                                            (d['pos_Sigma']['XX'], d['pos_Sigma']['XY'],d['pos_Sigma']['XZ'], \
                                            d['pos_Sigma']['YY'],d['pos_Sigma']['YZ'],d['pos_Sigma']['ZZ'])) # XY: %.3f XZ: %.3f YY: %.3f YZ: %.3f ZZ: %.3f
    logger.info(" --> Position BS Sigma Lat:      yy: %f" % (d['position_BS_sigma_yy']))
    logger.info(" --> Position BS Sigma Lon:      xx: %f" % (d['position_BS_sigma_xx']))
    logger.info(" --> Position PS Sigma Lat:      yy: %f" % (d['position_PS_sigma_yy']))
    logger.info(" --> Position PS Sigma Lon:      xx: %f" % (d['position_PS_sigma_xx']))
    logger.info(" --> Magnitude:                  %.2f %s" %(d['mag'], d['mag_type']))
    logger.info(" --> Magnitude percentiles:      p16: %.2f p50: %.2f p84: %.2f" %(d['mag_percentiles']['p16'], d['mag_percentiles']['p50'],d['mag_percentiles']['p84']))
    logger.info(" --> MagSigma:                   %.3f" % (d['MagSigma']))
    logger.info(" --> Epicenter UTM region:       X: %.8f  Y: %.8f  Nr: %d  Code: %s" % (d['ee_utm']))


def geocoder_area2(**kwargs):

    lat = kwargs.get('lat', None)
    lon = kwargs.get('lon', None)


    coordinates = (lat, lon)
    # print("ACTION:", end = ' ')
    results = rg.search(coordinates)

    name = results[0]['cc'] + '_' + results[0]['name'] + '_' + results[0]['admin1']

    return name


def int_quake_cat2dict(**kwargs):

    json_string      = kwargs.get('json', None)
    geocode_area     = kwargs.get('geocode_area', None)
    mag_sigma_val    = kwargs.get('mag_sigma_val', None)  #default in ptf_parser.py (0.2)
    # mag_sigma_fix    = kwargs.get('mag_sigma_fix', None)
    logger        = kwargs.get('logger', None)

    d = dict()

    # Event Ids
    try:
        origin_id =  str(json_string['features'][0]['properties']['originid'])
    except:
        origin_id =  str(json_string['features'][0]['properties']['originId'])

    event_id      =  str(json_string['features'][0]['properties']['eventId'])
    author        =  str(json_string['features'][0]['properties']['author'])
    version       =  str(json_string['features'][0]['properties']['version'])

    # Epicenter informations
    lon           =  float(json_string['features'][0]['geometry']['coordinates'][0])
    lat           =  float(json_string['features'][0]['geometry']['coordinates'][1])
    depth         =  float(json_string['features'][0]['geometry']['coordinates'][2])
    area          =  str(json_string['features'][0]['properties']['place'])
    OT            =  str(json_string['features'][0]['properties']['time'])
    mag           =  float(json_string['features'][0]['properties']['mag'])
    ev_type       =  str(json_string['features'][0]['properties']['type'])
    mag_type      =  str(json_string['features'][0]['properties']['magType'])
    place          =  str(json_string['features'][0]['properties']['place'])

    #utm conversion
    ee_utm = utm.from_latlon(lat, lon)

    # to avoid issues due to negative depth
    if(depth < 0):
       depth = 0

    origin_time   = UTCDateTime(OT)

    Tmp_area      = area.rsplit(' [')
    Tmp_area      = Tmp_area[0].rsplit('[')
    area          = Tmp_area[0].replace(',','')
    area          = area.replace(' ','_')
    area          = area.replace('(','')
    area          = area.replace(')','')
    area          = area.replace('&','and')
    area          = area.replace('\\','_')
    area          = area.replace('/','_')

    if(geocode_area == 'No' or geocode_area == 'no'):
        area_geo  = 'unset'
    else:
        area_geo      = geocoder_area2(lat=lat, lon=lon)
        area_geo      = area_geo.replace(' ','-')
        area_geo      = area_geo.replace('&','_and_')
        area_geo      = area_geo.replace('(','')
        area_geo      = area_geo.replace(')','')
        area_geo      = area_geo.replace('[','')
        area_geo      = area_geo.replace(']','')
        area_geo      = area_geo.replace(',','')

    gmt_time      = gmtime()
    creation_time = strftime("%Y-%m-%d %H:%M:%S", gmt_time)
    origin_year   = origin_time.year
    origin_month  = ("%02d" % (origin_time.month))     #strftime("%m", gmt_time)
    origin_day    = ("%02d" % (origin_time.day))     #strftime("%d", gmt_time)

    # Specific Mag percentiles and covariant cov_matrix
    try:
        mag_percentiles = json_string['features'][0]['properties']['mag_percentiles']
    except:
        logger.info(f'Warning: magnitude percentiles not found in the json file; using default value mag_sigma_val = {mag_sigma_val}')
        mag_percentiles = dict()
        # mag_percentiles['p16'] = mag - 0.163
        # mag_percentiles['p50'] = mag + 0.0
        # mag_percentiles['p84'] = mag + 0.163
        mag_percentiles['p16'] = mag - mag_sigma_val
        mag_percentiles['p50'] = mag + 0.0
        mag_percentiles['p84'] = mag + mag_sigma_val

    try:
        mag_values = json_string['features'][0]['properties']['mag_values']
        mag_counts = json_string['features'][0]['properties']['mag_counts']
        if len(mag_values) != len(mag_counts):
            logger.info('WARNING: mag_values and mag_counts have different lengths. Magnitude percentiles will be used.')
            mag_values = []
            mag_counts = []
    except:
        logger.info('Magnitudes distribution not available. Magnitude percentiles will be used.')
        mag_values = []
        mag_counts = []

    try:
        cov_matrix      = json_string['features'][0]['properties']['cov_matrix']
    except:
        #'cov_matrix' : {'XX' : '1.70816', 'XY' : '0.270061', 'XZ' : '0.0988157', 'YY' : '2.74951', 'YZ' : '0.48637', 'ZZ' : '8.04068'},
        # From Zante
        cov_matrix = dict()
        cov_matrix['XX'] = 13.9478
        cov_matrix['XY'] = 4.8435
        cov_matrix['XZ'] = 0.2781
        cov_matrix['YY'] = 18.1122
        cov_matrix['YZ'] = 0.1786
        cov_matrix['ZZ'] = 10.2529

    pos_Sigma = cov_matrix.copy() #json_string['features'][0]['properties']['cov_matrix']

    cov_matrix['XX'] = float(cov_matrix['XX'])
    cov_matrix['XY'] = float(cov_matrix['XY'])
    cov_matrix['XZ'] = float(cov_matrix['XZ'])
    cov_matrix['YY'] = float(cov_matrix['YY'])
    cov_matrix['YZ'] = float(cov_matrix['YZ'])
    cov_matrix['ZZ'] = float(cov_matrix['ZZ'])

    pos_Sigma['XX']  = float(pos_Sigma['XX']) * 1e6
    pos_Sigma['XY']  = float(pos_Sigma['XY']) * 1e6
    pos_Sigma['XZ']  = float(pos_Sigma['XZ']) * 1e6
    pos_Sigma['YY']  = float(pos_Sigma['YY']) * 1e6
    pos_Sigma['YZ']  = float(pos_Sigma['YZ']) * 1e6
    pos_Sigma['ZZ']  = float(pos_Sigma['ZZ']) * 1e6

    mag_percentiles['p16']  = float(mag_percentiles['p16'])
    mag_percentiles['p50']  = float(mag_percentiles['p50'])
    mag_percentiles['p84']  = float(mag_percentiles['p84'])

    d['eventid']        = event_id
    d['originid']       = origin_id
    d['version']        = "%03d" % (float(version))
    d['author']         = author
    d['area']           = area
    d['area_geo']       = area_geo
    d['lat']            = lat
    d['lon']            = lon
    d['depth']          = depth
    d['ot']             = OT
    #d['mag']            = mag
    d['mag']            = mag_percentiles['p50']
    d['mag_percentiles'] = mag_percentiles
    d['MagSigma']        = get_magsigma(mag_percentiles = mag_percentiles)
    d['mag_values']     = mag_values
    d['mag_counts']     = mag_counts
    d['type']           = ev_type
    d['mag_type']       = mag_type
    d['ee_utm']         = ee_utm
    d['place']          = place

    d['ct']             = creation_time
    d['ot_year']        = origin_year
    d['ot_month']       = origin_month
    d['ot_day']         = origin_day
#    d['routing_key']    = routing_key

    d['tsunami_message_initial'] = 'initial'
    d['message_number']          = '000'

    d['cov_matrix']      = cov_matrix

    d['pos_Sigma']       = pos_Sigma


    d['ee_PosCovMat_2d'] = np.array([[cov_matrix['XX'], cov_matrix['XY']], \
                                     [cov_matrix['XY'], cov_matrix['YY']]])

    d['PosMean_2d']      = np.array([d['ee_utm'][0], \
                                     d['ee_utm'][1]])

    d['PosCovMat_3d']    = np.array([[cov_matrix['XX'], cov_matrix['XY'], cov_matrix['XZ']], \
                                     [cov_matrix['XY'], cov_matrix['YY'], cov_matrix['YZ']], \
                                     [cov_matrix['XZ'], cov_matrix['YZ'], cov_matrix['ZZ']]])

    d['PosCovMat_3dm']    = d['PosCovMat_3d']*1000000

    d['PosMean_3d']      = np.array([d['ee_utm'][0], \
                                     d['ee_utm'][1], \
                                     d['depth'] * 1000.0])

    d['root_name']       = str(d['ot_year']) + str(d['ot_month']) + str(d['ot_day']) + '_' + \
                           d['area']
                           #str(d['eventid']) + '_' + str(d['version']) + '_' + d['area']

    return d

def get_magsigma(**kwargs):

    mag_percentiles = kwargs.get('mag_percentiles', None)

    p16 = float(mag_percentiles['p16'])
    p84 = float(mag_percentiles['p84'])
    msigma=0.5*(p84-p16)

    return msigma


def event_dict_to_json(**kwargs):

    rabbit_event = kwargs.get('rabbit_event', None)

    # epicenter
    lat = round(float(rabbit_event['lat']),4)
    lon = round(float(rabbit_event['lon']),4)
    dep = round(float(rabbit_event['depth']),4)
    point = [lon, lat, dep]

    mag_p = dict()
    mag_p['p16'] = rabbit_event['mag_percentiles']['p16']
    mag_p['p50'] = rabbit_event['mag_percentiles']['p50']
    mag_p['p84'] = rabbit_event['mag_percentiles']['p84']

    cov_matrix = dict()
    cov_matrix['XX'] = rabbit_event['cov_matrix']['XX']
    cov_matrix['XY'] = rabbit_event['cov_matrix']['XY']
    cov_matrix['XZ'] = rabbit_event['cov_matrix']['XZ']
    cov_matrix['YY'] = rabbit_event['cov_matrix']['YY']
    cov_matrix['YZ'] = rabbit_event['cov_matrix']['YZ']
    cov_matrix['ZZ'] = rabbit_event['cov_matrix']['ZZ']

    properties = dict()
    properties['originId']              = rabbit_event['originid']
    properties['eventId']               = rabbit_event['eventid']
    properties['version']               = int(rabbit_event['version'])
    properties['time']                  = rabbit_event['ot']
    properties['mag']                   = float(rabbit_event['mag'])
    properties['geojson_creationTime']  = ''
    properties['magAuthor']             = rabbit_event['author']
    properties['magType']               = rabbit_event['mag_type']
    properties['type']                  = rabbit_event['type']
    properties['place']                 = rabbit_event['area']
    properties['author']                = rabbit_event['author']
    properties['mag_percentiles']       = mag_p
    properties['cov_matrix']            = cov_matrix
    properties['mag_percentiles']       = mag_p
    properties['cov_matrix']            = cov_matrix

    geojson = {'type': 'FeatureCollection',
               'features': [{'geometry': {'type': 'Point', 'coordinates': point},
               'properties':properties}]}

    return geojson

#unused functions
#
#def json_to_event_dictionary(**kwargs):
#
#    json_dump = kwargs.get('json_dump', None)
#
#    event_dict = dict()
#
#    return event_dict
#
