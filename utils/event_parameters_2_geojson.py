#!/opt/python/3.8/bin/python
#!/usr/bin/env python

import os
import json
import sys
import time
import argparse
import socket
import getpass
import datetime
import numpy as np
import reverse_geocoder as rg
import pathlib


from geojson import Feature, Point, FeatureCollection
from obspy.core import UTCDateTime


class ptf_json_event:

    def __init__(self, args):

        # Create class attributes from argsparse
        for arg in vars(args):
            res_name = getattr(args, arg)
            setattr(self, arg, res_name)



    def geocoder_area2(self):

        coordinates = (self.lat, self.lon)
        results = rg.search(coordinates)
        #self.area = results[0]['cc'] + '_' + results[0]['name'] + '_' + results[0]['admin1']
        self.area = (results[0]['name'] + '_' + results[0]['admin1']).replace(' ', '_')

    def get_author(self):

        self.author = getpass.getuser() + "@" + socket.getfqdn()

    def None_2_defaults(self):

        if self.origin_id == None:
            self.origin_id = ot_2_ids(self)

        if self.event_id == None:
            self.event_id = ot_2_ids(self)

        if self.version == None:
            self.version = ot_2_ids(self)[-3:]

        if self.author == None:
            self.get_author()

        if self.area == None:
            self.geocoder_area2()

        if self.c_time == None:
            self.c_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

        if self.mag_p16 == None:
            self.mag_p16 = ("%.3f" % (float(self.mag) - float(self.mag_unc)))

        if self.mag_p50 == None:
            self.mag_p50 = ("%.3f" % (float(self.mag)))

        if self.mag_p84 == None:
            self.mag_p84 = ("%.3f" % (float(self.mag) + float(self.mag_unc)))

        if self.out_file == None:
            self.out_file = '_'.join([self.event_id, self.origin_id, self.version]) + '_event.json'


def ptf_event_2_geojson(event):


    ma_p = {'p16': event.mag_p16, 'p50': event.mag_p50, 'p84': event.mag_p84}
    co_m = {'XX': event.cm_xx, 'XY': event.cm_xy, 'XZ': event.cm_xz, 'YY': event.cm_yy, 'YZ': event.cm_yz, 'ZZ': event.cm_zz}
    prop = {'mag_percentiles' : ma_p,
            'cov_matrix' : co_m,
            'originId': event.origin_id,
            'eventId': event.event_id,
            'version': event.version,
            'time': event.ot,
            'mag': event.mag,
            'geojson_creationTime': event.c_time,
            'magAuthor': event.author,
            'magType': event.mag_type,
            'type': 'earthquake',
            'place': event.area,
            'author': event.author}


    my_feature = Feature(geometry=Point((float(event.lon),float(event.lat),float(event.depth))), properties=prop)
    feature_collection = FeatureCollection([my_feature])

    return feature_collection

def ot_2_ids(event):

    tmp = UTCDateTime(event.ot)

    return str(int(time.mktime(tmp.timetuple())))


def parse_stdin():

    description = 'pyPTF event gejoson creator'
    example     = 'Example\n' + sys.argv[0] + ' --ot 2000-01-01T12:00:00.0 --lat 23.24 --lon 34.67\n\n'

    parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter, epilog = example)

    parser.add_argument('--ot',                default = None,        help = 'Origin Time format YYYY-MM-ddThh-mm-ss.ms. Default = None')
    parser.add_argument('--origin_id',         default = None,        help = 'Origin ID. If None derived from OT. Default = None')
    parser.add_argument('--event_id',          default = None,        help = 'Event ID. If None derived from OT. Default = None')
    parser.add_argument('--version',           default = None,        help = 'Version. If None derived from OT. 3 digits. Default = event')
    parser.add_argument('--lat',               default = None,        help = 'Latitude. Default = None')
    parser.add_argument('--lon',               default = None,        help = 'Longitude. Default = None')
    parser.add_argument('--depth',             default = None,        help = 'Depth. Default = None')
    parser.add_argument('--mag',               default = None,        help = 'Magnitude. Default = None')
    parser.add_argument('--mag_type',          default = 'Mwp',       help = 'Magnitude Type. Default = Mwp')
    parser.add_argument('--area',              default = None,        help = 'Region. If None derived from epicenter. Default = None')
    parser.add_argument('--author',            default = None,        help = 'Author. If None user@host. Default = None')
    parser.add_argument('--c_time',            default = None,        help = 'Author. If None user@host. Default = Now')
    parser.add_argument('--type',              default = None,        help = 'earthquake. Default = earthquake')
    parser.add_argument('--mag_p16',           default = None,        help = 'Mag Percentiles P16. If None Mag-mag_unc. Default = None')
    parser.add_argument('--mag_p50',           default = None,        help = 'Mag Percentiles P50. If None Mag. Default = None')
    parser.add_argument('--mag_p84',           default = None,        help = 'Mag Percentiles P84. If None Mag+mag_unc. Default = None')
    parser.add_argument('--cm_xx',             default = '2.0',       help = 'Covariant Matrix Element XX. Default = 2.0')
    parser.add_argument('--cm_xy',             default = '0.5',       help = 'Covariant Matrix Element XX. Default = 2.0')
    parser.add_argument('--cm_xz',             default = '0.1',       help = 'Covariant Matrix Element XX. Default = 2.0')
    parser.add_argument('--cm_yy',             default = '3.0',       help = 'Covariant Matrix Element XX. Default = 2.0')
    parser.add_argument('--cm_yz',             default = '0.4',       help = 'Covariant Matrix Element XX. Default = 2.0')
    parser.add_argument('--cm_zz',             default = '10.0',      help = 'Covariant Matrix Element XX. Default = 2.0')
    parser.add_argument('--mag_unc',           default = '0.2',       help = 'Covariant Matrix Element XX. Default = 0.2')
    parser.add_argument('--out_path',          default = '.',         help = 'Target writing path for json file. Default = ./')
    parser.add_argument('--out_file',          default = None,        help = 'Target json file name. If None: = event_id_origin_id_version_event.json')

    args = parser.parse_args()

    if not sys.argv[1:]:
           print ("Use -h or --help option for Help")
           sys.exit(0)

    if args.lat == None or args.lon == None or args.depth == None or args.ot == None:
        print('You must provide at least latitude, longithde, depth and origintime')
        sys.exit(0)

    return args

def main():

    # Parse stdin
    args    = parse_stdin()

    # Create my_data object using args values
    my_data = ptf_json_event(args)
    my_data.None_2_defaults()

    # Convert my_data object to geojson
    geojson = ptf_event_2_geojson(my_data)

    # Check if out_path exists, if not create
    pathlib.Path(my_data.out_path).mkdir(parents=True, exist_ok=True)

    with open(my_data.out_path + os.sep + my_data.out_file, "w") as write_file:
        json.dump(geojson, write_file, indent=4)
    print(write_file.name)

if __name__ == "__main__":
    main()