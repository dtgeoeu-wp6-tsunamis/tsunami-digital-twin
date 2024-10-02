#!/usr/bin/env python

import os
import sys
import subprocess
import netCDF4 as nc
import numpy as np
import reverse_geocoder as rg

def geocoder_area2(**kwargs):

    lat = kwargs.get('lat', None)
    lon = kwargs.get('lon', None)


    coordinates = (lat, lon)
    #print("ACTION:", end = ' ')
    results = rg.search(coordinates)

    name = results[0]['cc'] + '_' + results[0]['name'] + '_' + results[0]['admin1']
    name = name.replace(' ', '-')

    return name

def get_path(**kwargs):

    pbs = kwargs.get('pbs', None)
    pps = kwargs.get('pps', None)
    typ = kwargs.get('typ', None)
    siz = kwargs.get('siz', None)

    if(typ == 'bs'):
        out = pbs + os.sep + 'bs_' + siz

    if (typ == 'ps'):
        out = pbs + os.sep + 'ps_' + siz

    return out

def data_lists(file, typ):


    data1 = nc.Dataset(file)
    mag_list = np.array(data1.variables['mag'][:])
    ids_list = data1.variables['id'][:]
    hypo_list = np.array(data1.variables['hypo'][:])
    lon_list = hypo_list[0, :]
    lat_list = hypo_list[1, :]
    dep_list = hypo_list[2, :]

    new_ids = []
    for i in range(len(ids_list)):
        u = str("%2s%010d" % (typ, int(ids_list[i])))
        new_ids.append(u)

    return new_ids, mag_list, lat_list, lon_list, dep_list



#nc_files = ['Mag_small6_BS.nc', 'Mag_small6_PS.nc', 'Mag_big6_BS.nc', 'Mag_big6_PS.nc']
nc_files = ['Mag_small6_BS.nc']
nc_types = ['bs']
nc_sizes = ['small6']
nc_path  = '/data/users/fabrizio/tests_ptf/netcdf_files'
jbs_path = '/data/users/fabrizio/tests_ptf/events/infiles/json_bs_30x30x30'
jps_path = '/data/users/fabrizio/tests_ptf/events/infiles/json_ps_30x30x30'
jsn_creator = '/home/fabrizio/gitwork/pyptf/utils/event_parameters_2_geojson.py'



### Definitions to complete the json files
# origin_id = o_id
# event_id  = e_id
ot          = '2000-01-01T00:00:00.0'
mag_unc     = 0.15
version     = '000'
author      = 'fabrizio_romano'
# out_path
# out_file


for i in range(len(nc_files)):

    ids, mag, lat, lon, dep = data_lists(nc_path + os.sep + nc_files[i], nc_types[i])
    print(mag[0:20])
    sys.exit()

    out_path = get_path(pbs=jbs_path, pps=jps_path, typ=nc_types[i], siz=nc_sizes[i])

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print(i,nc_files[i], out_path)

    for j in range(len(ids)):

        origin_id = 'o' + ids[j]
        event_id  = 'e' + ids[j]
        out_file  = nc_types[i] + '_' + nc_sizes[i] + '_' + \
                    event_id + '_' + origin_id + '_' + version + '_event.json'
        area      = geocoder_area2(lat=lat[j], lon=lon[j])



        #print(origin_id, event_id, out_file, area)

        #print(jsn_creator + ' --mag ' + str(mag[i]) + ' --lat ' +str(lat[i])+ ' --lon '+ str(lon[i])+ ' --depth ' +str(dep[i])+ ' --mag_unc '+ str(mag_unc) + ' --ot ' + ot)

        subprocess.run([jsn_creator, '--mag', str(mag[j]), '--lat', str(lat[j]), '--lon', str(lon[j]), '--depth', str(dep[j]),
                       '--ot', ot, '--mag_unc', str(mag_unc), '--area', area, '--event_id', event_id, '--origin_id', origin_id,
                        '--author', author, '--version', version, '--out_path', out_path, '--out_file', out_file])

        sys.exit()



#file1 = '/data/users/fabrizio/tests_ptf/netcdf_files/Mag_small6_BS.nc'
#ids, mag, lat, lon, dep = data_lists(file1, 'bs')

