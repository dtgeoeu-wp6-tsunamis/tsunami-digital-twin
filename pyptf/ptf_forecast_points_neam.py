import os
import sys
import ast
import numpy as np
import json

def pois_to_fcp_levels_all(**kwargs):

    pois_alert_levels  = kwargs.get('pois_alert_levels', None)
    method = kwargs.get('method', None)
    fcp = kwargs.get('fcp', None)
    fcp_dict = kwargs.get('fcp_dict', None)
    pois = kwargs.get('pois', None)

    print(" --> Method for fcp alert level: ", method)

    n_fcp  = len(fcp.keys())
    n_type = len(pois_alert_levels.keys())
    # n_pois = len(level[list(level.keys())[0]])
    # pois_idx = pois['pois_index']
    #pois_labels = [pois['pois_labels'][i] for i in pois_idx]
    pois_labels = list(pois['pois_labels'])

    fcp_type_tmp  = np.zeros((n_fcp, n_type))
    fcp_type  = dict()
    fcp_name  = []
    fcp_pois  = []
    fcp_coordinates = []
    fcp_id  = []

    for i, key in enumerate(fcp):

        # get fcp name
        fcp_name.append(key)
        fcp_id.append(fcp_dict[key])
        # get coordinates - [lon,lat]) - for each fcp
        fcp_coordinates.append(fcp[key][1])

        # get name of the pois for each fcp
        pois_list = fcp[key][0]
        fcp_pois.append(pois_list)

        # look for indx of pois to keep
        try:
            # idx = sortedindex(pois['pois_labels'], pois_list)
            idx = sortedindex(pois_labels, pois_list)
        except:
            idx = []

        if not idx:
            #fcp_type_tmp[i,:] = 1
            print('The FCP {0} is not associated to any used POI.'.format(key))

        else:
            for j, k in enumerate(pois_alert_levels.keys()):
                fcp_type_tmp[i,j] = do_method(pois_alert_levels[k][idx].transpose(), method)

    for i, key in enumerate(pois_alert_levels.keys()):
        fcp_type[key] = fcp_type_tmp[:,i]

    return fcp_type,fcp_name, fcp_coordinates, fcp_id


def do_method(vec, method):

    if method['rule'] == 'max':
        fcp_t = np.max(vec)
    elif method['rule'] == 'min':
        fcp_t = np.min(vec)
    elif method['rule'] == 'mean':
        fcp_t = np.rint(np.mean(vec))
    elif method['rule'] == 'mean_low':
        fcp_t = np.floor(np.mean(vec))
    elif method['rule'] == 'mean_top':
        fcp_t = np.ceil(np.mean(vec))
    else:
        fcp_t = 0

#    if(fcp_t == 0):
#        fcpl = 'unknown'
#    if(fcp_t == 1):
#        fcpl = 'information'
#    if(fcp_t == 2):
#        fcpl = 'advisory'
#    if(fcp_t == 3):
#        fcpl = 'watch'

    return fcp_t#, fcpl


def sortedindex(lst,find):
    find.sort()
    indices  = []
    start = 0
    for item in find:
        start = lst.index(item,start)
        indices.append(start)
    return indices



