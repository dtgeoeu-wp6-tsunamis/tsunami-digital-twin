import os
import sys
import ast
from geopy.distance import geodesic
import configparser
import numpy as np
from ptf_preload import load_intensity_thresholds
from ptf_matrix import define_level_alerts_decision_matrix
from ptf_matrix import set_points_alert_level
from ptf_forecast_points_neam import pois_to_fcp_levels_all
# import utm
# import collections


def main(**kwargs):


    cfg_file = kwargs.get('cfg_file', None)                       # Configuration file
    workflow_dict = kwargs.get('workflow_dict', None)             # workflow dictionary
    event_dict    = kwargs.get('event_dict', None)              # event dictionary

    Config = configparser.RawConfigParser()
    Config.read(cfg_file)

    workdir = workflow_dict['workdir']
    percentiles = workflow_dict['percentiles']
    n_percentiles = len(percentiles)

    # loading intensity thresholds (mih)
    thresholds, intensity_measure = load_intensity_thresholds(cfg = Config)

    pois = np.load(os.path.join(workdir, workflow_dict['pois']), allow_pickle=True).item()
    # n_pois, _ = pois['pois_coords'].shape
    pois_idx = pois['pois_index']
    n_pois = len(pois_idx)

    run_up_yn  = ast.literal_eval(Config.get('Settings', 'run_up_yn'))
    intensity  = ast.literal_eval(Config.get('alert_levels', 'intensity'))
    if run_up_yn:
        intensity_values = ast.literal_eval(intensity['run_up_yes'])
    else:
        intensity_values = ast.literal_eval(intensity['run_up_no'])

    intensity_values.append(sys.float_info.max)
    intensity_values = np.array(intensity_values)

    # read 
    level_values = np.load(os.path.join(workdir, workflow_dict['step3_hc_perc'] + '.npy'), allow_pickle=True).item()

    print(" --> Alert levels at POIs based on PTF")

    pois_alert_levels = dict()
    for key in level_values.keys():
        pois_tmp = np.zeros((n_pois)).astype('int')
        for i in range(n_pois):
            pois_tmp[i] = np.where(intensity_values >= level_values[key][i])[0][0]
            if pois_tmp[i] == 0:
                pois_tmp[i] = 1

        pois_alert_levels[key] = pois_tmp

    # np.set_printoptions(threshold=sys.maxsize)
    # print(pois_alert_levels['mean'])
         
    print(" --> Alert levels at POIs based on Decision Matrix")
    ev_lon = event_dict['lon']
    ev_lat = event_dict['lat']
    event = (ev_lat, ev_lon)
    # distances = [ geodesic(event, (pois['pois_coords'][i,1], pois['pois_coords'][i,0])).km for i in range(n_pois) ]
    distances = [ geodesic(event, (pois['pois_coords'][i,1], pois['pois_coords'][i,0])).km for i in pois_idx ]
    event_parameters = define_level_alerts_decision_matrix(event_parameters=event_dict, cfg=Config)
    pois_alert_levels['matrix'] = set_points_alert_level(distances=distances, event_parameters=event_parameters, cfg=Config)

    np.save(os.path.join(workflow_dict['workdir'], workflow_dict['step4_alert_levels_POI']), pois_alert_levels, allow_pickle=True)

    # print(" --> best estimation")
    # levels = get_alerts_best(level = levels, hazard = hazard, pois = pois, intensity = intensity_values)
    
    # METTERE OPZIONALE PER MESSAGGI AREA NEAM????????????????
    print(" --> POIs to forecast points")
    fcp_alert_levels = pois_to_fcp_levels_all(level = pois_alert_levels, cfg = Config, pois = pois)
    np.save(os.path.join(workflow_dict['workdir'], workflow_dict['step4_alert_levels_FCP']), fcp_alert_levels, allow_pickle=True)

    # print(" --> Set official alert level at fcp for pyPTF")
    # levels = set_fcp_alert_ptf(level = levels, cfg = Config)



if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
