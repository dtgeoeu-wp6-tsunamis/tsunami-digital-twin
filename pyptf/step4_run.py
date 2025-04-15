import os
import sys
import ast
import numpy as np

from pyptf.ptf_matrix import set_alert_levels_matrix
from pyptf.ptf_forecast_points_neam import pois_to_fcp_levels_all


def main(**kwargs):

    workflow_dict = kwargs.get('workflow_dict', None)             # workflow dictionary
    event_dict = kwargs.get('event_dict', None)              # event dictionary
    pois_d  = kwargs.get('pois_d', None)
    fcp  = kwargs.get('fcp', None)
    fcp_dict = kwargs.get('fcp_dict', None)
    level_values  = kwargs.get('hc_d', None)
    logger = kwargs.get('logger', None)

    workdir = workflow_dict['workdir']

    pois_idx = pois_d['pois_index']
    n_pois = len(pois_idx)

    al_run_up_yn = workflow_dict['al_run_up_yn']
    al_thresholds = workflow_dict['al_thresholds']
    if al_run_up_yn:
        intensity_values = ast.literal_eval(al_thresholds['run_up_yes'])
    else:
        intensity_values = ast.literal_eval(al_thresholds['run_up_no'])

    intensity_values.append(sys.float_info.max)
    intensity_values = np.array(intensity_values)

    # read hc 
    if level_values is None:
        file_hc_perc = os.path.join(workdir, workflow_dict['step3_hc_perc'] + '.npy')
        try:
            level_values = np.load(file_hc_perc, allow_pickle=True).item()
        except:
            raise Exception(f"Error reading file: {file_hc_perc}")

    logger.info(" --> Alert levels at POIs based on PTF")

    pois_alert_levels = dict()

    for key in level_values.keys():
        pois_tmp = np.zeros((n_pois)).astype('int')
        for i in range(n_pois):
            pois_tmp[i] = np.where(intensity_values >= level_values[key][i])[0][0]
            if pois_tmp[i] == 0:
                pois_tmp[i] = 1

        pois_alert_levels[key] = pois_tmp

    # np.set_printoptions(threshold=sys.maxsize)
    # logger.info(pois_alert_levels['mean'])
    if  event_dict['area'] == 'cat_area':
        logger.info(" --> Alert levels at POIs based on Decision Matrix")
        pois_alert_levels = set_alert_levels_matrix(workflow_dict = workflow_dict,
                                                    event_dict = event_dict,
                                                    pois_d = pois_d,
                                                    n_pois = n_pois,
                                                    pois_alert_levels = pois_alert_levels)

    logger.info(" --> POIs to forecast points")
    fcp_alert_levels, fcp_names, fcp_coordinates, fcp_ids = pois_to_fcp_levels_all(pois_alert_levels = pois_alert_levels, 
                                                                                   method = workflow_dict['al_fcp_method'],
                                                                                   fcp = fcp,
                                                                                   fcp_dict = fcp_dict,
                                                                                   pois = pois_d)

    # logger.info(" --> Set official alert level at fcp for pyPTF")
    # levels = set_fcp_alert_ptf(level = levels, cfg = Config)
    
    return pois_alert_levels, fcp_alert_levels, fcp_names, fcp_coordinates, fcp_ids



if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
