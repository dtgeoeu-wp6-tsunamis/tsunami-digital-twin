import os
import utm
import sys
import math
import socket
import glob
#import shutil
#import numbers
import ast
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import numpy as np
import numpy.matlib as npm
from numba import jit
from pathlib import Path
from scipy.stats import norm
from shapely import geometry
import pygmt


def create_workflow_dict(**kwargs):

    args   = kwargs.get('args', None)
    Config = kwargs.get('Config', None)
    logger = kwargs.get('logger', None)

    wd = dict()

    inpjson = args.input_workflow
    f = open(inpjson, 'r').read()
    jsn_object = eval(f)

    wd['sigma'] = float(args.sigma)
    wd['sigma_inn'] = wd['sigma']
    wd['sigma_out'] = wd['sigma'] + 0.5
    wd['negligible_prob'] = 2*norm.cdf(-1. * wd['sigma'])

    wd['hazard_mode'] = args.hazard_mode
    wd['ptf_version'] = args.ptf_version
    wd['mag_sigma_val'] = args.mag_sigma_val
    wd['type_df'] = args.type_df
    wd['percentiles'] = np.array(args.percentiles)*0.01

    wd['step1'] = jsn_object['STEPS']['step1']    # ENSEMBLE DEFINITION
    wd['step2'] = jsn_object['STEPS']['step2']    # TSUNAMI SIMULATIONS
    wd['step3'] = jsn_object['STEPS']['step3']    # HAZARD AGGREGATION
    wd['step4'] = jsn_object['STEPS']['step4']    # ALERT LEVELS
    wd['step5'] = jsn_object['STEPS']['step5']    # VISUALIZATION

    wd['domain'] = jsn_object['SETTINGS']['domain']
    wd['tsu_sim'] = jsn_object['SETTINGS']['tsu_sim']
    wd['UCmode'] = jsn_object['SETTINGS']['UCmode']
    wd['TEWmode'] = jsn_object['SETTINGS']['TEWmode']
    wd['sim_domain'] = jsn_object['SETTINGS']['sim_domain']   # regional or local scale

    wd['workpath'] = Config.get('save_ptf', 'save_main_path')
    wd['pois'] = Config.get('save_ptf', 'pois')
    wd['step1_prob_BS_root'] = Config.get('save_ptf', 'step1_prob_BS')
    wd['step1_prob_PS_root'] = Config.get('save_ptf', 'step1_prob_PS')
    wd['step1_list_BS_root'] = Config.get('save_ptf', 'step1_list_BS')
    wd['step1_list_PS_root'] = Config.get('save_ptf', 'step1_list_PS')
    wd['step1_prob_SBS_root'] = Config.get('global', 'step1_prob_SBS')
    wd['step1_list_SBS_root'] = Config.get('global', 'step1_list_SBS')

    wd['step2_hmax_pre_BS_root'] = Config.get('save_ptf', 'step2_hmax_pre_BS')
    wd['step2_hmax_pre_PS_root'] = Config.get('save_ptf', 'step2_hmax_pre_PS')
    wd['step2_hmax_sim_BS_root'] = Config.get('save_ptf', 'step2_hmax_sim_BS')
    wd['step2_hmax_sim_PS_root'] = Config.get('save_ptf', 'step2_hmax_sim_PS')
    wd['step2_hmax_sim_SBS_root'] = Config.get('global', 'step2_hmax_sim_SBS')
    wd['step2_dir_sim_BS_root'] = Config.get('save_ptf', 'step2_dir_sim_BS')
    wd['step2_dir_sim_PS_root'] = Config.get('save_ptf', 'step2_dir_sim_PS')
    wd['step2_dir_sim_SBS_root'] = Config.get('global', 'step2_dir_sim_SBS')
    wd['step2_dir_log_BS_root'] = Config.get('save_ptf', 'step2_dir_log_BS')
    wd['step2_dir_log_PS_root'] = Config.get('save_ptf', 'step2_dir_log_PS')
    wd['step2_dir_log_SBS_root'] = Config.get('global', 'step2_dir_log_SBS')
    wd['step2_log_failed_BS_root'] = Config.get('save_ptf', 'step2_log_failed_BS')
    wd['step2_log_failed_PS_root'] = Config.get('save_ptf', 'step2_log_failed_PS')
    wd['step2_log_failed_SBS_root'] = Config.get('global', 'step2_log_failed_SBS')
    wd['step2_list_simdir_BS_root']  = Config.get('save_ptf', 'step2_list_simdir_BS') 
    wd['step2_list_simdir_PS_root']  = Config.get('save_ptf', 'step2_list_simdir_PS') 
    wd['step2_list_simdir_SBS_root']  = Config.get('global', 'step2_list_simdir_SBS') 
    wd['step2_list_all_sims_BS']  = Config.get('save_ptf', 'step2_list_all_sims_BS') 
    wd['step2_list_all_sims_PS']  = Config.get('save_ptf', 'step2_list_all_sims_PS') 
    wd['step2_list_all_sims_SBS']  = Config.get('global', 'step2_list_all_sims_SBS') 
    wd['step2_newsimulations_BS_root']  = Config.get('save_ptf', 'step2_newsimulations_BS') 
    wd['step2_newsimulations_PS_root']  = Config.get('save_ptf', 'step2_newsimulations_PS') 
    wd['step2_newsimulations_SBS_root']  = Config.get('global', 'step2_newsimulations_SBS') 
    wd['step2_ts_file']  = Config.get('save_ptf', 'step2_ts_file') 

    wd['step3_hc_root'] = Config.get('save_ptf', 'step3_hc')
    wd['step3_hc_perc_root'] = Config.get('save_ptf', 'step3_hc_perc')
    wd['step3_hazard_pdf_root'] = Config.get('save_ptf', 'step3_hazard_pdf')
    wd['step4_alert_levels_FCP_root'] = Config.get('save_ptf', 'step4_alert_levels_FCP')
    wd['step4_alert_levels_POI_root'] = Config.get('save_ptf', 'step4_alert_levels_POI')
    wd['step5_figures_root'] = Config.get('save_ptf', 'step5_figures')
    wd['step5_alert_levels_root'] = Config.get('save_ptf', 'step5_alert_levels')
    wd['step5_hazard_maps_root'] = Config.get('save_ptf', 'step5_hazard_maps')
    wd['step5_hazard_curves_root'] = Config.get('save_ptf', 'step5_hazard_curves')
    wd['workflow_dictionary_root'] = Config.get('save_ptf', 'workflow_dictionary')
    wd['event_dictionary_root'] = Config.get('save_ptf', 'event_dictionary')
    wd['status_file_root'] = Config.get('save_ptf', 'status_file')

    wd['wf_path'] = Config.get('pyptf', 'wf_path')
    wd['inpdir'] = Config.get('pyptf', 'data_path')
    # wd['event_path'] = Config.get('pyptf', 'event_path')

    wd['bs_empty_regions'] = np.array(ast.literal_eval(Config.get('ScenariosList', 'bs_empty'))).astype('float64')

    # for global version
    wd['dist_from_epi'] = float(Config.get('global', 'dist_from_epi'))
    wd['magnitude_values'] = np.array(ast.literal_eval(Config.get('global', 'magnitude_values')))
    wd['position_step'] = float(Config.get('global', 'position_step'))
    wd['depth_step'] = float(Config.get('global', 'depth_step'))
    wd['strike_step'] = float(Config.get('global', 'strike_step'))
    wd['strike_sigma'] = float(Config.get('global', 'strike_sigma'))
    wd['dip_step'] = float(Config.get('global', 'dip_step'))
    wd['dip_sigma'] = float(Config.get('global', 'dip_sigma'))
    wd['rake_step'] = float(Config.get('global', 'rake_step'))
    wd['rake_sigma'] = float(Config.get('global', 'rake_sigma'))
    wd['rigidity'] = float(Config.get('global', 'rigidity'))
    wd['grid_global'] = Config.get('global', 'grid_global')

    # for sampling
    wd['sampling_mode'] = jsn_object['SETTINGS']['sampling mode']
    wd['sampling_type'] = jsn_object['SETTINGS']['sampling type']
    wd['number_of_scenarios'] = jsn_object['SETTINGS']['number of scenarios']

    # for tsunami simulations
    wd['ptf_measure_type'] = Config.get('Settings', 'ptf_measure_type')
    # wd['regional_pois_filename'] = Config.get('tsu_sims', 'regional_pois')
    wd['bathy_folder'] = Config.get('tsu_sims', 'bathy_folder')
    wd['regional_bathy_file'] = Config.get('tsu_sims', 'regional_bathy_file')
    # wd['regional_pois_depth'] = Config.get('tsu_sims', 'regional_pois_depth')
    wd['ps_inicond_med'] = Config.get('tsu_sims', 'ps_inicond_med')
    wd['n_digits'] = int(Config.get('tsu_sims', 'n_digits'))
    wd['propagation'] = jsn_object['SETTINGS']['propagation']
    wd['sim_postproc'] = Config.get('tsu_sims', 'sim_postproc')
    wd['final_postproc'] = Config.get('tsu_sims', 'final_postproc')
    wd['BS_parfile_tmp'] = Config.get('tsu_sims', 'BS_parfile_tmp')
    wd['PS_parfile_tmp'] = Config.get('tsu_sims', 'PS_parfile_tmp')
    wd['SBS_parfile_tmp'] = Config.get('global', 'SBS_parfile_tmp')
    wd['bathy_filename'] = Config.get('tsu_sims', 'bathy_filename')
    wd['depth_filename'] = Config.get('tsu_sims', 'depth_filename')
    wd['run_sim_filename'] = Config.get('tsu_sims', 'run_sim_filename')
    wd['run_post_filename'] = Config.get('tsu_sims', 'run_post_filename')

    # for computing the PDF
    wd['compute_pdf'] = jsn_object['SETTINGS']['compute_pdf']

    # for hazard curves
    wd['logn_sigma'] = float(args.logn_sigma)
    wd['save_nc'] = jsn_object['SETTINGS']['save_nc']
    
    # for alert levels (al)
    wd['al_run_up_yn']  = ast.literal_eval(Config.get('alert_levels', 'run_up_yn'))
    wd['al_thresholds']  = ast.literal_eval(Config.get('alert_levels', 'thresholds'))
    wd['min_mag'] = float(Config.get('matrix', 'min_mag_for_message'))
    wd['local_distance']  = Config.getfloat('matrix', 'local_distance')
    wd['regional_distance'] = Config.getfloat('matrix', 'regional_distance')
    wd['al_fcp_method'] = ast.literal_eval(Config.get('alert_levels', 'fcp_method'))
    wd['pois_to_fcp'] = Config.get('Files', 'pois_to_fcp')

    # local host is where the workflow is executed
    wd['local_host'] = socket.gethostname()
    if 'leonardo' in wd['local_host']:
        wd['local_host'] = 'leonardo'
    # wd['local_host'] = wd['local_host_full'].split('.')[0]
    # fullname=subprocess.run("domainname -A | awk '{print $1}' ", shell=True, stdout=subprocess.PIPE)
    # fullname.stdout.decode()
    try:
        wd['local_user'] = os.getlogin()
    except:
        wd['local_user'] = 'unknown'
    wd['hpc_cluster'] = jsn_object['HPC CLUSTER']['cluster']  # hpc cluster for simulations
    if wd['hpc_cluster'] != wd['local_host']:
        wd['remote_workpath'] = Config.get('save_ptf', 'save_main_path_remote')
        wd['remote_inpdir'] = Config.get('pyptf', 'data_path_remote')
        wd['remote_user'] = jsn_object['HPC CLUSTER']['username']  # username on the cluster

    #wd['passwd'] = jsn_object['HPC CLUSTER']['m100_passwd'] 
    wd['envfile'] = jsn_object['HPC CLUSTER']['env_file'] 
    wd['account'] = jsn_object['HPC CLUSTER']['leonardo_account'] 
    wd['partition'] = jsn_object['HPC CLUSTER']['leonardo_partition']   # Partition for the resource allocation
    wd['quality'] = jsn_object['HPC CLUSTER']['leonardo_quality']       # Quality of service for the job

    if wd['hpc_cluster'] == 'leonardo':
        wd['run_sim_tmp'] = Config.get('tsu_sims','run_sim_tmp_leonardo')
        wd['run_post_tmp'] = Config.get('tsu_sims','run_post_tmp_leonardo')
        # wd['ps_inicond_med_remote'] = Config.get('tsu_sims', 'ps_inicond_med_leonardo')
        wd['job_status_cmd'] = 'squeue'
    elif wd['hpc_cluster'] == 'mercalli':
        wd['run_sim_tmp'] = Config.get('tsu_sims','run_sim_tmp_mercalli')
        wd['run_post_tmp'] = Config.get('tsu_sims','run_post_tmp_mercalli')
        # wd['ps_inicond_med_remote'] = Config.get('tsu_sims', 'ps_inicond_med_mercalli')
        wd['job_status_cmd'] = 'qstat'
    elif wd['hpc_cluster'] != 'leonardo' and wd['hpc_cluster'] != 'mercalli': 
        logger.warning('HPC cluster for simulations not properly defined')

    return wd


def create_output_names(**kwargs):
    """
    """
    wd = kwargs.get('workflow_dict', None)           # workflow dictionary

    wd['step1_prob_BS'] = wd['step1_prob_BS_root'] + wd['uniqueID'] + '.npy'
    wd['step1_prob_PS'] = wd['step1_prob_PS_root'] + wd['uniqueID'] + '.npy'
    wd['step1_prob_SBS'] = wd['step1_prob_SBS_root'] + wd['uniqueID'] + '.npy'
    wd['step1_list_BS'] = wd['step1_list_BS_root'] + wd['uniqueID'] + '.txt'
    wd['step1_list_PS'] = wd['step1_list_PS_root'] + wd['uniqueID'] + '.txt'
    wd['step1_list_SBS'] = wd['step1_list_SBS_root'] + wd['uniqueID'] + '.txt'
    wd['step2_hmax_pre_BS'] = wd['step2_hmax_pre_BS_root'] + wd['uniqueID'] + '.nc'
    wd['step2_hmax_pre_PS'] = wd['step2_hmax_pre_PS_root'] + wd['uniqueID'] + '.nc'
    wd['step2_hmax_sim_BS'] = wd['step2_hmax_sim_BS_root'] + wd['uniqueID'] + '.nc'
    wd['step2_hmax_sim_PS'] = wd['step2_hmax_sim_PS_root'] + wd['uniqueID'] + '.nc'
    wd['step2_hmax_sim_SBS'] = wd['step2_hmax_sim_SBS_root'] + wd['uniqueID'] + '.nc'
    wd['step2_dir_sim_BS'] = wd['step2_dir_sim_BS_root'] + wd['uniqueID']
    wd['step2_dir_sim_PS'] = wd['step2_dir_sim_PS_root'] + wd['uniqueID']
    wd['step2_dir_sim_SBS'] = wd['step2_dir_sim_SBS_root'] + wd['uniqueID']
    wd['step2_dir_log_BS'] = wd['step2_dir_log_BS_root'] + wd['uniqueID']
    wd['step2_dir_log_PS'] = wd['step2_dir_log_PS_root'] + wd['uniqueID']
    wd['step2_dir_log_SBS'] = wd['step2_dir_log_SBS_root'] + wd['uniqueID']
    wd['step2_log_failed_BS'] = wd['step2_log_failed_BS_root'] + wd['uniqueID'] + '.log'
    wd['step2_log_failed_PS'] = wd['step2_log_failed_PS_root'] + wd['uniqueID'] + '.log'
    wd['step2_log_failed_SBS'] = wd['step2_log_failed_SBS_root'] + wd['uniqueID'] + '.log'
    wd['step2_list_simdir_BS'] = wd['step2_list_simdir_BS_root'] +  wd['uniqueID'] + '.txt'
    wd['step2_list_simdir_PS'] = wd['step2_list_simdir_PS_root'] +  wd['uniqueID'] + '.txt'
    wd['step2_list_simdir_SBS'] = wd['step2_list_simdir_SBS_root'] +  wd['uniqueID'] + '.txt'
    wd['step2_newsimulations_BS'] = wd['step2_newsimulations_BS_root'] + wd['uniqueID'] + '.txt'
    wd['step2_newsimulations_PS'] = wd['step2_newsimulations_PS_root'] + wd['uniqueID'] + '.txt'
    wd['step2_newsimulations_SBS'] = wd['step2_newsimulations_SBS_root'] + wd['uniqueID'] + '.txt'
    wd['step3_hc'] = wd['step3_hc_root'] + wd['uniqueID']
    wd['step3_hc_perc'] = wd['step3_hc_perc_root'] + wd['uniqueID']
    wd['step3_hazard_pdf'] = wd['step3_hazard_pdf_root'] + wd['uniqueID'] + '.npy'
    wd['step4_alert_levels_FCP'] = wd['step4_alert_levels_FCP_root'] + wd['uniqueID'] + '.npy'
    wd['step4_alert_levels_POI'] = wd['step4_alert_levels_POI_root'] + wd['uniqueID'] + '.npy'
    wd['step5_figures'] = wd['step5_figures_root'] + wd['uniqueID']
    wd['step5_alert_levels'] = wd['step5_alert_levels_root'] + wd['uniqueID']
    wd['step5_hazard_maps'] = wd['step5_hazard_maps_root'] + wd['uniqueID']
    wd['step5_hazard_curves'] = wd['step5_hazard_curves_root'] + wd['uniqueID']
    wd['workflow_dictionary'] = wd['workflow_dictionary_root'] + wd['uniqueID']
    wd['event_dictionary'] = wd['event_dictionary_root'] + wd['uniqueID']
    wd['status_file'] = wd['status_file_root'] + wd['uniqueID'] + '.txt'


def select_pois_cat_area(**kwargs):

    POIs = kwargs.get('pois_dictionary', None)

    SelectedPOIs = 'Mediterranean'
    
    tmppois = np.array(POIs[SelectedPOIs])
    tmp = np.nonzero(tmppois)[0]
    SelectedPOIs = [POIs['name'][j] for j in tmp]
    Selected_lon = [POIs['lon'][j] for j in tmp] 
    Selected_lat = [POIs['lat'][j] for j in tmp] 
    Selected_dep = [POIs['dep'][j] for j in tmp]

    POIs['selected_pois']    = SelectedPOIs
    POIs['selected_coords']  = np.column_stack((Selected_lon, Selected_lat))
    POIs['selected_depth']   = np.array(Selected_dep)
    POIs['nr_selected_pois'] = len(SelectedPOIs)
    
    return POIs

def create_simulation_grid(**kwargs):
    '''
    '''
    longitude = kwargs.get('longitude', None)
    latitude = kwargs.get('latitude', None)
    distance = kwargs.get('distance', None)
    gridfile_input = kwargs.get('gridfile_input', None)
    gridfile_output = kwargs.get('gridfile_output', None)
    minlon, maxlon, minlat, maxlat = rectangle_around_the_epicenter(longitude, latitude, distance)
    pygmt.grdcut(grid=gridfile_input, region=[minlon, maxlon, minlat, maxlat], outgrid=gridfile_output)


def rectangle_around_the_epicenter(longitude, latitude, distance):
    '''
    define a rectangle around the epicenter
    '''
    minlon = longitude - distance
    maxlon = longitude + distance
    minlat = latitude - distance
    maxlat = latitude + distance
    return minlon, maxlon, minlat, maxlat

def select_pois_from_epicenter(**kwargs):

    pois_d = kwargs.get('pois_d', None)
    longitude = kwargs.get('longitude', None)
    latitude = kwargs.get('latitude', None)
    distance = kwargs.get('distance', None)
    logger = kwargs.get('logger', None)

    #define a rectangle around the epicenter
    minlon, maxlon, minlat, maxlat = rectangle_around_the_epicenter(longitude, latitude, distance)

    # Northwest latlon point
    NW = (maxlat, minlon)
    # Northeast latlon point
    NE = (maxlat, maxlon)
    # Southeast latlon point
    SE = (minlat, maxlon)
    # Southwest latlon point
    SW = (minlat, minlon)
    # Create the rectangle from the latlon coords
    p = geometry.Polygon([NW, NE, SE, SW])

    #select pois inside the rectangle
    list_selected = []
    for ind, poi in enumerate(pois_d['pois_coords']):
        if p.contains(geometry.Point(poi[1], poi[0])):
            list_selected.append(ind)

    pois_d['pois_labels'] = [pois_d['pois_labels'][j] for j in list_selected] 
    pois_d['pois_coords'] = pois_d['pois_coords'][list_selected] 
    pois_d['pois_depth'] = pois_d['pois_depth'][list_selected] 
    pois_d['pois_index'] = [pois_d['pois_index'][j] for j in list_selected] 

    # plot_selected_pois(pois_d, event_dict)
    logger.info(f"Number of selected POIs: {len(pois_d['pois_index'])}")

    return pois_d


def plot_selected_pois(pois_d, event_dict):
    '''
    CONTROLLO SU MAPPA DELLA SELEZIONE DEI POIS
    FORSE DA SPOSTARE IN STEP 5
    '''

    import cartopy
    import matplotlib.pyplot as plt

    proj = cartopy.crs.PlateCarree()
    #cmap = plt.cm.magma_r
    cmap = plt.cm.jet
    ev_lon = event_dict['lon']
    ev_lat = event_dict['lat']
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=cartopy.crs.Mercator())
    coastline = cartopy.feature.GSHHSFeature(scale='low', levels=[1])
    #coastline = cartopy.feature.GSHHSFeature(scale='high', levels=[1])
    ax.add_feature(coastline, edgecolor='#000000', facecolor='#cccccc', linewidth=1)
    ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
    ax.add_feature(cartopy.feature.STATES.with_scale('50m'))
    ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'))
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1,
                        color="#ffffff", alpha=0.5, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    ax.plot(pois_d['pois_coords'][:,0], pois_d['pois_coords'][:,1], markerfacecolor='#ff0000', marker="o", 
                linewidth=0, markeredgecolor="#000000",
                transform=proj, zorder=10)

    ax.plot(ev_lon, ev_lat, linewidth=0, marker='*', markersize=14, 
            markerfacecolor='magenta', markeredgecolor='#000000', 
            transform=proj)

    ax.set_xlabel(r'Longitude ($^\circ$)', fontsize=14)
    ax.set_ylabel(r'Latitude ($^\circ$)', fontsize=14)
    plt.savefig('/work/tonini/test_pyptf/check_selection_pois.png', format='png', dpi=150, bbox_inches='tight')


def check_previous_versions(**kwargs):
    '''
    '''
    wd = kwargs.get('wd', None)
    seismicity_types = kwargs.get('seis_types', None)
    par_bs = kwargs.get('par_bs', None)
    par_ps = kwargs.get('par_ps', None)
    par_sbs = kwargs.get('par_sbs', None)
    logger = kwargs.get('logger', None)

    logger.info('Checking if previous versions of the same event have been already processed')
    step1_outfiles = glob.glob(wd['workdir'] + os.sep + 'step1*.txt')
    step1_previous_run = [f for f in step1_outfiles if not f.endswith(f"{wd['version']}.txt")]

    file_sim = dict()
    
    if (len(step1_previous_run) > 0):
        previous_list = dict()
        previous_list['BS'] = os.path.join(wd['workdir'], wd['step2_list_all_sims_BS'])
        previous_list['PS'] = os.path.join(wd['workdir'], wd['step2_list_all_sims_PS'])
        previous_list['SBS'] = os.path.join(wd['workdir'], wd['step2_list_all_sims_SBS'])

        for seistype in seismicity_types:
            if os.path.getsize(previous_list[seistype][0]) > 0:
               file_sim[seistype] = compare_list_scenarios(wd = wd,
                                                           seistype = seistype,
                                                           previous_list = previous_list[seistype])

    else:
        logger.info('No previous versions of this event found')
 
        par_scen = {'BS': par_bs, 'PS': par_ps, 'SBS': par_sbs}
        for seistype in seismicity_types:
            file_sim[seistype] = create_file_scenarios(par_scen = par_scen[seistype], 
                                                       seistype = seistype, 
                                                       wd = wd)


    return file_sim['BS'], file_sim['PS'], file_sim['SBS']


def create_file_scenarios(**kwargs):
    par = kwargs.get('par_scen', None)
    seistype = kwargs.get('seistype', None)
    wd = kwargs.get('wd', None)        

    ndig = wd['n_digits']

    file_sim = os.path.join(wd['workdir'], wd['step1_list_' + seistype])
    if par is None:
        try: 
            if os.path.getsize(file_sim) > 0:
                with open(file_sim) as f:
                    lines = f.readlines()
            else:
                lines = []
        except:
            raise Exception(f"Error reading file: {file_sim}")
    else:
        if seistype == 'BS' or seistype == 'SBS':
            lines = [" ".join(f"{param:.6f}" for param in params) + "\n" for params in list(par)]
        elif seistype == 'PS':
            lines = [" ".join(params) + "\n" for params in par]
        else:
            raise Exception(f"Seismicity type bad definition {seistype}")

    nscen = len(lines)
    if nscen > 0:

        if (par is not None):
            if seistype == 'BS' or seistype == 'SBS':
                np.savetxt(file_sim, par, delimiter = " ", fmt = '%.6f')
            elif seistype == 'PS':
                np.savetxt(file_sim, par, delimiter = " ", fmt = "%s")
            else:
                raise Exception(f"Seismicity type bad definition {seistype}")
        
        list_scenarios = [wd['uniqueID'] + ' ' + line for line in lines]

        list_simdir = []
        for i in range(nscen):
            scen_id = seistype + '_Scenario' +  str(i+1).zfill(ndig)
            sim_dir = os.path.join(wd['step2_dir_sim_' + seistype], scen_id)
            list_simdir.append(sim_dir)

        filename = os.path.join(wd['workdir'], wd['step2_list_simdir_' + seistype])
        with open(filename,'w') as f:
            f.write('\n'.join(list_simdir) + '\n')              

        filename = os.path.join(wd['workdir'], wd['step2_list_all_sims_' + seistype])
        with open(filename,'w') as f:
            f.write("".join(list_scenarios))         

    else:
        # creating the empty files 
        if (par is not None):
            f_list = open(file_sim, 'w')
            f_list.close()

        # TODO serve creare anche il file vuoto di 'step2_list_simdir_' oppure no?
        filename = os.path.join(wd['workdir'], wd['step2_list_all_sims_' + seistype])
        f = open(filename, 'w')
        f.close()

    return file_sim


def compare_list_scenarios(**kwargs):
    wd = kwargs.get('wd', None)        
    seistype = kwargs.get('seistype', None)
    previous_list = kwargs.get('previous_list', None)

    ndig = wd['n_digits']

    prev_uniqueID = previous_list.split(seistype)[1][1:-4]
    with open(previous_list) as f:
       #try?
       lines = f.readlines()
       lines_prev_list = [x.split()[2:] for x in lines]
       uniqueID_prev_list = [x.split()[0] for x in lines]
       ind_prev_list = [x.split()[1] for x in lines]
    with open(os.path.join(wd['workdir'],wd['step1_list_' + seistype])) as f:
       lines = f.readlines()
       lines_new_list = [x.split()[1:] for x in lines]
       ind_new_list = [x.split()[0] for x in lines]

    new_scenarios = []
    old_scenarios = []
    list_simdir = []
    for line,ind in zip(lines_new_list,ind_new_list):
       if line in lines_prev_list:
          old_scenarios.append(line)
          iold = lines_prev_list.index(line)
          # scen_id = seistype + '_Scenario' + str(lines_prev_list.index(line) + 1).zfill(ndig)
          scen_id = seistype + '_Scenario' + ind_prev_list[iold].zfill(ndig)
          sim_dir = os.path.join(wd['step2_dir_sim_' + seistype + '_root'] + uniqueID_prev_list[iold], scen_id)
          list_simdir.append(sim_dir)
       else:
          new_scenarios.append(ind + ' ' + ' '.join(line))
          scen_id = seistype + '_Scenario' +  ind.zfill(ndig)
          sim_dir = os.path.join(wd['step2_dir_sim_' + seistype], scen_id)
          list_simdir.append(sim_dir)

    print(f'Number of new {seistype} scenarios: {len(new_scenarios)}')
    print(f'Number of {seistype} scenarios matching old ensemble:  {len(old_scenarios)}')
    print('Writing file with new scenarios to run')

    filename = os.path.join(wd['workdir'], wd['step2_list_simdir_' + seistype])
    with open(filename,'w') as f:
       f.write('\n'.join(list_simdir) + '\n')              

    file_newsim = os.path.join(wd['workdir'], wd['step2_newsimulations_' + seistype])
    if len(new_scenarios) > 0:
       with open(file_newsim,'w') as f:
          f.write('\n'.join(new_scenarios) + '\n')              
    else:
       f = open(file_newsim, 'w')
       f.close()

    filename = os.path.join(wd['workdir'], wd['step2_list_all_sims_' + seistype])
    with open(filename,'a') as f:
       f.write('\n'.join([wd['uniqueID'] + ' ' + line for line in new_scenarios]) + '\n')              

    return file_newsim


def find_epicentral_area(**kwargs):

    lon = kwargs.get('lon', None)
    lat = kwargs.get('lat', None)
    cfg = kwargs.get('cfg', None)

    areas  = dict(cfg.items('bounds'))
    #TODO remove the for loop if we are checking only the CAT area
    for name, area in areas.items():
        in_area = ray_tracing_method(float(lon), float(lat), eval(area))
        if in_area:
            return name 

    return None
    # if not in_area:
    #     sys.exit('Epicentre located in an unknown area.')


def conversion_to_utm(**kwargs):

    long = kwargs.get('longTerm', None)
    #pois = kwargs.get('Poi',      None)
    ee   = kwargs.get('event',    None)
    PSBa = kwargs.get('PSBarInfo',    None)

    a = utm.from_latlon(np.array(long['Discretizations']['BS-2_Position']['Val_y']), np.array(long['Discretizations']['BS-2_Position']['Val_x']), ee['ee_utm'][2])
    long['Discretizations']['BS-2_Position']['utm_y']   = a[1]
    long['Discretizations']['BS-2_Position']['utm_x']   = a[0]
    long['Discretizations']['BS-2_Position']['utm_nr']  = a[2]
    long['Discretizations']['BS-2_Position']['utm_reg'] = a[3]

    a = utm.from_latlon(np.array(long['Discretizations']['PS-2_PositionArea']['Val_y']), np.array(long['Discretizations']['PS-2_PositionArea']['Val_x']), ee['ee_utm'][2])
    long['Discretizations']['PS-2_PositionArea']['utm_y']   = a[1]
    long['Discretizations']['PS-2_PositionArea']['utm_x']   = a[0]
    long['Discretizations']['PS-2_PositionArea']['utm_nr']  = a[2]
    long['Discretizations']['PS-2_PositionArea']['utm_reg'] = a[3]

    # a = utm.from_latlon(np.array(pois['lat']), np.array(pois['lon']), ee['ee_utm'][2])
    # pois['utm_lat'] = a[1]
    # pois['utm_lon'] = a[0]
    # pois['utm_nr']  = a[2]
    # pois['utm_reg'] = a[3]

    for i in range(len(PSBa['BarPSperModel'])):
        for j in range(len(PSBa['BarPSperModel'][i])):
            #print(type(PSBa['BarPSperModel'][i][j]['pos_yy']))
            #sys.exit()
            if PSBa['BarPSperModel'][i][j]['pos_yy'].size < 1:
                PSBa['BarPSperModel'][i][j]['utm_pos_lat'] = np.array([])
                PSBa['BarPSperModel'][i][j]['utm_pos_lon'] = np.array([])
                PSBa['BarPSperModel'][i][j]['utm_pos_nr'] = np.array([])
                PSBa['BarPSperModel'][i][j]['utm_pos_reg'] = np.array([])
                pass
            else:
                a = utm.from_latlon(np.array(PSBa['BarPSperModel'][i][j]['pos_yy']), np.array(PSBa['BarPSperModel'][i][j]['pos_xx']), ee['ee_utm'][2])
                PSBa['BarPSperModel'][i][j]['utm_pos_lat'] = a[0]
                PSBa['BarPSperModel'][i][j]['utm_pos_lon'] = a[1]
                PSBa['BarPSperModel'][i][j]['utm_pos_nr']  = a[2]
                PSBa['BarPSperModel'][i][j]['utm_pos_reg'] = a[3]
            #print("+++++++", i,j,PSBa['BarPSperModel'][i][j]['utm_pos_lat'].size, PSBa['BarPSperModel'][i][j]['utm_pos_lon'].size)

    #print(PSBa['BarPSperModel'][2][1]['utm_pos_lat'].size)

    # return long, pois, PSBa
    return long, PSBa


def NormMultiDvec(**kwargs):

    """
    # Here mu and sigma, already inserted into ee dictionary
    # Coordinates in utm
    mu = tmpmu =PosMean_3D = [EarlyEst.lonUTM,EarlyEst.latUTM,EarlyEst.Dep*1.E3]
    Sigma = tmpCOV = EarlyEst.PosCovMat_3D = [EarlyEst.PosSigmaXX EarlyEst.PosSigmaXY EarlyEst.PosSigmaXZ; ...
                         EarlyEst.PosSigmaXY EarlyEst.PosSigmaYY EarlyEst.PosSigmaYZ; ...
                         EarlyEst.PosSigmaXZ EarlyEst.PosSigmaYZ EarlyEst.PosSigmaZZ];
    mu =     np.array([ee['lon'], ee['lat'], ee['depth']*1000.0])
    sigma =  np.array([[ee['cov_matrix']['XX'], ee['cov_matrix']['XY'], ee['cov_matrix']['XZ']], \
                       [ee['cov_matrix']['XY'], ee['cov_matrix']['YY'], ee['cov_matrix']['YZ']], \
                       [ee['cov_matrix']['XZ'], ee['cov_matrix']['YZ'], ee['cov_matrix']['ZZ']]])
    """

    x     = kwargs.get('x', None)
    mu    = kwargs.get('mu', None)
    sigma = kwargs.get('sigma', None)
    ee    = kwargs.get('ee', None)

    n = len(mu)

    #mu = np.reshape(mu,(3,1))
    mu = np.reshape(mu,(n,1))
    t1  = (2 * math.pi)**(-1*len(mu)/2)
    t2  = 1 / math.sqrt(np.linalg.det(sigma))
    #c1  = npm.repmat(mu, 1, np.shape(mu)[0])
    c1  = npm.repmat(mu, 1, len(x))
    c11 = (x - c1.transpose()).transpose()
    c12 = x - c1.transpose()

    d  = np.linalg.lstsq(sigma, c11, rcond=None)[0]
    e = np.dot(c12, d)
    f = np.multiply(-0.5,np.diag(e))
    g = np.exp(f)
    h = t1 * t2 * g

    return h


@jit(nopython=True, cache=True)
def ray_tracing_method(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


def get_focal_mechanism(**kwargs):
    """
    Function modified from get_catalog.py in
    https://gitlab.rm.ingv.it/cat/services/get_catalog_fdsn
     
    """
    ee = kwargs.get('event_parameters', None)
    workdir = kwargs.get('workdir', None)
    logger = kwargs.get('logger', None)

    logger.info('--> Retrieving focal mechanism')

    dt = 5 # minutes
    dmag = 0.3 # magnitude
    dlon = 0.5 # degrees
    dlat = 0.5 # degrees
    #ddepth = # km

    # origin time
    ot = ee['ot'].split('.')[0]
    origin_time = datetime.strptime(ot, "%Y-%m-%dT%H:%M:%S")
    starttime = origin_time - timedelta(minutes = dt)
    endtime = origin_time + timedelta(minutes = dt)
    # longitude
    minlongitude = str(float(ee['lon']) - dlon)
    maxlongitude = str(float(ee['lon']) + dlon)
    # latitude
    minlatitude = str(float(ee['lat']) - dlat)
    maxlatitude = str(float(ee['lat']) + dlat)
    # magnitude
    minmagnitude = str(float(ee['mag']) - dmag)
    maxmagnitude = str(float(ee['mag']) + dmag)
    # print(minmagnitude, maxmagnitude)

    nodes = {
             #"auspass": "https://auspass.edu.au",
             #"eida": "http://eida-federator.ethz.ch",
             #"emsc": "http://www.seismicportal.eu",
             #"eth": "http://eida.ethz.ch",
             "gfz": "http://geofon.gfz-potsdam.de",
             #"ingv": "https://webservices.ingv.it",
             #"iris": "https://service.iris.edu",
             #"knmi": "https://rdsa.knmi.nl",
             #"orfeus": "https://www.orfeus-eu.org",
             #"usgs": "https://earthquake.usgs.gov",
             #"isc": "http://www.isc.ac.uk",
             #"koeri": "https://www.koeri.boun.edu.tr"
             }


    params = {
              "starttime": starttime.strftime("%Y-%m-%dT%H:%M:%S"),
              "endtime": endtime.strftime("%Y-%m-%dT%H:%M:%S"),
              "minlongitude": minlongitude,
              "maxlongitude": maxlongitude,
              "minlatitude": minlatitude,
              "maxlatitude": maxlatitude,
              "minmagnitude": minmagnitude,
              "maxmagnitude": maxmagnitude,
              "mindepth": "-1",
              "maxdepth": "700",
              #"eventtype": "earthquake",
              #"includefocalmechanism": "true"
              }


    for node, db_node in nodes.items():

        if node == 'gfz':
            params_update = params.copy()
            params_update['includefocalmechanism'] = 'true'
        elif node == 'usgs':
            params_update = params.copy()
            params_update['producttype']= 'focal-mechanism'
        else:
            sys.exit('Wrong focal mechanism provider')

        resp = requests.get(db_node + "/fdsnws/event/1/query?", params = params_update, verify=False)

        print(node, params, resp.status_code, resp.url, "\n")

        if resp.status_code in range(200,299):
            #station_xml_found = True

            logger.info(f'    Found Quakeml in: {node}')

            root = ET.fromstring(resp.text)  # .getroot()

            # Get number of events
            for eventParameters in root.findall("{http://quakeml.org/xmlns/bed/1.2}eventParameters"):
                num_events = len(eventParameters.findall("{http://quakeml.org/xmlns/bed/1.2}event"))
            print(num_events)
            if num_events > 1:
                sys.exit("Error: " + str(num_events) + " events found")
             
            # for element in root.findall('.//{http://quakeml.org/xmlns/bed/1.2}scalarMoment'):
            #     scalar_moment = float(element[0].text)

            strike = [] 
            for element in root.findall('.//{http://quakeml.org/xmlns/bed/1.2}strike'):
                strike.append(float(element[0].text))

            dip = []
            for element in root.findall('.//{http://quakeml.org/xmlns/bed/1.2}dip'):
                dip.append(float(element[0].text))

            rake = []
            for element in root.findall('.//{http://quakeml.org/xmlns/bed/1.2}rake'):
                rake.append(float(element[0].text))

            mt = {'np1': {'strike': strike[0], 'dip': dip[0], 'rake': rake[0]},
                  'np2': {'strike': strike[1], 'dip': dip[1], 'rake': rake[1]}}
           
            logger.info(f'    {mt}')

            # save every quakeml in xml file
            quakeml_outfile = os.path.join(workdir, f"{node}_quakeml.xml")
            with open(quakeml_outfile, "w") as xml_file:
               xml_file.write(resp.text)

        else:
            logger.info(f'    {node}: Quakeml not found')



    return mt


# unused functions

# def clean_tmp_ray_files(**kwargs):
# 
#     args   = kwargs.get('args', None)
#     Config = kwargs.get('cfg', None)
# 
#     if(bool(Config['ray']['clean_at_start']) == True):
# 
#         #find temp dir:
#         dirs_to_remove=sorted(glob.glob(Config['ray']['tmp_dir'] + os.sep + 'session_????_*'))
# 
#         for i in range(len(dirs_to_remove)):
# 
#             if (Path(dirs_to_remove[i]).owner() == Config['ray']['username']):
#                 try:
#                     shutil.rmtree(dirs_to_remove[i])
#                 except OSError as error:
#                     print(error)
#                     print("File path can not be removed")
# 
#     return True
# 
# def merge_event_dictionaries(**kwargs):
# 
#     event_ttt = kwargs.get('event_ttt', None)
#     event_ptf = kwargs.get('event_ptf', None)
# 
#     full_event = {**event_ttt, **event_ptf}
# 
#     return full_event
# 
# def st_to_float(x):
# 
#     # if any number
#     if isinstance(x,numbers.Number):
#         return x
#     # if non a number try convert string to float or it
#     for type_ in (float):
#         try:
#             return type_(x)
#         except ValueError:
#             continue
# 
# def iterdict(d):
#     for k,v in d.items():
#         if isinstance(v, dict):
#             iterdict(v)
#         else:
#             #print (k,":",v)
#             print(k)
# 
# def get_HazardCurveThresholds(**kwargs):
# 
#     table = kwargs.get('table', None)
# 
#     thrs = table["HazardCurveThresholds"]
#     lut = np.zeros((len(thrs),len(thrs)))
#     for i in range(len(thrs)):
#         lut[:i,i] = 1
# 
#     return lut
#
# def check_if_neam_event(**kwargs):
#
#     dictionary = kwargs.get('dictionary', None)
#     cfg        = kwargs.get('cfg', None)
#
#     area_neam  = eval(cfg.get('bounds','neam'))
#     inneam = ray_tracing_method(float(dictionary['lon']), float(dictionary['lat']), area_neam)
#
#     if (inneam == True):
#         dictionary['inneam'] = True
#     else:
#         dictionary['inneam'] = False
#
#     return dictionary
