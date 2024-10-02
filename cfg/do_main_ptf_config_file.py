#!/usr/bin/env python

import os
import sys
import socket
import shutil
import getpass
import argparse
import datetime
import configparser
from pathlib import Path


def parseMyLine():

    ll = sys.argv[1:]
    if not ll:
       print ("Type " + sys.argv[0] + " -h or --help for guide and examples")
       sys.exit(0)

    Description = "Create main configuration file for pyPTF code"

    examples    = "Example:\n" + sys.argv[0] + " --cfg ./cfg/ptf_main.config --data_path /data/INPUT4PTF --work_path /data/users/ptf_events "

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=Description, epilog=examples)

    parser.add_argument('--cfg', default='None', help='Configuration file name with path. Default=None')
    parser.add_argument('--wf_path',    default='./',   help='Workflow path name. Default=.')
    parser.add_argument('--data_path',  default='None', help='PTF Data working path name. Default=None')
    parser.add_argument('--work_path',  default='None', help='Working folder (output) path name. Default=None')
    # only if remote simulations are needed
    parser.add_argument('--data_path_remote',  default='None', help='PTF Data path on the remote cluster (if remote connection is required). Default=None')
    parser.add_argument('--work_path_remote',  default='None', help='Working folder on the remote cluster path name (if remote connection is required). Default=None')

    args=parser.parse_args()

    return args

#def set_parameters_for_server():
#
#
#    ru = dict()
#
#    ru['host_complete']       = socket.getfqdn()
#    ru['host_name']           = socket.getfqdn().split('.')[0]
#    ru['creation_time']       = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
#    ru['username']            = getpass.getuser()
#
#    return ru

def main(**kwargs):

    # 1: Read arguments
    # ==================================================================
    args=parseMyLine()

    # 2 If config file exists, create backup file
    # ==================================================================
    cfg_file = Path((args.cfg))
    if cfg_file.exists():
       shutil.move(args.cfg, args.cfg + ".back")

    # 3: Initialize config instance
    # ==================================================================
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option  # preserve case for letters

#    server_config_setup = set_parameters_for_server()
    
    # 4: Define main paths
    data_folder   = args.data_path
    wf_folder    = args.wf_path
    if wf_folder == './':
        wf_folder = os.getcwd()

    work_main_folder        = args.work_path
    work_main_folder_remote = args.work_path_remote
    data_folder_remote       = args.data_path_remote

    foc_mech_preprocessing_folder  = data_folder + os.path.sep + 'FocMech_PreProc'
    mix_npy_files_folder           = data_folder + os.path.sep + 'mix_npy_files'
    fcp_files_folder               = data_folder + os.path.sep + 'fcp'
    scenarios_list_folder          = data_folder + os.path.sep + 'ScenarioList'
    curves_gl_16                   = data_folder + os.path.sep + 'glVal_int16'
    path_mesh                      = data_folder + os.path.sep + 'mesh_files'
    path_coast_shapefiles          = data_folder + os.path.sep + 'coast_shapefiles'

    # 5: Files
    regionalization_npy            = mix_npy_files_folder + os.path.sep + 'regionalization.npy'
    pois_npy                       = mix_npy_files_folder + os.path.sep + 'POIs.npy'
    discretization_npy             = mix_npy_files_folder + os.path.sep + 'discretizations.npy'
    lookuptable_npy                = mix_npy_files_folder + os.path.sep + 'LookupTable.npy' 
    ps_bar_info                    = mix_npy_files_folder + os.path.sep + 'PSBarInfo.npy'  
    model_weight                   = mix_npy_files_folder + os.path.sep + 'ModelWeights.npy'
    fcp_json                       = fcp_files_folder + os.path.sep + 'fcp_ws_all.json'
    pois_to_fcp                    = fcp_files_folder + os.path.sep + 'pois_to_fcp.npy'
    slab_meshes                    = path_mesh + os.path.sep + 'slab_meshes_mediterranean.npy'
    coast_shapefile                = path_coast_shapefiles + os.path.sep + '50m_land.shp'
    intensity_thresholds           = data_folder + os.path.sep + 'intensity_thresholds.npy'
    
    config['pyptf']                = {}
    config['pyptf']['wf_path']     = wf_folder
    config['pyptf']['data_path']   = data_folder
    
    #REGIONALIZATION
    config['pyptf']['Regionalization_npy']       = regionalization_npy
    
    #SCENARIOS
    config['pyptf']['Scenarios_py_Folder']       = scenarios_list_folder
    config['pyptf']['bs_file_names']             = 'ScenarioListBS_Reg'
    config['pyptf']['ps_file_names']             = 'ScenarioListPS_Reg'
    
    # 
    config['pyptf']['curves_gl_16']              = curves_gl_16
    config['pyptf']['gl_bs_curves_file_names']   = 'glVal_BS_Reg'
    config['pyptf']['gl_ps_curves_file_names']   = 'glVal_PS_Reg'
    config['pyptf']['af_bs_curves_file_names']   = 'afVal_BS_Reg'
    config['pyptf']['af_ps_curves_file_names']   = 'afVal_PS_Reg'
    config['pyptf']['os_bs_curves_file_names']   = 'osVal_BS_Reg'
    config['pyptf']['os_ps_curves_file_names']   = 'osVal_PS_Reg'
    config['pyptf']['POIs_npy']                  = pois_npy
    config['pyptf']['Discretization_npy']        = discretization_npy
    config['pyptf']['LookUpTable_npy']           = lookuptable_npy
    config['pyptf']['FocMech_Preproc']           = foc_mech_preprocessing_folder
    config['pyptf']['PSBarInfo']                 = ps_bar_info
    config['pyptf']['ModelWeight']               = model_weight
    
    #coast_shapefile
    config['pyptf']['coast_shapefile']           = coast_shapefile

    config['pyptf']['intensity_thresholds']      = intensity_thresholds
    
    #  defining file/folder names
    config['save_ptf']                              = {}
    config['save_ptf']['save_main_path']            = work_main_folder
    config['save_ptf']['pois']                      = 'pois.npy'
    config['save_ptf']['step1_prob_BS']             = 'step1_prob_scenarios_BS_'
    config['save_ptf']['step1_prob_PS']             = 'step1_prob_scenarios_PS_'
    config['save_ptf']['step1_list_BS']             = 'step1_list_scenarios_BS_'
    config['save_ptf']['step1_list_PS']             = 'step1_list_scenarios_PS_'
    config['save_ptf']['step2_hmax_pre_BS']         = 'step2_hmax_pre_BS_'
    config['save_ptf']['step2_hmax_pre_PS']         = 'step2_hmax_pre_PS_'
    config['save_ptf']['step2_hmax_sim_BS']         = 'step2_hmax_sim_BS_'
    config['save_ptf']['step2_hmax_sim_PS']         = 'step2_hmax_sim_PS_'
    config['save_ptf']['step2_dir_sim_BS']          = 'step2_sim_BS_'
    config['save_ptf']['step2_dir_sim_PS']          = 'step2_sim_PS_'
    config['save_ptf']['step2_dir_log_BS']          = 'step2_log_BS_'
    config['save_ptf']['step2_dir_log_PS']          = 'step2_log_PS_'
    config['save_ptf']['step2_log_failed_BS']       = 'step2_log_failed_BS_'
    config['save_ptf']['step2_log_failed_PS']       = 'step2_log_failed_PS_'
    config['save_ptf']['step2_list_simdir_BS']      = 'step2_list_simdir_BS_'
    config['save_ptf']['step2_list_simdir_PS']      = 'step2_list_simdir_PS_'
    config['save_ptf']['step2_list_all_sims_BS']    = 'step2_list_all_sims_BS.txt'
    config['save_ptf']['step2_list_all_sims_PS']    = 'step2_list_all_sims_PS.txt'
    config['save_ptf']['step2_newsimulations_BS']   = 'step2_newsimulations_BS_'
    config['save_ptf']['step2_newsimulations_PS']   = 'step2_newsimulations_PS_'
    config['save_ptf']['step2_ts_file']             = 'step2_ts.dat'
    config['save_ptf']['step3_hc']                  = 'step3_hazard_curves_'
    config['save_ptf']['step3_hc_perc']             = 'step3_hazard_curves_percentiles_'
    config['save_ptf']['step3_hazard_pdf']          = 'step3_hazard_pdf_'
    config['save_ptf']['step4_alert_levels_FCP']    = 'step4_alert_levels_FCP_'
    config['save_ptf']['step4_alert_levels_POI']    = 'step4_alert_levels_POI_'
    config['save_ptf']['step5_figures']             = 'step5_figures_'
    config['save_ptf']['step5_alert_levels']        = 'alert_levels_map_'
    config['save_ptf']['step5_hazard_maps']         = 'hazard_map_'
    config['save_ptf']['step5_hazard_curves']       = 'hazard_curves_'
    config['save_ptf']['workflow_dictionary']       = 'workflow_dictionary_'
    config['save_ptf']['event_dictionary']          = 'event_dictionary_'
#    config['save_ptf']['save_format']            = 'npy'  # may be 'hdf5'
#    config['save_ptf']['poi_html_map']           = 'poi_map'
#    config['save_ptf']['message_dict']           = 'message_dict'  # may be 'hdf5'
#    config['save_ptf']['geojson']                = 'geojson_event'
   
    # for remote connection
    config['pyptf']['data_path_remote'] = data_folder_remote
    config['save_ptf']['save_main_path_remote']     = work_main_folder_remote

    ####################################################################
    
    config['Files']                                  = {}
    config['Files']['focal_mechanism_root_name']     = 'MeanProb_BS4_FocMech_Reg'
    config['Files']['meshes_dictionary']             = slab_meshes
    config['Files']['pois_to_fcp']                   = pois_to_fcp
    config['Files']['alert_message']                 = 'alert_message.txt'
    config['Files']['fcp_json']                      = fcp_json
    # config['Files']['probability_models_root_name']  = 'ModelsProb_Reg'
    # config['Files']['source']                        = 'epicentral_source.txt'
    # config['Files']['fcp_wrk']                       = 'fcp_all.txt'
    # config['Files']['fcp_time_wrk']                  = 'fcp_time.txt'
    # config['Files']['bathymetry_wrk']                = 'bathymetry_cut.b'
    # config['Files']['ttt_out_wrk']                   = 'ttt.b'
    

    # PATHS
    # config['PATHS']                      = {}
    ####################################################################
   
    # tunami simulations
    # regional_pois       = 'POIs.txt'
    bathy_folder        = 'bathy_grids'

    config['tsu_sims']                          = {}
    config['tsu_sims']['bathy_folder']          = bathy_folder
    # config['tsu_sims']['regional_pois']         = regional_pois
    config['tsu_sims']['regional_bathy_file']   = bathy_folder + os.path.sep + 'regional_domain.grd'
    config['tsu_sims']['regional_pois_depth']   = bathy_folder + os.path.sep + 'regional_domain_POIs_depth.npy'
    config['tsu_sims']['ps_inicond_med']        = 'INIT_COND_PS_TSUMAPS1.1'
    config['tsu_sims']['n_digits']              = '6'
    config['tsu_sims']['sim_postproc']          = wf_folder + os.path.sep + 'py/step2_extract_ts.py'
    config['tsu_sims']['final_postproc']        = wf_folder + os.path.sep + 'py/step2_create_ts_input_for_ptf.py'
    config['tsu_sims']['BS_parfile_tmp']        = wf_folder + os.path.sep + 'templates/step2_parfile_tmp.txt'
    config['tsu_sims']['PS_parfile_tmp']        = wf_folder + os.path.sep + 'templates/step2_parfile_TRI_tmp.txt'
    config['tsu_sims']['run_sim_tmp_mercalli']  = wf_folder + os.path.sep + 'sh/step2_run_tmp@mercalli.sh'
    config['tsu_sims']['run_sim_tmp_leonardo']  = wf_folder + os.path.sep + 'sh/step2_run_tmp@leonardo.sh'
    config['tsu_sims']['run_post_tmp_mercalli'] = wf_folder + os.path.sep + 'sh/step2_final_postproc_tmp@mercalli.sh'
    config['tsu_sims']['run_post_tmp_leonardo'] = wf_folder + os.path.sep + 'sh/step2_final_postproc_tmp@leonardo.sh'
    config['tsu_sims']['bathy_filename']        = 'step2_bathygrid.grd'
    config['tsu_sims']['depth_filename']        = 'step2_pois_depth.npy'
    config['tsu_sims']['run_sim_filename']      = 'step2_run_tmp.sh'
    config['tsu_sims']['run_post_filename']     = 'step2_final_postproc.sh'

    ####################################################################
    # BEGIN EVENT TREE  Definition ------------------------------------- ##
    
    # config['BS']                      = {}
    # config['BS']['EventTreeNodes']    = 'BS-1_Magnitude,BS-2_Position,BS-3_Depth,BS-4_FocalMechanism,BS-5_Area,BS-6_Slip'
    # config['BS']['Moho_File']         = 'Grid025_MOHO_FIXED.txt'
    # config['PS']                      = {}
    # config['PS']['EventTreeNodes']    = 'PS-1_Magnitude,PS-2_PositionArea,PS-3_Slip'
    # config['PS']['Barycenters_File']  = 'barycenters_xyz_ALL.txt'
    
    # END EVENT TREE  Definition ------------------------------------- ##
    ####################################################################
    
    config['matrix']                        = {}
    config['matrix']['local_distance']      = '100'
    config['matrix']['regional_distance']   = '400'
    # config['matrix']['basin_distance']      = '1000'
    config['matrix']['min_mag_for_message'] = '4.499'
    
    config['bounds']         = {}
    config['bounds']['neam'] = '([1.00, 28.00], [1.00, 32.00], [-7.00, 32.00], [-7.00, 42.00], [1.00, 42.00], [1.00, 47.00], [27.00, 47.00], [27.00, 41.15], [29.50, 41.15], [29.50, 41.00], [37.50, 41.00], [37.50, 30.00], [27.00, 30.00], [27.00, 28.00])'
    
    ####################################################################
    #  BEGIN    Settings  --------------------------------------------- #
    
    config['Settings']                                 = {}
    # config['Settings']['nr_cpu_max']                   = '28'
    # config['Settings']['Selected_Pois']                = 'mediterranean'
    config['Settings']['Space_Bin']                    = '2.5E3'                 #GRID SIZE IN M FOR 2D OR 3D INTEGRAL (ALONG Z)
    config['Settings']['Z2XYfact']                     = '2.5'                   #RATIO BETW EarlyEst.N GRID SIZE ALONG X,Y AND Z (IF > 1, FINER ON Z)
    config['Settings']['Mag_BS_Max']                   = '8.1'                   #MAXIMUM MODELLED MAGNITUDE FOR BS
    config['Settings']['Mag_PS_Max']                   = '9.1'
    config['Settings']['selected_intensity_measure']   = 'gl'                    #TO SELECT AMONG AF, GL AND OS
    config['Settings']['run_up_yn']                    = 'False'                  #if false, the intensity measure is used as it is; if true, a further amplification is applied (3 for af; 2 for gl)
    config['Settings']['nr_points_2d_ellipse']         = '1000'
    config['Settings']['hazard_curve_sigma']           = '1'
    config['Settings']['ptf_measure_type']             = 'ts_p2t_gl'            # Allowd values: 'ts_max', 'ts_max_gl', 'ts_max_off', 'ts_max_off_gl', 'ts_min', 'ts_min_off', 'ts_p2t', 'ts_p2t_gl'
    config['Settings']['hazard_mode']                  = 'lognormal_v1'       # Allowed values: 'no_uncertainty', 'lognormal', 'lognormal_v1'
    # config['Settings']['writeOutTesting']              = 'False'                 #???
    # config['Settings']['verboseYN']                    = 'False'                 #???
    # config['Settings']['figcheckYN']                   = 'False'                 #???
    # config['Settings']['lambdaFabrizioYN']             = 'False'                 #???
    
#    # Here ray Settings
#    config['ray']                                      = {}
#    config['ray']['nr_cpu_max']                        = '28'
#    config['ray']['tmp_dir']                           = '/tmp/ray'
#    config['ray']['clean_at_start']                    = 'True'
#    config['ray']['username']                          = server_config_setup['username']
#    #config['ray']['group']                             = 'True'
    
    
    # here the number of the region (starting from 1 like in hazard)
    config['regionsPerPS']                              = {}
    config['regionsPerPS']['1']                         = '[3,24,44,48,49]'
    config['regionsPerPS']['2']                         = '[10,16,54]'
    config['regionsPerPS']['3']                         = '[27,33,35,36]'
    config['regionsPerPS']['all']                       = '[3,10,16,24,27,33,35,36,44,48,49,54]'
    
    config['ScenariosList']                             = {}
    # config['ScenariosList']['BS_all_dict']              = 'No'
    # config['ScenariosList']['BS_parameter_nr_coll']     = '7'
    config['ScenariosList']['nr_regions']               = '110'
    config['ScenariosList']['bs_empty']                 = '[37,38,39,41,43,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,82,83,84,85,86,87,88,89,90,92,100,101,102,103,104,105,108,109]'
    
    config['mix_parameters']                           = {}
    config['mix_parameters']['ignore_regions']         = '42'  #!!REGION 43 NOT AVAILABLE IN TSUMAPS1.1!!#
    # config['mix_parameters']['vecID']                  = '[100000000,100000,100,1,0.0100,1.0000e-04,1.0000e-06]'
    
    config['lambda']                                   = {}
    config['lambda']['Ntetra']                         = '23382'
    config['lambda']['Vol_tetra']                      = '3.9990e+15'
    config['lambda']['a']                              = '0.6211'
    config['lambda']['b']                              = '0.4358'
    config['lambda']['subd_buffer']                    = '10000'            # min distance for psbs separation prob. in [m]
    config['lambda']['mesh_path']                      = path_mesh
    config['lambda']['mesh_nodes']                     = '[\'HA_mesh_nodes_x16.dat\', \'CA_mesh_nodes_x16.dat\', \'Cyprus_mesh_nodes_x16.dat\']'
    config['lambda']['mesh_faces']                     = '[\'HA_mesh_faces_x16.dat\', \'CA_mesh_faces_x16.dat\', \'Cyprus_mesh_faces_x16.dat\']'
    config['lambda']['mesh_names']                     = '[ \'HeA\', \'CaA\', \'CyA\']'
    config['lambda']['mesh_zones']                     = '{\'0\':\'[2,23,43,47,48]\', \'1\':\'[9,15,53]\', \'2\':\'[26,32,34,35]\'}'
    # config['lambda']['out_tetra_vol']                  = '1'                #   1=Yes, 0=No

    
    # Alert levels settings
    config['alert_levels']                  = {}
    config['alert_levels']['intensity']     = '{\'run_up_yes\':\'[0,0.40,1.0]\', \'run_up_no\':\'[0,0.2,0.5]\'}'
    config['alert_levels']['names']         = '[\'Information\',\'Advisory\',\'Watch\']'
    config['alert_levels']['fcp_method']    = '{\'method\': \'probability\', \'rule\': \'max\', \'probability_level\': 0.05}'
    # config['alert_levels']['type']          = '{\'matrix\':\'1\',  \'average\':\'1\', \'best\':\'1\', \'P\':\'[50,500,50]\' }'
    # config['alert_levels']['cc']            = '[\'[1,1,1]\', \'[0,1,0]\', \'[1,0.8,0]\', \'[1,0,0]\']'
    # config['alert_levels']['max_dist']      = '300'
    # config['alert_levels']['Nr_near']       = '3'

#    config['this_config']                   = {}
#    config['this_config']['host_complete']  = server_config_setup['host_complete']
#    config['this_config']['host_name'] = server_config_setup['host_name']
#    config['this_config']['creation_time'] = server_config_setup['creation_time']
#    config['this_config']['username'] = server_config_setup['username']

    # Rabbit
    config['rabbit']                      = {}
    config['rabbit']['rabbit_RK']         = 'INT.QUAKEEXTPTF.CATDEV' # rabbit-mq routing key
    config['rabbit']['rabbit_vhost']      = '/rbus' # rabbit virtual host
    config['rabbit']['rabbit_host']       = 'rabbitmq1-rm.int.ingv.it' # rabbit host server
    config['rabbit']['rabbit_port']       = '5672' # rabbit port number
    config['rabbit']['rabbit_login']      = 'rbus_writer' # rabbit login
    config['rabbit']['rabbit_pass']       = 'pass_writer' # rabbit password
    config['rabbit']['rabbit_exchange']   = 'cat' # rabbit exchange
    config['rabbit']['rabbit_type']       = 'topic' # rabbit type
    config['rabbit']['broker']            = 'rabbit' # broker [rabbit]/kafka
    config['rabbit']['rabbit_consumer_q'] = 'test_ptf_3cards' # rabbit consumer queue
    config['rabbit']['rabbit_mode']       = 'clean' # service start mode. save: hold and process the queue. [clean]: empty queue before starting





    # LAST: write configuration file
    # ==================================================================
    with open(args.cfg, 'w') as configfile:
         config.write(configfile)
    # ==================================================================

if __name__ == "__main__":
    #main(sys.argv[1], sys.argv[2], sys.argv[3])
    #main(**dict(arg.split('=') for arg in sys.argv[1:]))
    main()
