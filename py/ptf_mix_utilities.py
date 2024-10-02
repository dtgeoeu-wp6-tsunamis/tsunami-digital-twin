import os
import utm
import sys
import math
import socket
import glob
import shutil
import numbers
import numpy as np
import numpy.matlib as npm
from numba import jit
from pathlib import Path


def update_workflow_dict(**kwargs):

    from scipy.stats import norm

    args   = kwargs.get('args', None)
    wd     = kwargs.get('workflow_dict', None)

    if args.sigma != None:
        wd['sigma'] = float(args.sigma)

    if args.percentiles != None:
        wd['percentiles'] = args.percentiles

    if args.hazard_mode != None:
        wd['hazard_mode'] = args.hazard_mode

    # Moved definition of wd['workdir'] in ptf_load_event/load_event_dict, after .json of event is processed 
    # to name the workdir folder with the eventID
    
    wd['sigma_inn'] = wd['sigma']
    wd['sigma_out'] = wd['sigma'] + 0.5
    wd['negligible_prob'] = 2*norm.cdf(-1. * wd['sigma'])
    wd['percentiles'] = np.array(wd['percentiles'])*0.01

    return wd


def create_workflow_dict(**kwargs):

    args   = kwargs.get('args', None)
    Config = kwargs.get('Config', None)

    wd = dict()

    wd['event_file'] = args.event
    wd['event_name'] = os.path.basename(wd['event_file']).split('.')[0]
    
    inpjson = args.input_workflow
    f = open(inpjson, 'r').read()
    jsn_object = eval(f)

    wd['step1'] = jsn_object['STEPS']['step1']    # ENSEMBLE DEFINITION
    wd['step2'] = jsn_object['STEPS']['step2']    # TSUNAMI SIMULATIONS
    wd['step3'] = jsn_object['STEPS']['step3']    # HAZARD AGGREGATION
    wd['step4'] = jsn_object['STEPS']['step4']    # ALERT LEVELS
    wd['step5'] = jsn_object['STEPS']['step5']    # VISUALIZATION

    wd['domain'] = jsn_object['SETTINGS']['domain']
    # wd['event_name'] = jsn_object['SETTINGS']['event_name'] # only from command line
    # 2020_1030_samos => sigma=0.7 for testing
    # 2018_1025_zante => sigma=1.3 for testing
    wd['sigma'] = jsn_object['SETTINGS']['sigma']
    wd['percentiles'] = jsn_object['SETTINGS']['percentiles']
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
    wd['step2_hmax_pre_BS_root'] = Config.get('save_ptf', 'step2_hmax_pre_BS')
    wd['step2_hmax_pre_PS_root'] = Config.get('save_ptf', 'step2_hmax_pre_PS')
    wd['step2_hmax_sim_BS_root'] = Config.get('save_ptf', 'step2_hmax_sim_BS')
    wd['step2_hmax_sim_PS_root'] = Config.get('save_ptf', 'step2_hmax_sim_PS')

    wd['step2_dir_sim_BS_root'] = Config.get('save_ptf', 'step2_dir_sim_BS')
    wd['step2_dir_sim_PS_root'] = Config.get('save_ptf', 'step2_dir_sim_PS')
    wd['step2_dir_log_BS_root'] = Config.get('save_ptf', 'step2_dir_log_BS')
    wd['step2_dir_log_PS_root'] = Config.get('save_ptf', 'step2_dir_log_PS')
    wd['step2_log_failed_BS_root'] = Config.get('save_ptf', 'step2_log_failed_BS')
    wd['step2_log_failed_PS_root'] = Config.get('save_ptf', 'step2_log_failed_PS')
    wd['step2_list_simdir_BS_root']  = Config.get('save_ptf', 'step2_list_simdir_BS') 
    wd['step2_list_simdir_PS_root']  = Config.get('save_ptf', 'step2_list_simdir_PS') 
    wd['step2_list_all_sims_BS']  = Config.get('save_ptf', 'step2_list_all_sims_BS') 
    wd['step2_list_all_sims_PS']  = Config.get('save_ptf', 'step2_list_all_sims_PS') 
    wd['step2_newsimulations_BS_root']  = Config.get('save_ptf', 'step2_newsimulations_BS') 
    wd['step2_newsimulations_PS_root']  = Config.get('save_ptf', 'step2_newsimulations_PS') 
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

    wd['wf_path'] = Config.get('pyptf', 'wf_path')
    wd['inpdir'] = Config.get('pyptf', 'data_path')
    # wd['event_path'] = Config.get('pyptf', 'event_path')

    # for sampling
    wd['sampling_mode'] = jsn_object['SETTINGS']['sampling mode']
    wd['sampling_type'] = jsn_object['SETTINGS']['sampling type']
    wd['number_of_scenarios'] = jsn_object['SETTINGS']['number of scenarios']

    # for tsunami simulations
    wd['ptf_measure_type'] = Config.get('Settings', 'ptf_measure_type')
    # wd['regional_pois_filename'] = Config.get('tsu_sims', 'regional_pois')
    wd['bathy_folder'] = Config.get('tsu_sims', 'bathy_folder')
    wd['regional_bathy_file'] = Config.get('tsu_sims', 'regional_bathy_file')
    wd['regional_pois_depth'] = Config.get('tsu_sims', 'regional_pois_depth')
    wd['ps_inicond_med'] = Config.get('tsu_sims', 'ps_inicond_med')
    wd['n_digits'] = int(Config.get('tsu_sims', 'n_digits'))
    wd['propagation'] = jsn_object['SETTINGS']['propagation']
    wd['sim_postproc'] = Config.get('tsu_sims', 'sim_postproc')
    wd['final_postproc'] = Config.get('tsu_sims', 'final_postproc')
    wd['BS_parfile_tmp'] = Config.get('tsu_sims', 'BS_parfile_tmp')
    wd['PS_parfile_tmp'] = Config.get('tsu_sims', 'PS_parfile_tmp')
    wd['bathy_filename'] = Config.get('tsu_sims', 'bathy_filename')
    wd['depth_filename'] = Config.get('tsu_sims', 'depth_filename')
    wd['run_sim_filename'] = Config.get('tsu_sims', 'run_sim_filename')
    wd['run_post_filename'] = Config.get('tsu_sims', 'run_post_filename')

    # for computing the PDF
    wd['compute_pdf'] = jsn_object['SETTINGS']['compute_pdf']

    # for hazard curves
    wd['hazard_mode'] = Config.get('Settings', 'hazard_mode')
    wd['save_nc'] = jsn_object['SETTINGS']['save_nc']

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
        print('WARNING: HPC cluster for simulations not properly defined')

    return wd

def check_previous_versions(**kwargs):
    wd = kwargs.get('wd', None)
    ndig = wd['n_digits']

    print('Checking if previous versions of the same event have been already processed')
    step1_outfiles = glob.glob(wd['workdir'] + os.sep + 'step1*.txt')
    step1_previous_run = [f for f in step1_outfiles if not f.endswith(f"{wd['version']}.txt")]

    if (len(step1_previous_run) > 0):
        #previous_list_BS = [f for f in step1_previous_run if 'BS' in f]
        #previous_list_PS = [f for f in step1_previous_run if 'PS' in f]
        previous_list_BS = os.path.join(wd['workdir'], wd['step2_list_all_sims_BS'])
        previous_list_PS = os.path.join(wd['workdir'], wd['step2_list_all_sims_PS'])

        if os.path.getsize(previous_list_BS[0]) > 0:
           file_simBS = compare_list_scenarios(wd = wd,
                                               seistype = 'BS',
                                               previous_list = previous_list_BS)

        if os.path.getsize(previous_list_PS[0]) > 0:
           file_simPS = compare_list_scenarios(wd = wd,
                                               seistype = 'PS',
                                               previous_list = previous_list_PS)

    else:
        file_simBS = os.path.join(wd['workdir'],wd['step1_list_BS'])
        if os.path.getsize(file_simBS) > 0:
           list_scenarios = [] 
           with open(file_simBS) as f:
              lines = f.readlines()
              list_scenarios.append([wd['uniqueID'] + ' ' + line for line in lines])
              nscen = len(lines)

           list_simdir = []
           for i in range(nscen):
              scen_id = 'BS_Scenario' +  str(i+1).zfill(ndig)
              sim_dir = os.path.join(wd['step2_dir_sim_BS'], scen_id)
              list_simdir.append(sim_dir)

           filename = os.path.join(wd['workdir'], wd['step2_list_simdir_BS'])
           with open(filename,'w') as f:
              f.write('\n'.join(list_simdir) + '\n')              

           filename = os.path.join(wd['workdir'], wd['step2_list_all_sims_BS'])
           with open(filename,'w') as f:
              f.write("".join(list_scenarios[0]))              
        else:
           filename = os.path.join(wd['workdir'], wd['step2_list_all_sims_BS'])
           f = open(filename, 'w')
           f.close()

        file_simPS = os.path.join(wd['workdir'],wd['step1_list_PS'])
        if os.path.getsize(file_simPS) > 0:
           list_scenarios = [] 
           with open(file_simPS) as f:
              lines = f.readlines()
              list_scenarios.append([wd['uniqueID'] + ' ' + line for line in lines])
              nscen = len(lines)

           list_simdir = []
           for i in range(nscen):
              scen_id = 'PS_Scenario' +  str(i+1).zfill(ndig)
              sim_dir = os.path.join(wd['step2_dir_sim_PS'], scen_id)
              list_simdir.append(sim_dir)

           filename = os.path.join(wd['workdir'], wd['step2_list_simdir_PS'])
           with open(filename,'w') as f:
              f.write('\n'.join(list_simdir) + '\n')              

           filename = os.path.join(wd['workdir'], wd['step2_list_all_sims_PS'])
           with open(filename,'w') as f:
              f.write("".join(list_scenarios[0]))              
        else:
           filename = os.path.join(wd['workdir'], wd['step2_list_all_sims_PS'])
           f = open(filename, 'w')
           f.close()

        print('No previous versions of this event found')

    return file_simBS, file_simPS

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

def get_info_from_scenario_list(**kwargs):

    filename = kwargs.get('filename', None)
    seistype = kwargs.get('seistype', None)

    with open(filename, 'r') as f:
        first_line = f.readline()
        nsce = len(f.readlines()) + 1
        print('Number of ' + seistype + ' scenarios = ' + str(nsce))

    return nsce, first_line


def check_if_neam_event(**kwargs):

    dictionary = kwargs.get('dictionary', None)
    cfg        = kwargs.get('cfg', None)

    area_neam  = eval(cfg.get('bounds','neam'))

    inneam = ray_tracing_method(float(dictionary['lon']), float(dictionary['lat']), area_neam)

    if (inneam == True):
        dictionary['inneam'] = True
    else:
    	dictionary['inneam'] = False

    return dictionary


def conversion_to_utm(**kwargs):

    long = kwargs.get('longTerm', None)
    pois = kwargs.get('Poi',      None)
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

    a = utm.from_latlon(np.array(pois['lat']), np.array(pois['lon']), ee['ee_utm'][2])
    pois['utm_lat'] = a[1]
    pois['utm_lon'] = a[0]
    pois['utm_nr']  = a[2]
    pois['utm_reg'] = a[3]

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

    return long, pois, PSBa


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
