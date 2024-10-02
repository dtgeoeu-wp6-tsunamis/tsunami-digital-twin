#!/usr/bin/env python

# Import system modules
import os
import sys
import shutil
import time
import subprocess as sp
from pyutil import filereplace

from files_utils import force_symlink
from ptf_mix_utilities import get_info_from_scenario_list
import t_hysea as hysea
import step2_remote_connection as remote


def create_setup(**kwargs):
    wd = kwargs.get('workflow_dict', None)

    workdir = wd['workdir']
    inpdir = wd['inpdir']
    sim_domain = wd['sim_domain']
    domain = wd['domain']
    eventID = wd['uniqueID']
    hpc_cluster = wd['hpc_cluster']
    UCmode = wd['UCmode']
    pydir = os.path.join(wd['wf_path'], 'py')

    if sim_domain == 'regional':
        poifile = os.path.join(workdir, wd['pois'])
        bathyfile = os.path.join(inpdir, domain, wd['regional_bathy_file'])
        depthfile = os.path.join(inpdir, domain, wd['regional_pois_depth'])
    else:  # local (TODO: add automatic procedure for cutting the grid and selecting the POIs)
        poifile = os.path.join(inpdir, domain, 'POIs_local_domain_' + eventID + '.npy')
        bathyfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_' + eventID + '.grd')
        depthfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_' + eventID + '_POIs_depth.npy')
        # !!only fos SAMOS!!
        poifile = os.path.join(inpdir, domain, 'POIs_local_domain_2020_1030_samos.npy')
        bathyfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_2020_1030_samos.grd')
        depthfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_2020_1030_samos_POIs_depth.npy')

    hysea.save_pois_for_hysea(poifile = poifile,
                              workdir = workdir,
                              outfile = wd['step2_ts_file'])
    
    force_symlink(bathyfile, os.path.join(workdir, wd['bathy_filename']))
    force_symlink(depthfile, os.path.join(workdir, wd['depth_filename']))

    print('--------------')
    print('Preparing submission scripts for ' + hpc_cluster)

    cp = shutil.copy(wd['BS_parfile_tmp'], workdir)
    cp = shutil.copy(wd['PS_parfile_tmp'], workdir)
    cp = shutil.copy(wd['sim_postproc'], workdir)
    cp = shutil.copy(wd['final_postproc'], workdir)

    runtmp = os.path.join(workdir, wd['run_sim_filename'])
    postproc = os.path.join(workdir, wd['run_post_filename'])
    cp = shutil.copy(wd['run_sim_tmp'], runtmp) 
    cp = shutil.copy(wd['run_post_tmp'], postproc) 

    if hpc_cluster == 'leonardo':  # leonardo @CINECA#

        ngpus = 4  # 4 GPUs/node on leonardo @CINECA

        if UCmode:   # !to check
            cp = shutil.copy('sh/Step2_launch_simulARRAY_leonardo.sh', os.path.join(workdir, 'Step2_launch_simul.sh'))
            cp = shutil.copy('sh/Step2_runARRAY_tmp@leonardo.sh', runtmp) 
        filereplace(runtmp, 'leonardoACC', wd['account'])
        filereplace(runtmp, 'leonardoPART', wd['partition'])
        filereplace(runtmp, 'leonardoQOS', wd['quality'])
        filereplace(postproc, 'leonardoACC', wd['account'])
        filereplace(postproc, 'leonardoPART', wd['partition'])
        filereplace(postproc, 'leonardoQOS', wd['quality'])

    elif hpc_cluster == 'mercalli':  #Mercalli @INGV#

        ngpus = 8  # 8 GPUs/node on mercalli @INGV

    elif hpc_cluster != 'leonardo' and hpc_cluster != 'mercalli':

        print('HPC cluster for simulations not properly defined')
        sys.exit('Nothing to do....Exiting')

    filereplace(runtmp, 'LOADENV', wd['envfile'])
    filereplace(runtmp, 'PYDIR', pydir)
    # filereplace(runtmp, 'WDIR', workdir)
    filereplace(postproc, 'LOADENV', wd['envfile'])

    return  ngpus


def main(**kwargs):
    # Command args
    wd = kwargs.get('workflow_dict', None)
    inpfileBS = kwargs.get('file_simBS', None)
    inpfilePS = kwargs.get('file_simPS', None)

    wf_path = wd['wf_path']
    workdir = wd['workdir']
    inpdir = wd['inpdir']
    domain = wd['domain']
    hpc_cluster = wd['hpc_cluster']
    local_host = wd['local_host']
    UCmode = wd['UCmode']
    propagation = wd['propagation']
    sim_domain = wd['sim_domain']
    ptf_measure_type = wd['ptf_measure_type']
    
    # inpfileBS = os.path.join(workdir, wd['step1_list_BS'])
    # inpfilePS = os.path.join(workdir, wd['step1_list_PS'])

    print('Creating simulation setup')
    ngpus = create_setup(workflow_dict = wd)

    if hpc_cluster == local_host:

        print('...going to run T-HySea')
        print('--------------')
        os.chdir(workdir)
        
        if os.path.getsize(inpfileBS) > 0:   #BS
            nsce_bs, first_line_bs = get_info_from_scenario_list(filename = inpfileBS,
                                                                 seistype = 'BS')
            hysea.execute_lb_on_localhost(workflow_dict = wd,
                                          seistype      = 'BS',
                                          line          = first_line_bs)

            njobs = int(nsce_bs / ngpus) + (nsce_bs % ngpus > 0)
            print('Number of jobs = ' + str(njobs))

            hysea.submit_jobs(seistype = 'BS',
                              scenarios_file = inpfileBS,
                              workflow_dict = wd,
                              local_host = local_host,
                              hpc_cluster = hpc_cluster,
                              nscenarios = nsce_bs,
                              njobs = njobs,
                              ngpus = ngpus)

        if os.path.getsize(inpfilePS) > 0:   #PS
            nsce_ps, first_line_ps = get_info_from_scenario_list(filename = inpfilePS,
                                                                 seistype = 'PS')
            if domain == 'med-tsumaps':
                inifolder = os.path.join(inpdir, domain, wd['ps_inicond_med'])
            force_symlink(inifolder, os.path.join(workdir, wd['ps_inicond_med']))

            hysea.execute_lb_on_localhost(workflow_dict = wd,
                                          seistype      = 'PS',
                                          line          = first_line_ps)

            njobs = int(nsce_ps / ngpus) + (nsce_ps % ngpus > 0)
            print('Number of jobs = ' + str(njobs))

            hysea.submit_jobs(seistype = 'PS',
                              scenarios_file = inpfilePS,
                              workflow_dict = wd,
                              local_host = local_host,
                              hpc_cluster = hpc_cluster,
                              nscenarios = nsce_ps,
                              njobs = njobs,
                              ngpus = ngpus)

        # WAITING FOR SIMULATIONS TO FINISH
        time.sleep(10)  #wait 10 sec before checking the queue
        check_jobs = sp.getoutput(wd['job_status_cmd'] + ' | grep ' + wd['local_user'])
        while check_jobs:
            check_jobs = sp.getoutput(wd['job_status_cmd'] + ' | grep ' + wd['local_user'])
        else:
            print('Simulations completed')
        sp.run('rm -f run*sh', shell=True)
        
        # POSTPROC: Once ALL the simulations have been completed
        print('Postprocessing simulations...')

        if hpc_cluster == 'mercalli':
            cmd = 'qsub -W block=true -v ptf_measure_type=' + ptf_measure_type + ',log_failed_BS=' + wd['step2_log_failed_BS'] + ',log_failed_PS=' + wd['step2_log_failed_PS'] + \
                  ',outfile_BS=' + wd['step2_hmax_sim_BS'] + ',outfile_PS=' + wd['step2_hmax_sim_PS'] + \
                  ',file_simdir_BS=' + wd['step2_list_simdir_BS'] + ',file_simdir_PS=' + wd['step2_list_simdir_PS'] + ' ./step2_final_postproc.sh &'
        elif hpc_cluster == 'leonardo':
            cmd = 'sbatch -W ./step2_final_postproc.sh ' + ptf_measure_type + ' ' + wd['step2_log_failed_BS'] + ' ' + wd['step2_log_failed_PS'] + ' ' + \
                  wd['step2_hmax_sim_BS'] + ' ' +  wd['step2_hmax_sim_PS'] + ' ' + wd['step2_list_simdir_BS'] + ' ' + wd['step2_list_simdir_PS'] + '&'
        sp.run(cmd, shell=True)

        time.sleep(10)  #wait 10 sec before checking the queue
        check_jobs = sp.getoutput(wd['job_status_cmd'] + ' | grep ' + wd['local_user'])
        while check_jobs:
            check_jobs = sp.getoutput(wd['job_status_cmd'] + ' | grep ' + wd['local_user'])
        else:
            print('Postprocessing completed')

        os.chdir(wf_path)

    elif hpc_cluster != local_host:
        
        if os.path.getsize(inpfileBS) > 0:   #BS
            nsce_bs, first_line_bs = get_info_from_scenario_list(filename = inpfileBS,
                                                                 seistype = 'BS')
        if os.path.getsize(inpfilePS) > 0:   #PS
            nsce_ps, first_line_ps = get_info_from_scenario_list(filename = inpfilePS,
                                                                 seistype = 'PS')

        remote.connect_to_remote_cluster(workflow_dict    = wd,
                                         inpfileBS        = inpfileBS,
                                         inpfilePS        = inpfilePS,
                                         nsce_bs          = nsce_bs,
                                         nsce_ps          = nsce_ps,
                                         first_line_bs    = first_line_bs,
                                         first_line_ps    = first_line_ps,
                                         ngpus            = ngpus,
                                         ptf_measure_type = ptf_measure_type)

        print('Connection closed')

# ###########################################################
if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
