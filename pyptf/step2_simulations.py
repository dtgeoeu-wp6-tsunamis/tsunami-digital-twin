#!/usr/bin/env python

# Import system modules
import os
import sys
# import shutil
import time
import numpy as np
import h5py
import subprocess as sp

import pyptf.tsunami_simulations as tsu_sim
import pyptf.step2_remote_connection as remote


def main(**kwargs):
    # Command args
    wd = kwargs.get('workflow_dict', None)
    seismicity_types = kwargs.get('seis_types', None)
    inpfileBS = kwargs.get('file_simBS', None)
    inpfilePS = kwargs.get('file_simPS', None)
    inpfileSBS = kwargs.get('file_simSBS', None)
    pois_d = kwargs.get('pois_d', None)
    logger = kwargs.get('logger', None)

    wf_path = wd['wf_path']
    workdir = wd['workdir']
    hpc_cluster = wd['hpc_cluster']
    local_host = wd['local_host']
    ptf_measure_type = wd['ptf_measure_type']
    
    logger.info('Creating simulation setup')
    nscen, first_lines, ngpus = tsu_sim.create_setup(workflow_dict = wd,
                                                     seis_types    = seismicity_types,
                                                     inpfileBS     = inpfileBS,
                                                     inpfilePS     = inpfilePS,
                                                     inpfileSBS    = inpfileSBS,
                                                     pois_d        = pois_d,
                                                     logger        = logger)

    input_files = {'BS': inpfileBS, 'PS': inpfilePS, 'SBS': inpfileSBS}

    if hpc_cluster == local_host:

        logger.info('...going to run T-HySea')
        logger.info('--------------')
        os.chdir(workdir)

        for seistype in seismicity_types:

            if nscen[seistype] != 0:
                tsu_sim.execute_lb_on_localhost(workflow_dict = wd,
                                                seistype      = seistype,
                                                line          = first_lines[seistype])

                njobs = int(nscen[seistype] / ngpus) + (nscen[seistype] % ngpus > 0)
                logger.info('Number of jobs = ' + str(njobs))

                tsu_sim.submit_jobs(seistype = seistype,
                                    scenarios_file = input_files[seistype],
                                    workflow_dict = wd,
                                    local_host = local_host,
                                    hpc_cluster = hpc_cluster,
                                    nscenarios = nscen[seistype],
                                    njobs = njobs,
                                    ngpus = ngpus)

        # WAITING FOR SIMULATIONS TO FINISH
        time.sleep(10)  #wait 10 sec before checking the queue
        check_jobs = sp.getoutput(wd['job_status_cmd'] + ' | grep ' + wd['local_user'])
        while check_jobs:
            check_jobs = sp.getoutput(wd['job_status_cmd'] + ' | grep ' + wd['local_user'])
        else:
            logger.info('Simulations completed')
        sp.run('rm -f run*sh', shell=True)
        
        # POSTPROC: Once ALL the simulations have been completed
        logger.info('Postprocessing simulations...')

        if hpc_cluster == 'mercalli':
            cmd = 'qsub -W block=true -v ptf_measure_type=' + ptf_measure_type + \
                  ',log_failed_BS=' + wd['step2_log_failed_BS'] + ',log_failed_PS=' + wd['step2_log_failed_PS'] + ',log_failed_SBS=' + wd['step2_log_failed_SBS'] + \
                  ',outfile_BS=' + wd['step2_hmax_sim_BS'] + ',outfile_PS=' + wd['step2_hmax_sim_PS'] + ',outfile_SBS=' + wd['step2_hmax_sim_SBS'] + \
                  ',file_simdir_BS=' + wd['step2_list_simdir_BS'] + ',file_simdir_PS=' + wd['step2_list_simdir_PS'] + ',file_simdir_SBS=' + wd['step2_list_simdir_SBS'] + \
                  ' ./step2_final_postproc.sh &'
        elif hpc_cluster == 'leonardo':
            cmd = 'sbatch -W ./step2_final_postproc.sh ' + ptf_measure_type + ' ' + \
                  wd['step2_log_failed_BS'] + ' ' + wd['step2_log_failed_PS'] + ' ' + wd['step2_log_failed_SBS'] + ' ' + \
                  wd['step2_hmax_sim_BS'] + ' ' +  wd['step2_hmax_sim_PS'] + ' ' + wd['step2_hmax_sim_SBS'] + ' ' + \
                  wd['step2_list_simdir_BS'] + ' ' + wd['step2_list_simdir_PS'] + ' ' + wd['step2_list_simdir_SBS'] + '&'
        sp.run(cmd, shell=True)

        time.sleep(10)  #wait 10 sec before checking the queue
        check_jobs = sp.getoutput(wd['job_status_cmd'] + ' | grep ' + wd['local_user'])
        while check_jobs:
            check_jobs = sp.getoutput(wd['job_status_cmd'] + ' | grep ' + wd['local_user'])
        else:
            logger.info('Postprocessing completed')

    elif hpc_cluster != local_host:
        
        remote.connect_to_remote_cluster(workflow_dict    = wd,
                                         input_files      = input_files,
                                         nscen            = nscen,
                                         first_lines      = first_lines,
                                         seismicity_types = seismicity_types,
                                         ngpus            = ngpus,
                                         ptf_measure_type = ptf_measure_type)

        logger.info('Connection closed')

    # loading mih resulting from simulations to return
    mih_sim = dict()
    for seistype in seismicity_types:
        if nscen[seistype]  != 0:
            file_hmax_sim = os.path.join(workdir, wd['step2_hmax_sim_' + seistype])
            try:
                imax_scenarios = h5py.File(file_hmax_sim, 'r')
                mih_sim[seistype] = np.array(imax_scenarios[ptf_measure_type])
            except:
                raise Exception(f"Error reading file: {file_hmax_sim}")
        else:
            mih_sim[seistype] = np.empty((0, 0))

    
    os.chdir(wf_path)

    return mih_sim['BS'], mih_sim['PS'], mih_sim['SBS']


if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
