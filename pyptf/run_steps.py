#!/usr/bin/env python

# Import system modules
import os
import sys
import time
import numpy as np
import shutil
import subprocess as sp
from pyutil import filereplace

import pyptf.step1_run         as step1_run 
import pyptf.step2_simulations as step2_simulations
import pyptf.step2_precomputed as step2_precomputed
import pyptf.step3_run         as step3_run
import pyptf.step4_run         as step4_run
import pyptf.step5_run         as step5_run

from pyptf.ptf_mix_utilities import check_previous_versions
from pyptf.scenario_player import ScenarioPlayer
import pyptf.deformation as deformation
import pyptf.misfit_evaluator as misfit
# from pyptf.pyptf_exceptions  import PyPTFException


def main(**kwargs):
    """
    Running steps
    """
    cfg           = kwargs.get('cfg', None)                # Configuration file
    workflow_dict = kwargs.get('workflow_dict', None)           # workflow dictionary
    event_dict    = kwargs.get('event_dict', None)              # event dictionary
    pois_d        = kwargs.get('pois_d', None)
    fcp           = kwargs.get('fcp', None)
    fcp_dict      = kwargs.get('fcp_dict', None)
    thresholds    = kwargs.get('thresholds', None)
    args          = kwargs.get('args', None)
    ptf_results   = kwargs.get('ptf_results', None)
    logger        = kwargs.get('logger', None)
    dataset       = kwargs.get('dataset', None)

    t0 = time.time()
    seismicity_types = ['BS', 'PS', 'SBS']

    # -----------------------------------------------------------------------------
    # RUN STEPS
    # -----------------------------------------------------------------------------
    logger.info('Event: ' + workflow_dict['uniqueID'])

    # -----------------------------------------------------------------------------
    # 1. ENSEMBLE DEFINITION
    # -----------------------------------------------------------------------------
    logger.info('============================')
    logger.info('========== STEP 1 ==========')
    logger.info('============================')

    if workflow_dict['step1']:
        t = time.time()
        par_bs, prob_bs, par_ps, prob_ps, par_sbs, prob_sbs, workflow_dict = step1_run.main(cfg              = cfg,
                                                                                            args             = args,
                                                                                            workflow_dict    = workflow_dict,
                                                                                            event_dict       = event_dict,
                                                                                            dataset          = dataset,
                                                                                            seismicity_types = seismicity_types,
                                                                                            pois_d           = pois_d,
                                                                                            fcp              = fcp,
                                                                                            fcp_dict         = fcp_dict,
                                                                                            logger           = logger,
                                                                                            ptf_results      = ptf_results)

        if ptf_results.status is None:
            ptf_results.set_result(step_name    = 'step1', 
                                   result_key   = 'par_scen_bs', 
                                   result_value = par_bs)
        
            ptf_results.set_result(step_name    = 'step1', 
                                   result_key   = 'prob_scen_bs', 
                                   result_value = prob_bs)
        
            ptf_results.set_result(step_name    = 'step1', 
                                   result_key   = 'par_scen_ps', 
                                   result_value = par_ps)
        
            ptf_results.set_result(step_name    = 'step1', 
                                   result_key   = 'prob_scen_ps', 
                                   result_value = prob_ps)
        
            ptf_results.set_result(step_name    = 'step1', 
                                   result_key   = 'par_scen_sbs', 
                                   result_value = par_sbs)
        
            ptf_results.set_result(step_name    = 'step1', 
                                   result_key   = 'prob_scen_sbs', 
                                   result_value = prob_sbs)
        
            ptf_results.set_result(step_name    = 'step1', 
                                   result_key   = 'workflow_dict', 
                                   result_value = workflow_dict)
        
        logger.info('Elapsed time: ' + str(time.time()-t) + 'sec')

    else:
        par_bs = None
        par_ps = None
        par_sbs = None
        prob_bs = None
        prob_ps = None
        prob_sbs = None
        logger.info('Step 1 not required')

    logger.info('============================')
    logger.info('======== END STEP 1 ========')
    logger.info('============================')

    # -----------------------------------------------------------------------------
    # 2. Tsunami Simulations
    # -----------------------------------------------------------------------------
    logger.info('============================')
    logger.info('========== STEP 2 ==========')
    logger.info('============================')

    if workflow_dict['step2']:
        t = time.time()
        
        if workflow_dict['tsu_sim'] == 'to_run':   # run on-the-fly tsunami simulations
            # seismicity_types = ['BS', 'PS', 'SBS']

            # Check if previous versions of the same event already exist
            file_simBS, file_simPS, file_simSBS = check_previous_versions(wd = workflow_dict,
                                                                          seis_types = seismicity_types,
                                                                          par_bs = par_bs,  
                                                                          par_ps = par_ps,
                                                                          par_sbs = par_sbs,
                                                                          logger = logger)
            logger.info('Performing on-the-fly tsunami simulations')

            #step2_simulations.main(cfg_file      = cfg_file,
            mih_bs, mih_ps, mih_sbs = step2_simulations.main(workflow_dict = workflow_dict,
                                                             seis_types = seismicity_types,
                                                             file_simBS = file_simBS,
                                                             file_simPS = file_simPS,
                                                             file_simSBS = file_simSBS,
                                                             pois_d = pois_d, 
                                                             logger = logger)
            
        elif workflow_dict['tsu_sim'] == 'precomputed':   # loading pre-computed scenarios

            logger.info('Retrieving tsunami simulations from regional hazard database')

            mih_bs, mih_ps, mih_sbs = step2_precomputed.main(cfg           = cfg,
                                                             workflow_dict = workflow_dict,
                                                             par_bs        = par_bs,  
                                                             par_ps        = par_ps,
                                                             pois_d        = pois_d, 
                                                             dataset       = dataset,
                                                             logger        = logger)
            
        # set results for both simulations and linear combinations
        ptf_results.set_result(step_name    = 'step2', 
                               result_key   = 'mih_bs', 
                               result_value = mih_bs)
        
        ptf_results.set_result(step_name    = 'step2', 
                               result_key   = 'mih_ps', 
                               result_value = mih_ps)
        
        ptf_results.set_result(step_name    = 'step2', 
                                result_key   = 'mih_sbs', 
                                result_value = mih_sbs)
        
        logger.info('Elapsed time: ' + str(time.time()-t) + 'sec')
    else:
        mih_bs = None
        mih_ps = None
        mih_sbs = None
        logger.info('Step 2 not required')

    logger.info('============================')
    logger.info('======== END STEP 2 ========')
    logger.info('============================')

    # -----------------------------------------------------------------------------
    # 2.1 Scenario Player + Misfit Evaluator (DT-GEO)
    # -----------------------------------------------------------------------------
    if workflow_dict['compute_misfit'] and workflow_dict['tsu_sim'] == 'to_run':
        
        # create output folder for sea level and gnss data
        data_dir = os.path.join(workflow_dict['workdir'], 'data')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        # Scenario Player
        logger.info('Executing scenario player to retrieve data for the event')
        player = ScenarioPlayer(data_path = os.path.join(workflow_dict['inpdir'], 'dtgeo/Data_Archive'))
        sealevel_dict, gnss_df = player.run_scenario_player(event_dict = event_dict)
        np.save(os.path.join(data_dir, 'sealevel_data.npy'), sealevel_dict, allow_pickle=True)
        if not gnss_df.empty:
            deformation_dict = deformation.calculate(par_bs = par_bs,
                                                     par_ps = par_ps,
                                                     par_sbs = par_sbs,
                                                     gnss_df = gnss_df)
            np.save(os.path.join(data_dir, 'deformation_synthetic.npy'), deformation_dict, allow_pickle=True)
            gnss_df.to_csv(os.path.join(data_dir, 'gnss_data.csv'))

        # Search for the closest POIs to the sea level stations 
        pois_d = np.load(os.path.join(workflow_dict['workdir'], 'pois_fixed.npy'), allow_pickle=True).item() #fix cile and tohoku
        gauge_pois = misfit.find_closest_pois(sealevel_dict = sealevel_dict,
                                              pois_d        = pois_d,
                                              logger        = logger)
        np.save(os.path.join(data_dir, 'sealevel_gauge_pois.npy'), gauge_pois)

        # Misfit Evaluator
        logger.info('Executing misfit evaluator to compute the misfit')
        sim_folder_BS = os.path.join(workflow_dict['workdir'], workflow_dict['step2_dir_sim_BS'])
        sim_folder_PS = os.path.join(workflow_dict['workdir'], workflow_dict['step2_dir_sim_PS'])
        sim_folder_SBS = os.path.join(workflow_dict['workdir'], workflow_dict['step2_dir_sim_SBS'])

        if workflow_dict['hpc_cluster'] == 'mercalli':
            misfit.main(workdir = workflow_dict['workdir'],
                        gauge_data_dict = sealevel_dict,
                        sim_folder_BS = sim_folder_BS,
                        sim_folder_PS = sim_folder_PS,
                        sim_folder_SBS = sim_folder_SBS,
                        gauge_pois = gauge_pois,
                        synthetic_gauge = False, #Handle whether the gauge data is synthetic or real gauge data. True/False
                        statistical_misfit = False, #Handle whether the synthetic gauge data will be statistical (misfit analysis at all POIs) or not. True/False
                        arrival_time_percentage = 0.1, #Percentage of the maximum wave height at which the arrival time is picked
                        plot_arrivaltimes = False, #Option for plotting of the gauge data with picked arrival times. True/False
                        )
        elif workflow_dict['hpc_cluster'] == 'leonardo':
            run_misfit = os.path.join(workflow_dict['workdir'], 'misfit.sh')
            cp = shutil.copy(workflow_dict['run_misfit_tmp'], run_misfit)
            filereplace(run_misfit, 'LOADENV', workflow_dict['envfile'])
            filereplace(run_misfit, 'WFDIR', workflow_dict['wf_path'])
            filereplace(run_misfit, 'leonardoACC', workflow_dict['account'])
            filereplace(run_misfit, 'leonardoPART', workflow_dict['partition'])
            filereplace(run_misfit, 'leonardoQOS', workflow_dict['quality'])
            os.chdir(workflow_dict['workdir'])
            cmd = 'sbatch -W ./misfit.sh ' + workflow_dict['workdir'] + ' ' + \
                  sim_folder_BS + ' ' + sim_folder_PS + ' ' + sim_folder_SBS + ' &'
            sp.run(cmd, shell=True)
            os.chdir(workflow_dict['wf_path'])
        #SAMOS TEST
        # gauge_pois = [2115, 3800, 2053, 3337, 1755]  #samos test
        # sealevel_dict_samos = np.load('/leonardo_work/DTGEO_T1_2/samos/data/sealevel_data.npy', allow_pickle=True).item()
        # misfit.main(workdir = workflow_dict['workdir'],
        #             gauge_data_dict = sealevel_dict_samos,
        #             sim_folder_BS = '/leonardo_work/DTGEO_T1_2/samos/25scenarios/',
        #             sim_folder_PS = os.path.join(workflow_dict['workdir'], workflow_dict['step2_dir_sim_PS']),
        #             sim_folder_SBS = os.path.join(workflow_dict['workdir'], workflow_dict['step2_dir_sim_BS']), 
        #             gauge_pois = gauge_pois,
        #             synthetic_gauge = False, #Handle whether the gauge data is synthetic or real gauge data. True/False
        #             statistical_misfit = False, #Handle whether the synthetic gauge data will be statistical (misfit analysis at all POIs) or not. True/False
        #             arrival_time_percentage = 0.1, #Percentage of the maximum wave height at which the arrival time is picked
        #             plot_arrivaltimes = False, #Option for plotting of the gauge data with picked arrival times. True/False
        #             )

    # -----------------------------------------------------------------------------
    # 3. Hazard Aggregation
    # -----------------------------------------------------------------------------
    logger.info('============================')
    logger.info('========== STEP 3 ==========')
    logger.info('============================')

    if workflow_dict['step3']:
        t = time.time()
        hc_pois, hc_d, pdf_pois = step3_run.main(workflow_dict = workflow_dict,
                                                 mih_bs        = mih_bs,
                                                 mih_ps        = mih_ps,
                                                 mih_sbs       = mih_sbs,
                                                 prob_bs       = prob_bs,
                                                 prob_ps       = prob_ps,
                                                 prob_sbs      = prob_sbs,
                                                 thresholds    = thresholds,
                                                 pois_d        = pois_d,
                                                 logger        = logger)

        ptf_results.set_result(step_name    = 'step3', 
                                    result_key   = 'hc_pois', 
                                    result_value = hc_pois)

        ptf_results.set_result(step_name    = 'step3', 
                                    result_key   = 'hc_d', 
                                    result_value = hc_d)

        ptf_results.set_result(step_name    = 'step3', 
                                    result_key   = 'pdf_pois', 
                                    result_value = pdf_pois)

        logger.info('Elapsed time: ' + str(time.time()-t) + 'sec')
    else:
        hc_pois = None
        hc_d = None
        pdf_pois = None
        logger.info('Step 3 not required')

    logger.info('============================')
    logger.info('======== END STEP 3 ========')
    logger.info('============================')

    # -----------------------------------------------------------------------------
    # 4. Alert Levels
    # -----------------------------------------------------------------------------
    logger.info('============================')
    logger.info('========== STEP 4 ==========')
    logger.info('============================')

    if workflow_dict['step4']:
        t = time.time()
        pois_alert_levels, fcp_alert_levels, fcp_names, fcp_coordinates, fcp_ids = step4_run.main(workflow_dict = workflow_dict,
                                                                                                  event_dict    = event_dict,
                                                                                                  pois_d        = pois_d,
                                                                                                  fcp           = fcp,
                                                                                                  fcp_dict      = fcp_dict,
                                                                                                  hc_d          = hc_d,
                                                                                                  logger        = logger)

        ptf_results.set_result(step_name    = 'step4', 
                               result_key   = 'pois_alert_levels', 
                               result_value = pois_alert_levels)

        ptf_results.set_result(step_name    = 'step4', 
                               result_key   = 'fcp_alert_levels', 
                               result_value = fcp_alert_levels)

        ptf_results.set_result(step_name    = 'step4', 
                               result_key   = 'fcp_names', 
                               result_value = fcp_names)

        ptf_results.set_result(step_name    = 'step4', 
                               result_key   = 'fcp_coordinates', 
                               result_value = fcp_coordinates)

        ptf_results.set_result(step_name    = 'step4', 
                               result_key   = 'fcp_ids', 
                               result_value = fcp_ids)

        logger.info('Elapsed time: ' + str(time.time()-t) + 'sec')
    else:
        pois_alert_levels = None 
        fcp_alert_levels = None
        fcp_names = None
        fcp_coordinates = None
        fcp_ids = None
        logger.info('Step 4 not required')

    logger.info('============================')
    logger.info('======== END STEP 4 ========')
    logger.info('============================')

    # -----------------------------------------------------------------------------
    # 5. Visualization
    # -----------------------------------------------------------------------------
    logger.info('============================')
    logger.info('========== STEP 5 ==========')
    logger.info('============================')

    if workflow_dict['step5']:
        t = time.time()
        step5_run.main(workflow_dict = workflow_dict,
                       event_dict    = event_dict,
                       pois_d        = pois_d,
                       fcp           = fcp,
                       hc_pois       = hc_pois,
                       hc_d          = hc_d,
                       thresholds    = thresholds,
                       pois_al       = pois_alert_levels,
                       fcp_al        = fcp_alert_levels,
                       logo_png      = cfg.get('Files', 'logo'),
                       logger        = logger,
                       ptf_results   = ptf_results)

        logger.info('Elapsed time: ' + str(time.time()-t) + 'sec')
    else:
        logger.info('Step 5 not required')

    logger.info('============================')
    logger.info('======== END STEP 5 ========')
    logger.info('============================')

    if ptf_results.status is None:
        ptf_results.set_status(status='OK') 

    logger.info('Total elapsed time: ' + str(time.time()-t0) + 'sec')

#if __name__ == "__main__":
#    main(**dict(arg.split('=') for arg in sys.argv[1:]))
