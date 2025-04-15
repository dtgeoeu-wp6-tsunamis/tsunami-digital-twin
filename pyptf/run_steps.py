#!/usr/bin/env python

# Import system modules
import os
import sys
import time
import numpy as np

import pyptf.step1_run         as step1_run 
import pyptf.step2_simulations as step2_simulations
import pyptf.step2_precomputed as step2_precomputed
import pyptf.step3_run         as step3_run
import pyptf.step4_run         as step4_run
import pyptf.step5_run         as step5_run

from pyptf.ptf_mix_utilities import check_previous_versions
from pyptf.scenario_player import ScenarioPlayer
import pyptf.deformation as deformation
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
    if workflow_dict['tsu_sim'] == 'to_run':
        # Scenario Player
        logger.info('Executing scenario player to retrieve data for the event')
        player = ScenarioPlayer(data_path = os.path.join(workflow_dict['inpdir'], 'dtgeo/Data_Archive'))
        sealevel_dict, gnss_df = player.run_scenario_player(event_dict = event_dict)
        deformation_dict = deformation.calculate(par_bs = par_bs,
                                              par_ps = par_ps,
                                              par_sbs = par_sbs,
                                              gnss_df = gnss_df)

        # create output folder for sea level and gnss data
        data_dir = os.path.join(workflow_dict['workdir'], 'data')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        np.save(os.path.join(data_dir, 'sealevel_data.npy'), sealevel_dict, allow_pickle=True)
        gnss_df.to_csv(os.path.join(data_dir, 'gnss_data.csv'))
        np.save(os.path.join(data_dir, 'deformation_synthetic.npy'), deformation_dict, allow_pickle=True)


#    if workflow_dict['misfit']:
#        logger.info('Computing the misfit')

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
