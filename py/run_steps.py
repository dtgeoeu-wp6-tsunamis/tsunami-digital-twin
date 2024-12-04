#!/usr/bin/env python

# Import system modules
import os
import sys
import time
import glob

sys.path.append(os.getcwd() + '/py')
#import step1_run
#import step2_simulations
import step2_simulations_compss
#import step2_precomputed
#import step3_run
#import step4_run
#import step5_run
from ptf_mix_utilities import check_previous_versions

def create_output_names(**kwargs):
    """
    """
    wd = kwargs.get('workflow_dict', None)           # workflow dictionary

    wd['step1_prob_BS'] = wd['step1_prob_BS_root'] + wd['uniqueID'] + '.npy'
    wd['step1_prob_PS'] = wd['step1_prob_PS_root'] + wd['uniqueID'] + '.npy'
    wd['step1_list_BS'] = wd['step1_list_BS_root'] + wd['uniqueID'] + '.txt'
    wd['step1_list_PS'] = wd['step1_list_PS_root'] + wd['uniqueID'] + '.txt'
    wd['step2_hmax_pre_BS'] = wd['step2_hmax_pre_BS_root'] + wd['uniqueID'] + '.nc'
    wd['step2_hmax_pre_PS'] = wd['step2_hmax_pre_PS_root'] + wd['uniqueID'] + '.nc'
    wd['step2_hmax_sim_BS'] = wd['step2_hmax_sim_BS_root'] + wd['uniqueID'] + '.nc'
    wd['step2_hmax_sim_PS'] = wd['step2_hmax_sim_PS_root'] + wd['uniqueID'] + '.nc'
    wd['step2_dir_sim_BS'] = wd['step2_dir_sim_BS_root'] + wd['uniqueID']
    wd['step2_dir_sim_PS'] = wd['step2_dir_sim_PS_root'] + wd['uniqueID']
    wd['step2_dir_log_BS'] = wd['step2_dir_log_BS_root'] + wd['uniqueID']
    wd['step2_dir_log_PS'] = wd['step2_dir_log_PS_root'] + wd['uniqueID']
    wd['step2_log_failed_BS'] = wd['step2_log_failed_BS_root'] + wd['uniqueID'] + '.log'
    wd['step2_log_failed_PS'] = wd['step2_log_failed_PS_root'] + wd['uniqueID'] + '.log'
    wd['step2_list_simdir_BS'] = wd['step2_list_simdir_BS_root'] +  wd['uniqueID'] + '.txt'
    wd['step2_list_simdir_PS'] = wd['step2_list_simdir_PS_root'] +  wd['uniqueID'] + '.txt'
    wd['step2_newsimulations_BS'] = wd['step2_newsimulations_BS_root'] + wd['uniqueID'] + '.txt'
    wd['step2_newsimulations_PS'] = wd['step2_newsimulations_PS_root'] + wd['uniqueID'] + '.txt'
    wd['step3_hc'] = wd['step3_hc_root'] + wd['uniqueID']
    wd['step3_hc_perc'] = wd['step3_hc_perc_root'] + wd['uniqueID']
    wd['step3_hazard_pdf'] = wd['step3_hazard_pdf_root'] + wd['uniqueID'] + '.npy'
    wd['step4_alert_levels_FCP'] = wd['step4_alert_levels_FCP_root'] + wd['uniqueID'] + '.npy'
    wd['step4_alert_levels_POI'] = wd['step4_alert_levels_POI_root'] + wd['uniqueID'] + '.npy'
    wd['step5_figures'] = wd['step5_figures_root'] + wd['uniqueID']
    wd['step5_alert_levels'] = wd['step5_alert_levels_root'] + wd['uniqueID']
    wd['step5_hazard_maps'] = wd['step5_hazard_maps_root'] + wd['uniqueID']
    wd['step5_hazard_curves'] = wd['step5_hazard_curves_root'] + wd['uniqueID']
    # wd['workflow_dictionary'] = wd['workflow_dictionary_root'] + wd['uniqueID']
    # wd['event_dictionary'] = wd['event_dictionary_root'] + wd['uniqueID']


def main(**kwargs):
    """
    Running steps
    """
    cfg_file      = kwargs.get('cfg_file', None)                # Configuration file
    workflow_dict = kwargs.get('workflow_dict', None)           # workflow dictionary
    event_dict    = kwargs.get('event_dict', None)              # event dictionary
    args          = kwargs.get('args', None)

    t0 = time.time()

    create_output_names(workflow_dict = workflow_dict)
    # -----------------------------------------------------------------------------
    # RUN STEPS
    # -----------------------------------------------------------------------------
    print('Event: ' + workflow_dict['uniqueID'])

    # -----------------------------------------------------------------------------
    # 1. ENSEMBLE DEFINITION
    # -----------------------------------------------------------------------------
    print('============================')
    print('========== STEP 1 ==========')
    print('============================')

    if workflow_dict['step1']:
        t = time.time()
        step1_run.main(cfg_file      = cfg_file,
                       args          = args,
                       workflow_dict = workflow_dict,
                       event_dict    = event_dict)

        print('Elapsed time: ' + str(time.time()-t) + 'sec')


    else:
        print('Step 1 not required')

    print('============================')
    print('======== END STEP 1 ========')
    print('============================')

    # -----------------------------------------------------------------------------
    # 2. Tsunami Simulations
    # -----------------------------------------------------------------------------
    print('============================')
    print('========== STEP 2 ==========')
    print('============================')

    if workflow_dict['step2']:
        t = time.time()
        
        if workflow_dict['tsu_sim'] == 'to_run':   # run on-the-fly tsunami simulations
            # Check if previous versions of the same event already exist
            file_simBS, file_simPS = check_previous_versions(wd = workflow_dict)

            print('Performing on-the-fly tsunami simulations')

            #step2_simulations.main(cfg_file      = cfg_file,
# TESTING COMPSS
#            step2_simulations.main(workflow_dict = workflow_dict,
#                                   file_simBS    = file_simBS,
#                                   file_simPS    = file_simPS)
            
            step2_simulations_compss.main(workflow_dict = workflow_dict,
                                          file_simBS    = file_simBS,
                                          file_simPS    = file_simPS)
                                   

        elif workflow_dict['tsu_sim'] == 'precomputed':   # loading pre-computed scenarios

            print('Retrieving tsunami simulations from regional hazard database')

            step2_precomputed.main(cfg_file      = cfg_file,
                                   workflow_dict = workflow_dict)

        print('Elapsed time: ' + str(time.time()-t) + 'sec')
    else:
        print('Step 2 not required')

    print('============================')
    print('======== END STEP 2 ========')
    print('============================')

#    # -----------------------------------------------------------------------------
#    # 2.1 Misfit Evaluator
#    # -----------------------------------------------------------------------------
#    if workflow_dict['misfit']:
#        print('Computing the misfit')

    # -----------------------------------------------------------------------------
    # 3. Hazard Aggregation
    # -----------------------------------------------------------------------------
    print('============================')
    print('========== STEP 3 ==========')
    print('============================')

    if workflow_dict['step3']:
        t = time.time()
        step3_run.main(cfg_file      = cfg_file,
                       workflow_dict = workflow_dict)

        print('Elapsed time: ' + str(time.time()-t) + 'sec')
    else:
        print('Step 3 not required')

    print('============================')
    print('======== END STEP 3 ========')
    print('============================')

    # -----------------------------------------------------------------------------
    # 4. Alert Levels
    # -----------------------------------------------------------------------------
    print('============================')
    print('========== STEP 4 ==========')
    print('============================')

    if workflow_dict['step4']:
        t = time.time()
        step4_run.main(cfg_file      = cfg_file,
                       workflow_dict = workflow_dict,
                       event_dict    = event_dict)

        print('Elapsed time: ' + str(time.time()-t) + 'sec')
    else:
        print('Step 4 not required')

    print('============================')
    print('======== END STEP 4 ========')
    print('============================')

    # -----------------------------------------------------------------------------
    # 5. Visualization
    # -----------------------------------------------------------------------------
    print('============================')
    print('========== STEP 5 ==========')
    print('============================')

    if workflow_dict['step5']:
        t = time.time()
        step5_run.main(cfg_file      = cfg_file,
                       workflow_dict = workflow_dict,
                       event_dict    = event_dict)

        print('Elapsed time: ' + str(time.time()-t) + 'sec')
    else:
        print('Step 5 not required')

    print('============================')
    print('======== END STEP 5 ========')
    print('============================')

    print('Total elapsed time: ' + str(time.time()-t0) + 'sec')

#if __name__ == "__main__":
#    main(**dict(arg.split('=') for arg in sys.argv[1:]))
