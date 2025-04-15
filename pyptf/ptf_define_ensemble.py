# import os
# import sys
# import numpy as np

from pyptf.ptf_mix_utilities         import conversion_to_utm
from pyptf.ptf_lambda_bsps_load      import load_lambda_BSPS
from pyptf.ptf_lambda_bsps_sep       import separation_lambda_BSPS
from pyptf.ptf_pre_selection         import pre_selection_of_scenarios
from pyptf.ptf_ellipsoids            import build_ellipsoid_objects
from pyptf.ptf_short_term            import short_term_probability_distribution
from pyptf.ptf_probability_scenarios import compute_probability_scenarios
from pyptf.ptf_ensemble_sampling_SDE import compute_ensemble_sampling_SDE
from pyptf.ptf_matrix                import return_matrix
from pyptf.pyptf_exceptions          import PyPTFException

#@ray.remote
def define_ensemble(**kwargs):

    workflow_dict     = kwargs.get('workflow_dict', None)
    Scenarios_PS      = kwargs.get('Scenarios_PS', None)
    LongTermInfo      = kwargs.get('LongTermInfo', None)
    PSBarInfo         = kwargs.get('PSBarInfo', None)
    Mesh              = kwargs.get('Mesh', None)
    Region_files      = kwargs.get('Region_files', None)
    Config            = kwargs.get('cfg', None)
    event_parameters  = kwargs.get('event_data', None)
    pois_d            = kwargs.get('pois_d', None)
    fcp               = kwargs.get('fcp', None)
    fcp_dict          = kwargs.get('fcp_dict', None)
    logger            = kwargs.get('logger', None)
    ptf_results       = kwargs.get('ptf_results', None)

    # workdir         = workflow_dict['workdir']
    sigma           = workflow_dict['sigma']
    sigma_inn       = workflow_dict['sigma_inn']
    sigma_out       = workflow_dict['sigma_out']
    negligible_prob = workflow_dict['negligible_prob']
    samp_mode       = workflow_dict['sampling_mode'] 
    samp_scen       = workflow_dict['number_of_scenarios'] 
    samp_type       = workflow_dict['sampling_type'] 


    # Initialize ptf_out dictionary
    # ptf_out = dict()

    logger.info('Build ellipsoids objects')
    ellipses = build_ellipsoid_objects(event     = event_parameters,
                                       cfg       = Config,
                                       sigma_inn = sigma_inn,
                                       sigma_out = sigma_out)


    logger.info('Conversion to utm')
    LongTermInfo, PSBarInfo = conversion_to_utm(longTerm  = LongTermInfo,
                                                event     = event_parameters,
                                                PSBarInfo = PSBarInfo)


    # Set separation of lambda BS-PS
    logger.info('Separation of lambda BS-PS')
    lambda_bsps = load_lambda_BSPS(cfg              = Config,
                                   sigma            = sigma,
                                   event_parameters = event_parameters,
                                   LongTermInfo     = LongTermInfo,
                                   logger           = logger)


    lambda_bsps = separation_lambda_BSPS(cfg              = Config,
                                         event_parameters = event_parameters,
                                         lambda_bsps      = lambda_bsps,
                                         LongTermInfo     = LongTermInfo,
                                         mesh             = Mesh,
                                         logger           = logger)
    
    # Pre-selection of the scenarios
    logger.info('Pre-selection of the Scenarios')
    pre_selection = pre_selection_of_scenarios(sigma            = sigma,
                                               event_parameters = event_parameters,
                                               LongTermInfo     = LongTermInfo,
                                               PSBarInfo        = PSBarInfo,
                                               ellipses         = ellipses,
                                               logger           = logger)
    if (pre_selection == False):
    #     raise PyPTFException("PTF workflow not executed because no preselected scenarios have been found.")
        ptf_results.set_status(status='No results for PTF: event magnitude out of the NEAMTHM18 discretization used in PTF.')
        workflow_dict['step2'] = False
        workflow_dict['step3'] = False
        workflow_dict['step4'] = False
        workflow_dict['step5'] = False

        if  event_parameters['area'] == 'cat_area':
            return_matrix(workflow_dict = workflow_dict,
                          event_dict = event_parameters,
                          pois_d = pois_d,
                          fcp = fcp,
                          fcp_dict = fcp_dict,
                          ptf_results = ptf_results,
                          logo_png = Config.get('Files', 'logo'),
                          logger = logger)
        ptf_results = {'step1': {}, 'step2': {}, 'step3': {}}

        return False

    ##########################################################
    # COMPUTE PROB DISTR
    #
    #    Equivalent of shortterm.py with output: node_st_probabilities
    #    Output: EarlyEst.MagProb, EarlyEst.PosProb, EarlyEst.DepProb, EarlyEst.DepProb, EarlyEst.BarProb, EarlyEst.RatioBSonTot
    logger.info('Compute short term probability distribution')

    short_term_probability  = short_term_probability_distribution(cfg              = Config,
                                                                  event_parameters = event_parameters,
                                                                  negligible_prob  = negligible_prob,
                                                                  LongTermInfo     = LongTermInfo,
                                                                  PSBarInfo        = PSBarInfo,
                                                                  lambda_bsps      = lambda_bsps,
                                                                  pre_selection    = pre_selection,
                                                                  logger           = logger)
    if (short_term_probability == False):
        raise PyPTFException("PTF workflow not executed because no scenarios have been found by short_term_probability_distribution.")

    ##COMPUTE PROBABILITIES SCENARIOS
    logger.info('Compute Probabilities scenarios')
    probability_scenarios = compute_probability_scenarios(workflow_dict    = workflow_dict,
                                                          LongTermInfo     = LongTermInfo,
                                                          PSBarInfo        = PSBarInfo,
                                                          lambda_bsps      = lambda_bsps,
                                                          pre_selection    = pre_selection,
                                                          regions          = Region_files,
                                                          short_term       = short_term_probability,
                                                          Scenarios_PS     = Scenarios_PS,
                                                          samp_mode        = samp_mode,
                                                          logger           = logger)

    if (probability_scenarios == False):
        raise PyPTFException("PTF workflow not executed because no scenarios have been found by compute_probability_scenarios.")

    ################### Sampling discretized ensemble ########################
    if samp_mode == 'SDE':
        logger.info('############## Sampling Discretized Ensemble #################')
        sampled_ensemble_SDE = compute_ensemble_sampling_SDE(workflow_dict    = workflow_dict,
                                                            LongTermInfo     = LongTermInfo,
                                                            negligible_prob  = negligible_prob,
                                                            pre_selection    = pre_selection,
                                                            regions          = Region_files,
                                                            short_term       = short_term_probability,
                                                            proba_scenarios  = probability_scenarios,
                                                            samp_scen        = samp_scen,
                                                            samp_type        = samp_type,
                                                            logger           = logger)
       
        scenarios = sampled_ensemble_SDE
    else:
        scenarios = probability_scenarios

    return scenarios
