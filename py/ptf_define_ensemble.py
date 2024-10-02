import os
import sys
import numpy as np
# from datetime                  import datetime

from ptf_mix_utilities         import conversion_to_utm
from ptf_lambda_bsps_load      import load_lambda_BSPS
from ptf_lambda_bsps_sep       import separation_lambda_BSPS
from ptf_pre_selection         import pre_selection_of_scenarios
from ptf_ellipsoids            import build_ellipsoid_objects
from ptf_short_term            import short_term_probability_distribution
from ptf_probability_scenarios import compute_probability_scenarios
from ptf_ensemble_sampling_SDE import compute_ensemble_sampling_SDE


#@ray.remote
def define_ensemble(**kwargs):

    workflow_dict     = kwargs.get('workflow_dict', None)
    Scenarios_PS      = kwargs.get('Scenarios_PS', None)
    LongTermInfo      = kwargs.get('LongTermInfo', None)
    POIs              = kwargs.get('POIs', None)
    PSBarInfo         = kwargs.get('PSBarInfo', None)
    Mesh              = kwargs.get('Mesh', None)
    Region_files      = kwargs.get('Region_files', None)
    Config            = kwargs.get('cfg', None)
    event_parameters  = kwargs.get('event_data', None)
    Regionalization   = kwargs.get('Regionalization', None)
    # in_json           = kwargs.get('json_rabbit', None)
    # data              = kwargs.get('data', None)

    workdir         = workflow_dict['workdir']
    sigma           = workflow_dict['sigma']
    sigma_inn       = workflow_dict['sigma_inn']
    sigma_out       = workflow_dict['sigma_out']
    negligible_prob = workflow_dict['negligible_prob']
    samp_mode       = workflow_dict['sampling_mode'] 
    samp_scen       = workflow_dict['number_of_scenarios'] 
    samp_type       = workflow_dict['sampling_type'] 


    # Initialize ptf_out dictionary
    # ptf_out = dict()

    print('Build ellipsoids objects')
    ellipses = build_ellipsoid_objects(event             = event_parameters,
                                       cfg               = Config,
                                       sigma_inn         = sigma_inn,
                                       sigma_out         = sigma_out)


    print('Conversion to utm')
    LongTermInfo, POIs, PSBarInfo = conversion_to_utm(longTerm  = LongTermInfo,
                                                      Poi       = POIs,
                                                      event     = event_parameters,
                                                      PSBarInfo = PSBarInfo)

    # Set separation of lambda BS-PS
    print('Separation of lambda BS-PS')
    lambda_bsps = load_lambda_BSPS(cfg                   = Config,
                                   sigma                 = sigma,
                                   event_parameters      = event_parameters,
                                   LongTermInfo          = LongTermInfo)


    lambda_bsps = separation_lambda_BSPS(cfg              = Config,
                                         event_parameters = event_parameters,
                                         lambda_bsps      = lambda_bsps,
                                         LongTermInfo     = LongTermInfo,
                                         mesh             = Mesh)

    ##########################################################
    # Pre-selection of the scenarios
    #
    # Magnitude: First PS then BS
    # At this moment the best solution is to insert everything into a dictionary (in matlab is the PreSelection structure)
    print('Pre-selection of the Scenarios')
    pre_selection = pre_selection_of_scenarios(cfg              = Config,
                                               sigma            = sigma,
                                               event_parameters = event_parameters,
                                               LongTermInfo     = LongTermInfo,
                                               PSBarInfo        = PSBarInfo,
                                               ellipses         = ellipses)
    if (pre_selection == False):
        workflow_dict['exit message'] = 'PTF workflow not executed because no preselected scenarios have been found'
        np.save(os.path.join(workdir, 'workflow_dictionary.npy'), workflow_dict, allow_pickle=True)
        sys.exit("No preselected scenarios have been found.")

    ##########################################################
    # COMPUTE PROB DISTR
    #
    #    Equivalent of shortterm.py with output: node_st_probabilities
    #    Output: EarlyEst.MagProb, EarlyEst.PosProb, EarlyEst.DepProb, EarlyEst.DepProb, EarlyEst.BarProb, EarlyEst.RatioBSonTot
    print('Compute short term probability distribution')


    short_term_probability  = short_term_probability_distribution(cfg              = Config,
                                                                  event_parameters = event_parameters,
                                                                  negligible_prob  = negligible_prob,
                                                                  LongTermInfo     = LongTermInfo,
                                                                  PSBarInfo        = PSBarInfo,
                                                                  lambda_bsps      = lambda_bsps,
                                                                  pre_selection    = pre_selection)
    if (short_term_probability == False):
        workflow_dict['exit message'] = 'PTF workflow not executed because no scenarios have been found by short_term_probability_distribution'
        np.save(os.path.join(workdir, 'workflow_dictionary.npy'), workflow_dict, allow_pickle=True)
        sys.exit("No scenarios have been found.")

    ##COMPUTE PROBABILITIES SCENARIOS
    print('Compute Probabilities scenarios')
    probability_scenarios = compute_probability_scenarios(cfg              = Config,
                                                          workflow_dict    = workflow_dict,
                                                          workdir          = workdir,
                                                          event_parameters = event_parameters,
                                                          negligible_prob  = negligible_prob,
                                                          LongTermInfo     = LongTermInfo,
                                                          PSBarInfo        = PSBarInfo,
                                                          lambda_bsps      = lambda_bsps,
                                                          pre_selection    = pre_selection,
                                                          regions          = Region_files,
                                                          short_term       = short_term_probability,
                                                          Scenarios_PS     = Scenarios_PS,
                                                          Regionalization  = Regionalization,
                                                          samp_mode        = samp_mode)

    if (probability_scenarios == False):
        workflow_dict['exit message'] = 'PTF workflow not executed because no scenarios have been found by compute_probability_scenarios'
        np.save(os.path.join(workdir, 'workflow_dictionary.npy'), workflow_dict, allow_pickle=True)
        file_bs_list = os.path.join(workdir, workflow_dict['step1_list_BS'])
        file_ps_list = os.path.join(workdir, workflow_dict['step1_list_PS'])
        os.remove(file_bs_list)
        os.remove(file_ps_list)
        sys.exit("No scenarios have been found.")

    ################### Sampling discretized ensemble ########################
    if samp_mode == 'SDE':
       print('############## Sampling Discretized Ensemble #################')
       sampled_ensemble_SDE = compute_ensemble_sampling_SDE(cfg              = Config,
                                                            workflow_dict    = workflow_dict,
                                                            workdir          = workdir,
                                                            event_parameters = event_parameters,
                                                            LongTermInfo     = LongTermInfo,
                                                            negligible_prob  = negligible_prob,
                                                            PSBarInfo        = PSBarInfo,
                                                            lambda_bsps      = lambda_bsps,
                                                            pre_selection    = pre_selection,
                                                            regions          = Region_files,
                                                            short_term       = short_term_probability,
                                                            Scenarios_PS     = Scenarios_PS,
                                                            Regionalization  = Regionalization,
                                                            proba_scenarios  = probability_scenarios,
                                                            samp_scen        = samp_scen,
                                                            samp_type        = samp_type)

