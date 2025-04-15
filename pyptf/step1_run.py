#!/usr/bin/env python

import os
import sys
import numpy as np
import scipy
# import configparser

# Import functions from pyPTF modules
from pyptf.ptf_preload                import load_PSBarInfo
from pyptf.ptf_preload                import load_Scenarios_Reg
from pyptf.ptf_preload                import load_meshes
from pyptf.ptf_preload                import load_regionalization
from pyptf.ptf_preload                import load_region_files
from pyptf.ptf_preload                import load_discretization
from pyptf.ptf_preload                import load_Model_Weights
from pyptf.ptf_define_ensemble        import define_ensemble
from pyptf.ptf_define_ensemble_global import define_ensemble_global


def main(**kwargs):

    Config        = kwargs.get('cfg', None)                # Configuration file
    workflow_dict = kwargs.get('workflow_dict', None)      # workflow dictionary
    event_dict    = kwargs.get('event_dict', None)         # event dictionary
    args          = kwargs.get('args', None)
    dataset       = kwargs.get('dataset', None)
    seismicity_types = kwargs.get('seismicity_types', None)
    pois_d        = kwargs.get('pois_d', None)
    fcp           = kwargs.get('fcp', None)
    fcp_dict      = kwargs.get('fcp_dict', None)
    logger        = kwargs.get('logger', None)
    ptf_results   = kwargs.get('ptf_results', None)
    
    ps_type = args.ps_type                 # PS probability type: 1,2. Default 1

    # Initialize and load configuration file
    # Config = configparser.RawConfigParser()
    # Config.read(cfg_file)

    ### if dataset is available we avoid to load file runtime
    if dataset:
        logger.info("Get data pre-load in memory")
        LongTermInfo = dataset['LongTermInfo']
        PSBarInfo = dataset['PSBarInfo']
        Scenarios_PS =dataset['Scenarios_PS']
        Mesh = dataset['Mesh']
        Region_files = dataset['Region_files']
    else:
        # Load info from .npy files (originally converted from sptha output)
        PSBarInfo                      = load_PSBarInfo(cfg = Config)
        Scenarios_PS                   = load_Scenarios_Reg(cfg     = Config,
                                                            type_XS = 'PS', 
                                                            logger  = logger)
        Mesh                           = load_meshes(cfg = Config, logger = logger)
        LongTermInfo                    = {}
        LongTermInfo['Regionalization'] = load_regionalization(cfg = Config, 
                                                               logger = logger)
        LongTermInfo['Discretizations'] = load_discretization(cfg = Config, 
                                                              logger = logger)
        LongTermInfo['Model_Weights']   = load_Model_Weights(cfg     = Config,
                                                             ps_type = ps_type,
                                                             logger  = logger)
        LongTermInfo['vecID']           = 10.**np.array([8, 5, 2, 0, -2, -4, -6])
        
        Region_files                   = load_region_files(cfg    = Config,
                                                           logger = logger, 
                                                           Npoly  = LongTermInfo['Regionalization']['Npoly'],
                                                           Ttype  = LongTermInfo['Regionalization']['Ttypes'])


    # define the ensemble
    if workflow_dict['ptf_version'] == 'neam':

        #check if the event depth is over the Moho limit
        moho_ll = np.column_stack((LongTermInfo['Discretizations']['BS-2_Position']['Val_x'], LongTermInfo['Discretizations']['BS-2_Position']['Val_y']))
        ev_ll = np.column_stack((event_dict['lon'], event_dict['lat']))
        ev_depth_moho = -1. * scipy.interpolate.griddata(moho_ll,
                                                         LongTermInfo['Discretizations']['BS-2_Position']['DepthMoho'],
                                                         ev_ll)[0]
        if event_dict['depth'] >  ev_depth_moho:
            logger.info('WARNING: Event too deep. Depth moved to the Moho limit as defined in NEAMTHM18')
            event_dict['depth'] = ev_depth_moho
        
        scenarios = define_ensemble(workflow_dict = workflow_dict,
                                    Scenarios_PS  = Scenarios_PS,
                                    LongTermInfo  = LongTermInfo,
                                    PSBarInfo     = PSBarInfo,
                                    Mesh          = Mesh,
                                    Region_files  = Region_files,
                                    cfg           = Config,
                                    event_data    = event_dict,
                                    pois_d        = pois_d,
                                    fcp           = fcp,
                                    fcp_dict      = fcp_dict,
                                    logger        = logger,
                                    ptf_results   = ptf_results)
        
        if not scenarios:
            return None, None, None, None, None, None, workflow_dict

        # np.save(os.path.join(workflow_dict['workdir'], workflow_dict['workflow_dictionary']), workflow_dict, allow_pickle=True)
        # bs scenarios
        if scenarios['par_scenarios_bs'].size > 0:
            scenarios['par_scenarios_bs'] = np.around(scenarios['par_scenarios_bs'], decimals=6)
            id_par_scen_bs = np.arange(start=1, stop=len(scenarios['par_scenarios_bs'])+1, step=1, dtype=float)
            scenarios['par_scenarios_bs'] = np.column_stack([id_par_scen_bs, scenarios['par_scenarios_bs']])

        # ps scenarios
        if scenarios['par_scenarios_ps'].size > 0:
            # region labels
            reg_pos = [int(ireg-1) for ireg in scenarios['par_scenarios_ps'][:,0]]
            sel_reg_id = [LongTermInfo['Regionalization']['ID'][ireg] for ireg in reg_pos]
            # magnitude labels
            magnitude_labels = LongTermInfo['Discretizations']['PS-1_Magnitude']['ID']
            magnitude_vals = LongTermInfo['Discretizations']['PS-1_Magnitude']['Val']
            sel_mag_id = [magnitude_labels[magnitude_vals.index(magval)] for magval in scenarios['par_scenarios_ps'][:,1]]
            # position labels
            position_labels = LongTermInfo['Discretizations']['PS-2_PositionArea']['ID']
            position_vals = [(x,y) for x,y in zip(LongTermInfo['Discretizations']['PS-2_PositionArea']['Val_x'], LongTermInfo['Discretizations']['PS-2_PositionArea']['Val_y'])]
            sel_pos_id = [position_labels[position_vals.index((posx,posy))].split('_')[1] for posx,posy in zip(scenarios['par_scenarios_ps'][:,2],scenarios['par_scenarios_ps'][:,3])]
            # model alternative labels
            model_vals = [int(mval) for mval in scenarios['par_scenarios_ps'][:,4]]
            model_labels = ['Str_PYes_Hom', 'Str_PNo_Hom', 'Mur_PYes_Hom', 'Mur_PNo_Hom', 'Str_PYes_Var', 'Str_PNo_Var', 'Mur_PYes_Var', 'Mur_PNo_Var']
            sel_model_id = [model_labels[k-1] for k in model_vals]
            # slip labels
            sel_slip_id = ['S00' + str(int(sval)) for sval in scenarios['par_scenarios_ps'][:,5]]

            id_scenarios_ps = np.array(np.char.array(sel_reg_id)+'-PS-'+np.char.array(sel_model_id)+'-'+sel_mag_id+'_'+sel_pos_id+'_'+sel_slip_id+'.txt')
            id_scen_reg = scenarios['par_scenarios_ps'][:,0].astype(int)
            id_par_scen_ps = np.arange(start=1, stop=len(scenarios['par_scenarios_ps'])+1, step=1, dtype=int)
            # column_stack return an array of strings
            scenarios['par_scenarios_ps'] = np.column_stack([id_par_scen_ps, id_scen_reg, id_scenarios_ps])
            
        # empty sbs for global
        scenarios['par_scenarios_sbs'] = np.empty((0, 0))
        scenarios['ProbScenSBS'] = np.empty((0))

        #return scenarios['par_scenarios_bs'], scenarios['ProbScenBS'], scenarios['par_scenarios_ps'], scenarios['ProbScenPS'], workflow_dict#, pois_d

    elif workflow_dict['ptf_version'] == 'global':
        scenarios = define_ensemble_global(workflow_dict    = workflow_dict,
                                           cfg              = Config,
                                           event_parameters = event_dict,
                                           logger            = logger)
                                           # Scenarios_PS  = Scenarios_PS,
                                           # LongTermInfo  = LongTermInfo,
                                           # PSBarInfo     = PSBarInfo,
                                           # Mesh          = Mesh,
                                           # Region_files  = Region_files,

        scenarios['par_scenarios_bs'] = np.empty((0, 0))
        scenarios['ProbScenBS'] = np.empty((0))
        scenarios['par_scenarios_ps'] = np.empty((0, 0))
        scenarios['ProbScenPS'] = np.empty((0))
        #return scenarios['par_scenarios_sbs'], scenarios['prob_scenarios_sbs'], workflow_dict
    else:
        raise Exception(f"Error value in ptf_version: {workflow_dict['ptf_version']}")

    # TODO portare ensable qui per cat_area e global
    # if samp_mode == 'SDE':

    #     negligible_prob = workflow_dict['negligible_prob']
    #     samp_mode       = workflow_dict['sampling_mode'] 
    #     samp_scen       = workflow_dict['number_of_scenarios'] 
    #     samp_type       = workflow_dict['sampling_type'] 

    #     logger.info('############## Sampling Discretized Ensemble #################')
    #     sampled_ensemble_SDE = compute_ensemble_sampling_SDE(workflow_dict    = workflow_dict,
    #                                                         LongTermInfo     = LongTermInfo,
    #                                                         negligible_prob  = negligible_prob,
    #                                                         pre_selection    = pre_selection,
    #                                                         regions          = Region_files,
    #                                                         short_term       = short_term_probability,
    #                                                         proba_scenarios  = probability_scenarios,
    #                                                         samp_scen        = samp_scen,
    #                                                         samp_type        = samp_type,
    #                                                         logger           = logger)
       
    #     scenarios = sampled_ensemble_SDE

    #most probable scenario
    max_probs = []
    for seis_type in seismicity_types:
        try:
            max_probs.append(np.amax(scenarios['ProbScen' + seis_type]))
        except:
            max_probs.append(-1.)
    max_prob = max(max_probs)
    idx_max_prob = max_probs.index(max_prob)
    seis_type_max_prob = seismicity_types[idx_max_prob]
    idx_max_prob_scen = np.argmax(scenarios['ProbScen' + seis_type_max_prob])
    par_max_prob_scen = scenarios['par_scenarios_' + seis_type_max_prob.lower()][idx_max_prob_scen]

    workflow_dict['most_probable_scenario'] = dict()
    workflow_dict['most_probable_scenario']['seis_type'] = seis_type_max_prob
    workflow_dict['most_probable_scenario']['idx_seis_type'] = idx_max_prob
    workflow_dict['most_probable_scenario']['idscen'] = idx_max_prob_scen + 1 #step1_list starts from 1
    workflow_dict['most_probable_scenario']['prob'] = max_prob
    workflow_dict['most_probable_scenario']['params'] = par_max_prob_scen

    return scenarios['par_scenarios_bs'], scenarios['ProbScenBS'], \
           scenarios['par_scenarios_ps'], scenarios['ProbScenPS'], \
           scenarios['par_scenarios_sbs'], scenarios['ProbScenSBS'], workflow_dict



if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
