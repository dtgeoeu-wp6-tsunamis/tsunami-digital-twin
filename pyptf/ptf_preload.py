#!/usr/bin/env python

# Import system modules
import os
import ast
import sys
import glob
import numpy as np


def load_Model_Weights(**kwargs):

    Config  = kwargs.get('cfg', None)
    ps_type = kwargs.get('ps_type', None)
    logger  = kwargs.get('logger', None)
    
    weight_npy  = Config.get('pyptf','ModelWeight')
    empty_space = "%10s" % ('')

    logger.info('Loading model weights dictionary:      <------ {}'.format(weight_npy))
    foe = np.load(weight_npy, allow_pickle=True)
    modelweights = foe.item()
    for k in modelweights.keys():
        logger.info('{} {}'.format(empty_space, k))

    # select only one type of ps_weigth
    # Only one weigth mode can be selected
    #if(int(ps_type) != -1):
    selected_index = np.where(modelweights['BS1_Mag']['Type'] == int(ps_type))
    modelweights['BS1_Mag']['Type'] = modelweights['BS1_Mag']['Type'][selected_index]
    modelweights['BS1_Mag']['Wei'] = modelweights['BS1_Mag']['Wei'][selected_index]

    selected_index = np.where(modelweights['PS2_Bar']['Type'] == int(ps_type))
    modelweights['PS2_Bar']['Type'] = modelweights['PS2_Bar']['Type'][selected_index]
    modelweights['PS2_Bar']['Wei'] = modelweights['PS2_Bar']['Wei'][selected_index]

    return modelweights


def load_discretization(**kwargs):

    Config = kwargs.get('cfg', None)
    logger = kwargs.get('logger', None)

    discretization_npy = Config.get('pyptf','Discretization_npy')
    empty_space = "%10s" % ('')

    logger.info('Loading discretization dictionary:     <------ {}'.format(discretization_npy))
    foe = np.load(discretization_npy, allow_pickle=True)
    discretizations = foe.item()
    for i in discretizations.keys():
        logger.info('{} {}'.format(empty_space, i))

    return discretizations


def load_regionalization(**kwargs):

    Config = kwargs.get('cfg', None)
    logger = kwargs.get('logger', None)
    
    regionalization_npy = Config.get('pyptf','Regionalization_npy')

    logger.info('Load dictionary for regionalization:   <------ {} '.format(regionalization_npy))
    # load npy file into a dictionary. The option allow_pickle=True is needed if the npy file contains np.array
    foe = np.load(regionalization_npy, allow_pickle=True)
    regionalization = foe.item()
    logger.info("Regions found: {:3d}".format(regionalization['Npoly']))

    return regionalization


def load_region_files(**kwargs):

    Config       = kwargs.get('cfg', None)
    Npoly        = kwargs.get('Npoly', None)
    Ttype        = kwargs.get('Ttype', None)
    logger       = kwargs.get('logger', None)

    focal_mechanism_root_name    = Config.get('Files','focal_mechanism_root_name')
    #probability_models_root_name = Config.get('Files','probability_models_root_name')
    pyptf_focal_mechanism_dir    = Config.get('pyptf','FocMech_Preproc')
    region_to_ignore             = Config.get('mix_parameters','ignore_regions').split()
    region_to_ignore             = list(map(int, region_to_ignore))
    region_list                  = [x for x in range(Npoly) if x not in region_to_ignore]
    ModelsProb_Region_files      = dict()

    ## Define Region with PS now in this DICTIONARY
    region_ps_1                  = ast.literal_eval(Config.get('regionsPerPS','1'))
    region_ps_2                  = ast.literal_eval(Config.get('regionsPerPS','2'))
    region_ps_3                  = ast.literal_eval(Config.get('regionsPerPS','3'))
    region_listPs                =  [-1 for x in range(Npoly) if x not in region_to_ignore]
    for i in region_ps_1:
        region_listPs[i-1] = 1
    for i in region_ps_2:
        region_listPs[i-1] = 2
    for i in region_ps_3:
        region_listPs[i-1] = 3
    ModelsProb_Region_files['region_listPs'] = region_listPs

    regions_without_bs_focal_mechanism = []
    regions_with_bs_focal_mechanism    = []

    for iReg in range(len(region_list)+1):
        if (sum(itype for itype in Ttype[iReg] if itype == 1) > 0):
            regions_with_bs_focal_mechanism.append(iReg)
        else:
            regions_without_bs_focal_mechanism.append(iReg)

    ModelsProb_Region_files['ModelsProb_Region_files'] = []

    #logger.info("Loading MeanProb_BS4_FocMech dictionaries               <------ ", focal_mechanism_root_name, "and",  probability_models_root_name)
    logger.info("Loading MeanProb_BS4_FocMech dict: <------ {}".format(focal_mechanism_root_name))
    for iReg in regions_with_bs_focal_mechanism:

        # define files
        filename  = focal_mechanism_root_name + '{}'.format(str(iReg+1).zfill(3)) + '.npy'
        f_FocMech = os.path.join(pyptf_focal_mechanism_dir,filename)
        #logger.info(iReg, filename)

        #filename  = probability_models_root_name + '{}'.format(str(iReg+1).zfill(3)) + '_' +  regionalization['ID'][iReg] + '.mat'
        #f_ProbMod = os.path.join(tsumaps_probability_models_dir,filename)

        ModelsProb_Region_files['ModelsProb_Region_files'].append(f_FocMech)

    for iReg in regions_without_bs_focal_mechanism:
        filename  = focal_mechanism_root_name + '{}'.format(str(iReg+1).zfill(3)) + '.npy'
        f_FocMech = os.path.join(pyptf_focal_mechanism_dir,filename)

        ModelsProb_Region_files['ModelsProb_Region_files'].append(f_FocMech)

    ModelsProb_Region_files['ModelsProb_Region_files'].sort()
    ModelsProb_Region_files['regions_with_bs_focal_mechanism']    = regions_with_bs_focal_mechanism
    ModelsProb_Region_files['regions_without_bs_focal_mechanism'] = regions_without_bs_focal_mechanism

    return ModelsProb_Region_files


def load_pois(**kwargs):

    Config       = kwargs.get('cfg', None)
    logger       = kwargs.get('logger', None)
    pois_npy     = Config.get('pyptf','POIs_npy')

    logger.info('Load dictionary with POIs list:        <------ {}'.format(pois_npy))
    # load npy file into a dictionary. The option allow_pickle=True is needed if the npy file contains np.array
    foe = np.load(pois_npy, allow_pickle=True)
    POIs = foe.item()
    empty_space = "%10s" % ('')
    logger.info('{} {} POIs found'.format(empty_space, len(POIs['name'])))
    #logger.info('{} --> {} in the Mediterranean Sea'.format(empty_space, countX(POIs['Mediterranean'], 1)))
    #logger.info('{} --> {} in the Black Sea'.format(empty_space, countX(POIs['BlackSea'], 1)))
    #logger.info('{} --> {} in the Atlantic Ocean'.format(empty_space, countX(POIs['Atlantic'], 1)))

    return POIs

def select_pois_for_fcp(**kwargs):

    fcp = kwargs.get('fcp', None)
    pois = kwargs.get('pois', None)

    fcp_pois_tmp  = []
    for i, key in enumerate(fcp):
        # get name of the pois for each fcp
        pois_list = fcp[key][0]
        fcp_pois_tmp.append(pois_list)

    fcp_pois = [item for sublist in fcp_pois_tmp for item in sublist]
    fcp_pois = set(fcp_pois)

    pois_idx = []
    for label in fcp_pois:
        if label in pois:
            pois_idx.append(pois.index(label))
   
    return pois_idx

def load_meshes(**kwargs):

    Config = kwargs.get('cfg', None)
    logger = kwargs.get('logger', None)

    #path_mesh          = Config.get('lambda','mesh_path')
    mesh_file_npy      = Config.get('Files','meshes_dictionary')

    #logger.info('Load dictionary for slab meshes:                        <------', path_mesh + os.sep + mesh_file_npy)
    #foe = np.load(path_mesh + os.sep + mesh_file_npy, allow_pickle=True)
    logger.info('Load dictionary for slab meshes:       <------ {}'.format(mesh_file_npy))
    foe = np.load(mesh_file_npy, allow_pickle=True)
    mesh = foe.item()

    return mesh


def load_PSBarInfo(**kwargs):

    Config = kwargs.get('cfg', None)

    PSBarInfo_py   = Config.get('pyptf','PSBarInfo')
    PSBarInfo_Dict = np.load(PSBarInfo_py, allow_pickle=True).item()

    return PSBarInfo_Dict


def load_Scenarios_Reg(**kwargs):

    Config  = kwargs.get('cfg', None)
    type_XS = kwargs.get('type_XS', None)
    logger  = kwargs.get('logger', None)

    # inizialize empty dict, containing the list of the ps and bs scenarios
    scenarios    = dict()

    # Load scenarios path
    scenarios_py_folder = Config.get('pyptf', 'Scenarios_py_Folder')

    # Load compressed npy
    py_scenarios = sorted(glob.glob(scenarios_py_folder + os.sep + 'ScenarioList' + type_XS + '*npz'))

    scenarios = load_scenarios(list_scenarios = py_scenarios, 
                               type_XS        = type_XS, 
                               logger         = logger)
    return scenarios


def load_Scenarios_Sel_Reg(**kwargs):

    Config      = kwargs.get('cfg', None)
    sel_regions = kwargs.get('sel_regions', None)
    type_XS     = kwargs.get('type_XS', None)
    logger      = kwargs.get('logger', None)

    # inizialize empty dict, containing the list of the ps and bs scenarios
    scenarios = dict()

    # Load scenarios path
    scenarios_py_folder = Config.get('pyptf', 'Scenarios_py_Folder')

    # Load selected compressed npy
    py_scenarios = [glob.glob(scenarios_py_folder + os.sep + 'ScenarioList' + type_XS + '_Reg' + str(sel_region).zfill(3) + '*npz')[0] for sel_region in sel_regions]

    scenarios = load_scenarios(list_scenarios = py_scenarios, 
                               type_XS        = type_XS, 
                               logger         = logger)
    return scenarios


def load_scenarios(**kwargs):

    list_scenarios = kwargs.get('list_scenarios', None)
    type_XS        = kwargs.get('type_XS', None)
    logger         = kwargs.get('logger', None)

    dic = dict()

    if (type_XS == 'PS'):

        logger.info('Load PS_scenarios in memory')
        for i in range(len(list_scenarios)):

            tmp = np.load(list_scenarios[i], allow_pickle=True)
            # a = int(list_scenarios[i].split('_')[1].replace('Reg',''))
            a = int(list_scenarios[i].split('_')[-2].replace('Reg',''))
            dic[a] = {}
            dic[a]['Parameters']       = tmp['ScenarioListPSReg'].item()['Parameters']
            dic[a]['SlipDistribution'] = tmp['ScenarioListPSReg'].item()['SlipDistribution']
            dic[a]['magPSInd']         = tmp['ScenarioListPSReg'].item()['magPSInd']
            dic[a]['modelVal']         = tmp['ScenarioListPSReg'].item()['modelVal']
            dic[a]['ID']               = tmp['ScenarioListPSReg'].item()['ID']

    elif (type_XS == 'BS'):

        logger.info('Load BS_scenarios in memory')

        for i in range(len(list_scenarios)):
            tmp = np.load(list_scenarios[i])
            # a = int(list_scenarios[i].split('_')[1].replace('Reg',''))
            a = int(list_scenarios[i].split('_')[-2].replace('Reg',''))
            dic[a] = tmp['arr_0']

    return dic


def load_intensity_thresholds(**kwargs):
    """
    READ THRESHOLDS
    """
    
    Config               = kwargs.get('cfg', None)
    intensity_thresholds = Config.get('pyptf', 'intensity_thresholds')

    ith = np.load(intensity_thresholds, allow_pickle=True)
    intensity_measure = ith.item().keys()
    thresholds = ith.item()[list(intensity_measure)[0]]

    return thresholds, intensity_measure


def load_mih_from_linear_combinations(**kwargs):

    Config = kwargs.get('cfg', None)
    seis_type = kwargs.get('seis_type', None)

    curves_py_folder = Config.get('pyptf', 'curves_gl_16')
    intensity_type   = Config.get('Settings', 'selected_intensity_measure')
    files_tag        = f'{intensity_type}_{seis_type}_curves_file_names'

    py_curves = sorted(glob.glob(curves_py_folder + os.sep + Config.get('pyptf', files_tag) + '*'))

    hazard_curves_files = dict()
    hazard_curves_files[f'{intensity_type}_{seis_type}'] = py_curves

    return hazard_curves_files


def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count
