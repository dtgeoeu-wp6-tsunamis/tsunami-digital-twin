#!/usr/bin/env python

# Import system modules
import os
import ast
import sys
import glob
import numpy as np
#import re
#import h5py
#from pymatreader         import read_mat
#from mat4py              import loadmat
#from collections         import defaultdict
#from shapely             import geometry
#import datetime


def load_Model_Weights(**kwargs):

    Config             = kwargs.get('cfg', None)
    ps_type            = kwargs.get('ps_type', None)
    
    weight_npy         = Config.get('pyptf','ModelWeight')
    empty_space        = "%64s" % ('')

    print('Loading model weights dictionary                        <------ ', weight_npy)
    foe = np.load(weight_npy, allow_pickle=True)
    modelweights = foe.item()
    for k in modelweights.keys():
            print(empty_space, k)

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

    Config             = kwargs.get('cfg', None)

#    project_name       = Config.get('Project','Name')
    discretization_npy = Config.get('pyptf','Discretization_npy')
    empty_space        = "%64s" % ('')

    print('Loading discretization dictionary                       <------ ', discretization_npy)
    foe = np.load(discretization_npy, allow_pickle=True)
    discretizations = foe.item()
    for i in discretizations.keys():
        print(empty_space, i)

#    # clean unused keys
#    # ['BS-4_FocalMechanism', 'ID'] only used in ptf_short_term.py, get_hypocentral_prob_for_bs
#    discretizations = clean_dictionary(dict = discretizations, keys=[['BS-6_Slip'], ['BS-5_Area'], ['PS-2_PositionArea', 'Region'],
#                                                                   ['PS-2_PositionArea', 'ID'], ['PS-2_PositionArea', 'Val'],
#                                                                   ['BS-3_Depth', 'ID']])

    return discretizations


def load_regionalization(**kwargs):

    Config              = kwargs.get('cfg', None)
    regionalization_npy = Config.get('pyptf','Regionalization_npy')

    print('Load dictionary for regionalization:                    <------ ', regionalization_npy)
    # load npy file into a dictionary. The option allow_pickle=True is needed if the npy file contains np.array
    foe = np.load(regionalization_npy, allow_pickle=True)
    regionalization = foe.item()
    print("%64s Regions found: %3d" % ('', regionalization['Npoly']))

#    # clean unused keys
#    regionalization = clean_dictionary(dict = regionalization, keys=[['Tnames'], ['Tleng'], ['Tpoint'], ['ind']])

    return regionalization


def load_region_files(**kwargs):

    Config       = kwargs.get('cfg', None)
    Npoly        = kwargs.get('Npoly', None)
    Ttype        = kwargs.get('Ttype', None)

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

    #print("Loading MeanProb_BS4_FocMech dictionaries               <------ ", focal_mechanism_root_name, "and",  probability_models_root_name)
    print("Loading MeanProb_BS4_FocMech dictionaries               <------ ", focal_mechanism_root_name)
    for iReg in regions_with_bs_focal_mechanism:

        # define files
        filename  = focal_mechanism_root_name + '{}'.format(str(iReg+1).zfill(3)) + '.npy'
        f_FocMech = os.path.join(pyptf_focal_mechanism_dir,filename)
        #print(iReg, filename)

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
    pois_npy     = Config.get('pyptf','POIs_npy')

    print('Load dictionary with POIs list:                         <------ ', pois_npy)
    # load npy file into a dictionary. The option allow_pickle=True is needed if the npy file contains np.array
    foe = np.load(pois_npy, allow_pickle=True)
    POIs = foe.item()
    empty_space = "%64s" % ('')
    print(empty_space, '{} POIs found'.format(len(POIs['name'])))
    print(empty_space, '--> {} in the Mediterranean Sea'.format(countX(POIs['Mediterranean'], 1)))
    print(empty_space, '--> {} in the Black Sea'.format(countX(POIs['BlackSea'], 1)))
    print(empty_space, '--> {} in the Atlantic Ocean'.format(countX(POIs['Atlantic'], 1)))

    return POIs

def select_pois_for_fcp(**kwargs):

    Config = kwargs.get('cfg', None)
    pois   = kwargs.get('pois', None)

    fcp_lib = Config.get('Files', 'pois_to_fcp')
    fcp     = np.load(fcp_lib, allow_pickle=True).item()

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

    #path_mesh          = Config.get('lambda','mesh_path')
    mesh_file_npy      = Config.get('Files','meshes_dictionary')

    #print('Load dictionary for slab meshes:                        <------', path_mesh + os.sep + mesh_file_npy)
    #foe = np.load(path_mesh + os.sep + mesh_file_npy, allow_pickle=True)
    print('Load dictionary for slab meshes:                        <------', mesh_file_npy)
    foe = np.load(mesh_file_npy, allow_pickle=True)
    mesh = foe.item()

    return mesh


def load_PSBarInfo(**kwargs):

    Config = kwargs.get('cfg', None)

    PSBarInfo_py   = Config.get('pyptf','PSBarInfo')
    PSBarInfo_Dict = np.load(PSBarInfo_py, allow_pickle=True).item()

    return PSBarInfo_Dict


def load_Scenarios_Reg(**kwargs):

    Config      = kwargs.get('cfg', None)
    type_XS     = kwargs.get('type_XS', None)

    # inizialize empty dict, containing the list of the ps and bs scenarios
    scenarios    = dict()

    # Load scenarios path
    scenarios_py_folder  =  Config.get('pyptf',  'Scenarios_py_Folder')

    # Load compressed npy
    py_scenarios = sorted(glob.glob(scenarios_py_folder + os.sep + 'ScenarioList' + type_XS + '*npz'))

    scenarios = load_scenarios(list_scenarios=py_scenarios, type_XS=type_XS, cfg=Config)
    return scenarios


def load_Scenarios_Sel_Reg(**kwargs):

    Config      = kwargs.get('cfg', None)
    sel_regions = kwargs.get('sel_regions', None)
    type_XS     = kwargs.get('type_XS', None)

    # inizialize empty dict, containing the list of the ps and bs scenarios
    scenarios    = dict()

    # Load scenarios path
    scenarios_py_folder  =  Config.get('pyptf', 'Scenarios_py_Folder')

    # Load selected compressed npy
    py_scenarios = [glob.glob(scenarios_py_folder + os.sep + 'ScenarioList' + type_XS + '_Reg' + str(sel_region).zfill(3) + '*npz')[0] for sel_region in sel_regions]

    scenarios = load_scenarios(list_scenarios=py_scenarios, type_XS=type_XS, cfg=Config)
    return scenarios


def load_scenarios(**kwargs):

    list_scenarios  = kwargs.get('list_scenarios', None)
    type_XS         = kwargs.get('type_XS', None)
    Config          = kwargs.get('cfg', None)

    dic = dict()

    if (type_XS == 'PS'):

        print('Load PS_scenarios in memory' )
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

        print('Load BS_scenarios in memory' )

        for i in range(len(list_scenarios)):
            tmp = np.load(list_scenarios[i])
            # print("....", list_scenarios[i])
            # if(all_dict == "Yes"):
            #     tmp = np.load(list_scenarios[i])
            # else:
            #     tmp = np.load(list_scenarios[i])

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


def select_pois_and_regions(**kwargs):

    Config          = kwargs.get('cfg', None)
    wd              = kwargs.get('workflow_dict', None)
    regions         = kwargs.get('regions', None)
    ignore_regions  = kwargs.get('ignore_regions', None)
    POIs            = kwargs.get('pois_dictionary', None)
    regionalization = kwargs.get('regionalization_dictionary', None)


    # if(selpois == '-1' or selpois == 'mediterranean'):
    #   SelectedPOIs = ['mediterranean']
    if wd['domain'] == 'med-tsumaps':
       SelectedPOIs = 'Mediterranean'
    elif wd['domain'] == 'atlantic':
       SelectedPOIs = 'Atlantic'
    elif wd['domain'] == 'black_sea':
       SelectedPOIs = 'BlackSea'
    elif wd['domain'] == 'pacific':
       SelectedPOIs = 'Pacific'
    else:
       sys.exit('domain not defined')

    # tmp = []
    # if(selpois != '-1' and selpois != 'mediterranean'):   #in case of selection of a subset of pois (Mediterranean-4: 1 POI every 4; med09159 med09174)
    #    ele_args = selpois.split(' ')
    #    for i in range(len(ele_args)):
    #        if(RepresentsInt(ele_args[i]) == True):
    #            tmp.append(int(ele_args[i]))
    #        else:
    #            tmp.append(ele_args[i])
    #    SelectedPOIs = tmp
    # if (len(SelectedPOIs) == 0):
    #   SelectedPOIs = POIs['name']
    # elif (len(SelectedPOIs) == 1 and  SelectedPOIs[0].startswith('mediterranean')):
    #   tmpmed = np.array(POIs['Mediterranean'])
    #   tmp = np.nonzero(tmpmed)[0]
    #   xpoi = SelectedPOIs[0].split('-')
    #   if (len(xpoi) == 1):
    #      SelectedPOIs = [POIs['name'][j] for j in tmp] #All
    #      Selected_lon = [POIs['lon'][j] for j in tmp] #All
    #      Selected_lat = [POIs['lat'][j] for j in tmp] #All
    #   else:
    #      step = int(xpoi[1])
    #      SelectedPOIs = [POIs['name'][j] for j in tmp[::step]] #1 every step
    #      Selected_lon = [POIs['lon'][j] for j in tmp[::step]] #1 every step
    #      Selected_lat = [POIs['lat'][j] for j in tmp[::step]] #1 every step

    tmppois = np.array(POIs[SelectedPOIs])
    tmp = np.nonzero(tmppois)[0]
    SelectedPOIs = [POIs['name'][j] for j in tmp] #All
    Selected_lon = [POIs['lon'][j] for j in tmp] #All
    Selected_lat = [POIs['lat'][j] for j in tmp] #All

    tmp = []
    reg_args = regions.split(' ')
    if(regions == '-1' or regions == 'all'):
        SelectedRegions = [-1]
    elif(len(reg_args) >= 1):
        for i in range(len(reg_args)):
            tmp.append(int(reg_args[i]))
        SelectedRegions = tmp

    if (ignore_regions != None):
        noreg_args = ignore_regions.split(' ')
        noreg_args = [ int(x) for x in noreg_args ]
        IgnoreRegions = noreg_args
    else:
        IgnoreRegions = []

    if (len(SelectedRegions) == 0):
      SelectedRegions = range(regionalization['Npoly'])
      #IgnoreRegions = [42]  #!!REGION 43 NOT AVAILABLE!!#
      IgnoreRegions = Config.get('mix_parameters','ignore_regions').split()
      IgnoreRegions = list(map(int, IgnoreRegions))
    else:
      IgnoreRegions = []

    POIs['selected_pois']    = SelectedPOIs
    POIs['selected_coords']  = np.column_stack((Selected_lon,Selected_lat))
    POIs['selected_regions'] = SelectedRegions
    POIs['ignore_regions']   = IgnoreRegions
    POIs['nr_selected_pois'] = len(SelectedPOIs)

    return POIs


def load_mih_from_linear_combinations(**kwargs):

    Config = kwargs.get('cfg', None)
    #in_memory = kwargs.get('in_memory', False)

    scenarios = dict()

    curves_py_folder = Config.get('pyptf', 'curves_gl_16')
    intensity_type   = Config.get('Settings', 'selected_intensity_measure')
    bs_files_tag     = intensity_type + '_bs_curves_file_names'
    ps_files_tag     = intensity_type + '_ps_curves_file_names'

    py_bs_curves = sorted(glob.glob(curves_py_folder + os.sep + Config.get('pyptf', bs_files_tag) + '*'))
    py_ps_curves = sorted(glob.glob(curves_py_folder + os.sep + Config.get('pyptf', ps_files_tag) + '*'))

    # # Maybe useless
    # py_bs_curves = reallocate_curves(curve_files=py_bs_curves, cfg=Config, name=Config.get('pyptf', bs_files_tag))
    # py_ps_curves = reallocate_curves(curve_files=py_ps_curves, cfg=Config, name=Config.get('pyptf', ps_files_tag))
  
    hazard_curves_files = dict()
    hazard_curves_files[intensity_type + '_ps'] = py_ps_curves
    hazard_curves_files[intensity_type + '_bs'] = py_bs_curves

    return hazard_curves_files


def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count


# unused functions
#
# def reallocate_curves(**kwargs):
# 
#     Config    = kwargs.get('cfg', None)
#     c_files   = kwargs.get('curve_files', None)
#     name      = kwargs.get('name', None)
# 
#     curves_py_folder  = Config.get('pyptf',  'curves_gl_16')
# 
#     list_out = []
#     for i in range(0,int(Config.get('ScenariosList', 'nr_regions'))):
#         d = "%03d" % (i+1)
#         def_name = curves_py_folder + os.sep + name + d + '-empty.npy'
#         list_out.append(def_name)
# 
#     for i in range(len(c_files)):
#         ref_nr = int(c_files[i].split(name)[-1][0:3])
#         list_out[ref_nr-1] = c_files[i]
# 
#     return list_out
#
#
#def load_Conditional_HCurves_table(**kwargs):
#
#    Config             = kwargs.get('cfg', None)
#    lookup_tables_npy  = Config.get('pyptf','LookUpTable_npy')
#    empty_space        = "%64s" % ('')
#
#    print('Loading lookup table for conditional hazard dictionary  <------ ', lookup_tables_npy)
#    # load npy file into a dictionary. The option allow_pickle=True is needed if the npy file contains np.array
#    foe = np.load(lookup_tables_npy, allow_pickle=True)
#    LookupTableConditionalHazardCurves = foe.item()
#    for i in LookupTableConditionalHazardCurves.keys():
#        print(empty_space, i)
#
#    return LookupTableConditionalHazardCurves
#
#def clean_dictionary(**kwargs):
#
#    Dict = kwargs.get('dict', None)
#    Keys = kwargs.get('keys', None)
#
#    for i in range(len(Keys)):
#
#        if len(Keys[i]) == 1:
#            Dict.pop(Keys[i][0])
#
#        if len(Keys[i]) == 2:
#            Dict[Keys[i][0]].pop(Keys[i][1])
#
#
#    return Dict
#
#def check_if_path_exists(**kwargs):
#
#    path      = kwargs.get('path', None)
#    action    = bool(kwargs.get('path', False))
#
#    if os.path.isdir(path):
#        return True
#
#    if os.path.isdir(path) == False:
#        if action == True:
#            os.mkdir(path)
#        else:
#            return False
#
#    return True
#
#def RepresentsInt(s):
#    try:
#       int(s)
#       return True
#    except ValueError:
#       return False
#
#def load_BS_Scenarios_Reg(**kwargs):
#
#    Config    = kwargs.get('cfg', None)
#
#    # inizialize empty dict, containing the list of the ps and bs scenarios
#    scenarios    = dict()
#
#    # Load scenarios path
#    scenarios_py_folder  =  Config.get('pyptf',  'Scenarios_py_Folder')
#
#    # Load compressed npy
#    py_bs_scenarios = sorted(glob.glob(scenarios_py_folder + os.sep + 'ScenarioListBS*npz'))
#
#    bs_scenarios = load_scenarios(list_scenarios=py_bs_scenarios, type_XS='BS', cfg=Config)
#
#    return bs_scenarios
#
#
#def load_PS_Scenarios_Reg(**kwargs):
#
#    Config    = kwargs.get('cfg', None)
#
#    # inizialize empty dict, containing the list of the ps and bs scenarios
#    scenarios    = dict()
#
#    # Load scenarios path
#    scenarios_py_folder  =  Config.get('pyptf',  'Scenarios_py_Folder')
#
#    # Load compressed npy
#    py_ps_scenarios = sorted(glob.glob(scenarios_py_folder + os.sep + 'ScenarioListPS*npz'))
#
#    ps_scenarios = load_scenarios(list_scenarios=py_ps_scenarios, type_XS='PS', cfg=Config)
#
#    return ps_scenarios


