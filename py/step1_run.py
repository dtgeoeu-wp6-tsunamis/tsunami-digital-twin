#!/usr/bin/env python

import os
import sys
import json
import configparser
import numpy as np

# Import functions from pyPTF modules
from ptf_preload             import load_PSBarInfo
from ptf_preload             import load_Scenarios_Reg
from ptf_preload             import load_meshes
from ptf_preload             import load_pois
from ptf_preload             import load_regionalization
from ptf_preload             import load_region_files
from ptf_preload             import load_discretization
from ptf_preload             import load_Model_Weights
from ptf_preload             import select_pois_and_regions
from ptf_define_ensemble     import define_ensemble
from ptf_preload             import select_pois_for_fcp


def main(**kwargs):

    cfg_file      = kwargs.get('cfg_file', None)                # Configuration file
    workflow_dict = kwargs.get('workflow_dict', None)           # workflow dictionary
    event_dict    = kwargs.get('event_dict', None)              # event dictionary
    args          = kwargs.get('args', None)
    
    regions = args.regions                 # regions to load [1-100]. -1=all. May select one ore more. Default=-1
    ignore_regions = args.ignore_regions   # regions to ignore [1-100]
    geocode_area = args.geocode_area       # Get the area name from geocode search instead of json file. Can take 1.5 seconds. [No]/Yes
    mag_sigma_fix = args.mag_sigma_fix     # Fix the magnitude event sigma. If Yes take the mag_sigma_val value. [No]/yes
    mag_sigma_val = args.mag_sigma_val     # Assumed magnitude event sigma. Needs --mag_sigma_fix=Yes. Default=0.15
    ps_type = args.ps_type                 # PS probability type: 1,2. Default 1

    # Initialize and load configuration file
    Config = configparser.RawConfigParser()
    Config.read(cfg_file)
    # Config = update_cfg(cfg=Config, args=args)

    # Load info from .npy files (originally converted from sptha output)
    PSBarInfo                      = load_PSBarInfo(cfg = Config)
    Scenarios_PS                   = load_Scenarios_Reg(cfg = Config,type_XS = 'PS')
    Mesh                           = load_meshes(cfg = Config)
    Regionalization                = load_regionalization(cfg = Config)
    POIs                           = load_pois(cfg = Config)
    POIs                           = select_pois_and_regions(cfg                        = Config,
                                                             workflow_dict              = workflow_dict,
                                                             regions                    = regions,
                                                             ignore_regions             = ignore_regions,
                                                             pois_dictionary            = POIs,
                                                             regionalization_dictionary = Regionalization)

    Discretization                 = load_discretization(cfg = Config)
    Model_Weights                  = load_Model_Weights(cfg     = Config,
                                                        ps_type = ps_type)
    Region_files                   = load_region_files(cfg   = Config,
                                                       Npoly = Regionalization['Npoly'],
                                                       Ttype = Regionalization['Ttypes'])

    LongTermInfo                    = {}
    LongTermInfo['Regionalization'] = Regionalization
    LongTermInfo['Discretizations'] = Discretization
    LongTermInfo['Model_Weights']   = Model_Weights
    LongTermInfo['vecID']           = 10.**np.array([8, 5, 2, 0, -2, -4, -6])

    # save the selected POIs as npy in the working folder
    #outfile = os.path.join(workflow_dict['workdir'], 'pois.npy')
    outfile = os.path.join(workflow_dict['workdir'], workflow_dict['pois'])
    pois_d = dict()
    pois_d['pois_coords'] = POIs['selected_coords']
    pois_d['pois_labels'] = POIs['selected_pois']
    
    if workflow_dict['TEWmode']:
        pois_idx_for_fcp = select_pois_for_fcp(cfg = Config, pois = POIs['selected_pois'])
        pois_d['pois_index'] = sorted(pois_idx_for_fcp)
    else:
        pois_d['pois_index'] = list(range(len(POIs['selected_pois'])))
    np.save(outfile, pois_d, allow_pickle=True)

# define the ensemble
    #ptf_out = define_ensemble(workflow_dict     = workflow_dict,
    define_ensemble(workflow_dict     = workflow_dict,
                    Scenarios_PS      = Scenarios_PS,
                    LongTermInfo      = LongTermInfo,
                    POIs              = POIs,
                    PSBarInfo         = PSBarInfo,
                    Mesh              = Mesh,
                    Region_files      = Region_files,
                    cfg               = Config,
                    event_data        = event_dict,
                    Regionalization   = Regionalization)

    np.save(os.path.join(workflow_dict['workdir'], workflow_dict['workflow_dictionary']), workflow_dict, allow_pickle=True)


if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
