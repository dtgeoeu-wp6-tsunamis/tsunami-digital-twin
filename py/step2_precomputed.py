#!/usr/bin/env python

# Import system modules
import os
import sys
import json
import configparser
import numpy as np
import xarray as xr
from ismember import ismember

from ptf_preload import load_mih_from_linear_combinations
from ptf_preload import load_Scenarios_Sel_Reg 


def save_mih_precomputed(**kwargs):

    n_pois = kwargs.get('n_pois', None)
    n_scenarios = kwargs.get('n_scenarios', None)
    seis_type = kwargs.get('seis_type', None)
    filename_out = kwargs.get('filename_out', None)
    workdir = kwargs.get('workdir', None)
    mih = kwargs.get('mih', None)

    # convert mih to cm
    mih = np.rint(mih * 100.)

    pois = range(n_pois)
    scenarios = range(n_scenarios)
    # outfile = os.path.join(workdir, 'Step2_' + seis_type + '_hmax_pre.nc')
    outfile = os.path.join(workdir, filename_out)

    ds = xr.Dataset(
            data_vars={'ts_max_gl': (['scenarios', 'pois'], np.transpose(mih))},
            coords={'pois': pois, 'scenarios': scenarios},
            attrs={'description': outfile, 'unit': 'cm', 'format': 'int16'})

    # encode = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 
    encode = {'zlib': True, 'complevel': 5, 'dtype': 'int16', 
              '_FillValue': False}
    encoding = {var: encode for var in ds.data_vars}
    ds.to_netcdf(outfile, format='NETCDF4', encoding=encoding)


def main(**kwargs):

    # Command args
    cfg_file = kwargs.get('cfg_file', None)                      #Configuration file
    Config   = configparser.RawConfigParser()
    Config.read(cfg_file)
    
    wd = kwargs.get('workflow_dict', None)
    workdir = wd['workdir']
    inpdir = wd['inpdir']

    # POIs
    # pois = np.load(os.path.join(workdir, 'pois.npy'), allow_pickle=True).item()['pois_coords']
    # n_pois, _ = pois.shape
    pois = np.load(os.path.join(workdir, wd['pois']), allow_pickle=True).item()['pois_index']
    n_pois = len(pois)

    # Precomputed filenames
    mih_file = load_mih_from_linear_combinations(cfg=Config)

    # BS
    # loading selected BS scenarios lists
    inpfileBS = os.path.join(workdir, wd['step1_list_BS'])
    if os.path.getsize(inpfileBS) > 0:
        print()
        par_scenarios_bs = np.loadtxt(inpfileBS)
        sel_regions_bs = np.unique(par_scenarios_bs[:,1]).astype('int')
        print('Regions BS:', sel_regions_bs)

        # matching BS scenarios between Step 1 list and the full list of scenarios in regions
        Scenarios_BS = load_Scenarios_Sel_Reg(cfg=Config, sel_regions=sel_regions_bs, type_XS='BS')
        n_bs, _ = par_scenarios_bs.shape
        mih_bs = np.full((n_pois, n_bs), np.nan)
        check_n_scenarios = 0
        for sel_region_bs in sel_regions_bs:
            reg_map, ind_bs = ismember(par_scenarios_bs[:,2:9], Scenarios_BS[sel_region_bs], 'rows')
            mih_tmp = np.load(mih_file['gl_bs'][sel_region_bs-1],mmap_mode='r')[pois,:]
            mih_bs[:,reg_map] = mih_tmp[:,ind_bs]
            del mih_tmp
            check_n_scenarios = check_n_scenarios + len(ind_bs) 
            print('Region {0}: {1} scenarios'.format(sel_region_bs, len(ind_bs)))

        if np.isnan(mih_bs).any() or check_n_scenarios != n_bs:
            sys.exit('Error in matching BS scenarios between Step 1 list and the full list of scenarios in regions')
    
        save_mih_precomputed(n_pois       = n_pois,
                             n_scenarios  = n_bs,
                             seis_type    = 'BS',
                             filename_out = wd['step2_hmax_pre_BS'],
                             workdir      = workdir,
                             mih          = mih_bs)

    # PS
    # loading selected PS scenarios lists
    inpfilePS = os.path.join(workdir, wd['step1_list_PS'])
    if os.path.getsize(inpfilePS) > 0:
        print()
        par_scenarios_ps = np.loadtxt(inpfilePS, dtype='str')
        sel_regions_ps = np.unique(par_scenarios_ps[:,1]).astype('int')
        print('Regions PS:', sel_regions_ps)

        # matching PS scenarios between Step 1 list and the full list of scenarios in regions
        Scenarios_PS = load_Scenarios_Sel_Reg(cfg=Config, sel_regions=sel_regions_ps, type_XS='PS')
        n_ps, _ = par_scenarios_ps.shape
        mih_ps = np.full((n_pois, n_ps), np.nan)
        
        for sel_region_ps in sel_regions_ps:
            reg_map, ind_ps = ismember(np.char.strip(par_scenarios_ps[:,2], '.txt'),
                                       Scenarios_PS[sel_region_ps]['ID'])
            mih_tmp = np.load(mih_file['gl_ps'][sel_region_ps-1],mmap_mode='r')[pois,:]
            mih_ps[:,reg_map] = mih_tmp[:,ind_ps]
            del mih_tmp
            print('Region {0}: {1} scenarios'.format(sel_region_ps, len(ind_ps)))

        if np.isnan(mih_ps).any():
            sys.exit('Error in matching PS scenarios between Step 1 list and the full list of scenarios in regions')

        save_mih_precomputed(n_pois       = n_pois,
                             n_scenarios  = n_ps,
                             seis_type    = 'PS',
                             filename_out = wd['step2_hmax_pre_PS'],
                             workdir      = workdir,
                             mih          = mih_ps)



if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
