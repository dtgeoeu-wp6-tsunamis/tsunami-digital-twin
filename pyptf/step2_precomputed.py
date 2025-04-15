#!/usr/bin/env python

# Import system modules
import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from ismember import ismember
# import xarray as xr
# import configparser

from pyptf.ptf_preload import load_mih_from_linear_combinations
from pyptf.ptf_preload import load_Scenarios_Sel_Reg 


def main(**kwargs):

    # Command args
    Config   = kwargs.get('cfg', None)                      #Configuration file
    wd       = kwargs.get('workflow_dict', None)
    par_scenarios_bs = kwargs.get('par_bs', None)
    par_scenarios_ps = kwargs.get('par_ps', None)
    pois_d   = kwargs.get('pois_d', None)
    dataset  = kwargs.get('dataset', None)
    logger   = kwargs.get('logger', None)

    # Config = configparser.RawConfigParser()
    # Config.read(cfg_file)
    
    workdir = wd['workdir']

    # POIs
    pois = pois_d['pois_index']
    n_pois = len(pois)
    
    # BS
    if par_scenarios_bs is None:
        inpfileBS = os.path.join(workdir, wd['step1_list_BS'])
        try: 
            if os.path.getsize(inpfileBS) > 0:
                par_scenarios_bs = np.loadtxt(inpfileBS)
            else:
                par_scenarios_bs = np.empty((0, 0))
        except:
            raise Exception(f"Error reading file: {inpfileBS}")

    n_bs, _ = par_scenarios_bs.shape

    if n_bs > 0:
        sel_regions_bs = np.unique(par_scenarios_bs[:,1]).astype('int')
        logger.info("...Selected BS Regions: {}".format(sel_regions_bs))
        #print('Regions BS:', sel_regions_bs)
        
        if dataset:
            logger.info("Get BS data pre-load in memory")

            # matching BS scenarios between Step 1 list and the full list of scenarios in regions
            Scenarios_BS = {id_reg:dataset['Scenarios_BS'][id_reg] for id_reg in dataset['Scenarios_BS'] if id_reg in sel_regions_bs}

            #n_bs, _ = par_scenarios_bs.shape
            mih_bs = np.full((n_pois, n_bs), np.nan)
            check_n_scenarios = 0
            for sel_region_bs in sel_regions_bs:
                logger.info("...Region {}".format(sel_region_bs))
                #t0 = time.time()
                if (dataset['type_df'] == 'pandas'):
                    logger.info(f"......Matching with dataframe type {dataset['type_df']}")
                    df_scenarios_bs = pd.DataFrame(par_scenarios_bs)
                    df_scenarios_bs = df_scenarios_bs[df_scenarios_bs[1]==sel_region_bs]
                    df_scenarios_bs['ID'] = df_scenarios_bs.apply(lambda x: "_".join(str(i) for i in x[2:9]),axis=1)
                    df_scenarios_bs = df_scenarios_bs.set_index('ID')
                    df_scenarios_bs = pd.merge(df_scenarios_bs, dataset['Scenarios_BS_df'][sel_region_bs], left_index=True, right_index=True)[[0,"ind_bs"]]
                    ind_bs = df_scenarios_bs['ind_bs'].values
                    list_reg_id = df_scenarios_bs[0].values.astype('int')-1
                    reg_map = np.zeros(len(par_scenarios_bs), dtype=bool)
                    reg_map[list_reg_id] = True

                elif (dataset['type_df'] == "polars"):
                    logger.info(f"......Matching with dataframe type {dataset['type_df']}")
                    df_scenarios_bs = pl.DataFrame(par_scenarios_bs)
                    df_scenarios_bs = df_scenarios_bs.filter(pl.col('column_1')==sel_region_bs)
                    df_scenarios_bs = df_scenarios_bs.with_columns(pl.concat_str([pl.col("^column_[2-8]$")],separator='_').alias("ID"))
                    df_scenarios_bs = df_scenarios_bs.join(dataset['Scenarios_BS_df'][sel_region_bs],on="ID",how='inner')[['column_0',"ind_bs"]]
                    ind_bs = df_scenarios_bs["ind_bs"].to_numpy()
                    list_reg_id = df_scenarios_bs['column_0'].cast(pl.Int32).to_numpy()-1
                    reg_map = np.zeros(len(par_scenarios_bs), dtype=bool)
                    reg_map[list_reg_id] = True

                else:
                    logger.info(f"......Matching with ismember method")
                    reg_map, ind_bs = ismember(par_scenarios_bs[:,2:9], Scenarios_BS[sel_region_bs], 'rows')

                mih_tmp = dataset['mih_values']['gl_bs'][sel_region_bs]#[pois,:]
                mih_bs[:,reg_map] = mih_tmp[:,ind_bs]
                check_n_scenarios = check_n_scenarios + len(ind_bs)
                #logger.info("...Region {0}: {1} scenarios".format(sel_region_bs, len(ind_bs)))

        else:
            # Precomputed filenames
            mih_file = load_mih_from_linear_combinations(cfg = Config,
                                                         seis_type = 'bs')

            # matching BS scenarios between Step 1 list and the full list of scenarios in regions
            Scenarios_BS = load_Scenarios_Sel_Reg(cfg=Config, sel_regions=sel_regions_bs, type_XS='BS', logger = logger)
            n_bs, _ = par_scenarios_bs.shape
            mih_bs = np.full((n_pois, n_bs), np.nan)
            check_n_scenarios = 0
            for sel_region_bs in sel_regions_bs:
                reg_map, ind_bs = ismember(par_scenarios_bs[:,2:9], Scenarios_BS[sel_region_bs], 'rows')
                mih_tmp = np.load(mih_file['gl_bs'][sel_region_bs-1],mmap_mode='r')[pois,:]
                mih_bs[:,reg_map] = mih_tmp[:,ind_bs]
                del mih_tmp
                check_n_scenarios = check_n_scenarios + len(ind_bs) 
                logger.info('Region {0}: {1} scenarios'.format(sel_region_bs, len(ind_bs)))

        if np.isnan(mih_bs).any() or check_n_scenarios != n_bs:
            raise Exception('Error in matching BS scenarios between Step 1 list and the full list of scenarios in regions')
    
    else:
        mih_bs = np.empty((0, 0))
    
    # PS
    if par_scenarios_ps is None:
        inpfilePS = os.path.join(workdir, wd['step1_list_PS'])
        try: 
            if os.path.getsize(inpfilePS) > 0:
                par_scenarios_ps = np.loadtxt(inpfilePS, dtype='str')
            else:
                par_scenarios_ps = np.empty((0, 0))
        except:
            raise Exception(f"Error reading file: {inpfilePS}")

    n_ps, _ = par_scenarios_ps.shape

    if n_ps > 0:
        logger.info('')
        sel_regions_ps = np.unique(par_scenarios_ps[:,1]).astype('int')
        logger.info('Regions PS: {}'.format(sel_regions_ps))

        if dataset:
            logger.info("Get PS data pre-load in memory")
            # matching PS scenarios between Step 1 list and the full list of scenarios in regions
            Scenarios_PS = {id_reg:dataset['Scenarios_PS'][id_reg] for id_reg in dataset['Scenarios_PS'] if id_reg in sel_regions_ps}

            #n_ps, _ = par_scenarios_ps.shape
            mih_ps = np.full((n_pois, n_ps), np.nan)

            #TODO: When using polars keep attention to the order or rows after join
            for sel_region_ps in sel_regions_ps:
                logger.info("...Region: {}".format(sel_region_ps))
                if(dataset['type_df']=='pandas'):
                    logger.info(f"......Matching with dataframe type {dataset['type_df']}")
                    df_scenarios_ps = pd.DataFrame(np.char.strip(par_scenarios_ps[:,2], '.txt'))
                    df_scenarios_ps = df_scenarios_ps.reset_index(names='id_scen')
                    df_scenarios_ps = df_scenarios_ps.set_index(0)
                    df_scenarios_ps = pd.merge(df_scenarios_ps,dataset['Scenarios_PS_df'][sel_region_ps],left_index=True, right_index=True)
                    ind_ps = df_scenarios_ps['ind_ps'].values
                    list_reg_id = df_scenarios_ps['id_scen'].values.astype('int')
                    reg_map = np.zeros(len(par_scenarios_ps), dtype=bool)
                    reg_map[list_reg_id] = True

                #TODO: L'ordine di ind_ps è diverso, capire se cambia qualcosa ai fini del risultato
                elif(dataset['type_df']=='polars'):
                    logger.info(f"......Matching with dataframe type {dataset['type_df']}")
                    df_scenarios_ps = pl.DataFrame(np.char.strip(par_scenarios_ps[:,2], '.txt'),schema=[("ID", pl.String)])
                    df_scenarios_ps = df_scenarios_ps.with_row_index(name='id_scen')
                    df_scenarios_ps = df_scenarios_ps.join(dataset['Scenarios_PS_df'][sel_region_ps],on="ID",how='inner').sort(by='id_scen')
                    ind_ps = df_scenarios_ps["ind_ps"].to_numpy()
                    list_reg_id = df_scenarios_ps['id_scen'].to_numpy()
                    reg_map = np.zeros(len(par_scenarios_ps), dtype=bool)
                    reg_map[list_reg_id] = True

                else:
                    logger.info(f"......Matching with ismember method")
                    reg_map, ind_ps = ismember(np.char.strip(par_scenarios_ps[:,2], '.txt'),Scenarios_PS[sel_region_ps]['ID'])

                mih_tmp = dataset['mih_values']['gl_ps'][sel_region_ps]#[pois,:]
                mih_ps[:,reg_map] = mih_tmp[:,ind_ps]
                #logger.info("......Select mih values for selected PS scenarios ({} over {}): time elapsed = {} sec".format(len(ind_ps), mih_tmp.shape[1], time.time()-t0))

           
        else:
            # Precomputed filenames
            mih_file = load_mih_from_linear_combinations(cfg = Config,
                                                         seis_type = 'ps')
            # matching PS scenarios between Step 1 list and the full list of scenarios in regions
            Scenarios_PS = load_Scenarios_Sel_Reg(cfg=Config, sel_regions=sel_regions_ps, type_XS='PS', logger = logger)
            n_ps, _ = par_scenarios_ps.shape
            mih_ps = np.full((n_pois, n_ps), np.nan)
            
            for sel_region_ps in sel_regions_ps:
                reg_map, ind_ps = ismember(np.char.strip(par_scenarios_ps[:,2], '.txt'),
                                        Scenarios_PS[sel_region_ps]['ID'])
                mih_tmp = np.load(mih_file['gl_ps'][sel_region_ps-1],mmap_mode='r')[pois,:]
                mih_ps[:,reg_map] = mih_tmp[:,ind_ps]
                del mih_tmp
                logger.info('Region {0}: {1} scenarios'.format(sel_region_ps, len(ind_ps)))

        if np.isnan(mih_ps).any():
            raise Exception('Error in matching PS scenarios between Step 1 list and the full list of scenarios in regions')

    else:
        mih_ps = np.empty((0, 0))

    mih_sbs = np.empty((0, 0))
    # transposing and arounding to be consistent with the file format
    return np.around(np.transpose(mih_bs), decimals=2), np.around(np.transpose(mih_ps), decimals=2), mih_sbs

if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
