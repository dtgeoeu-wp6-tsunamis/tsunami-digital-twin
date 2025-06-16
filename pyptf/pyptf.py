import configparser
import numpy as np
import time
import os
import logging
import shutil
import sys

from scipy.stats import norm
import pandas as pd
import polars as pl
import json

import pyptf.ptf_save_results
from pyptf import ptf_mix_utilities
from pyptf import ptf_preload
from pyptf import ptf_load_event
from pyptf import run_steps
from pyptf.pyptf_exceptions  import PyPTFException

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class PyPTF:


    def __init__(self, cfg, input_workflow, hazard_mode, ptf_version, user_gridname, 
                 mag_sigma_val, sigma, logn_sigma, in_memory, type_df, percentiles, save=True, logger=None):

        starttime = time.time()

        # Determine whether we save the results in working directory (save=True) or we return them as output of method run (save=False)
        self.save = save

        self.logger = logger
        if not self.logger:
            self.logger = self._create_logger()

        self.logger.info('Initializing pyPTF class')
        ### The class Namespace simulate the behaviour of parameters passed to command line
        self.args = Namespace(
            cfg = cfg,
            input_workflow = input_workflow,
            hazard_mode = hazard_mode,
            ptf_version = ptf_version,
            user_gridname = user_gridname,
            mag_sigma_val = mag_sigma_val,
            type_df = type_df,
            ps_type = 1,
            percentiles = percentiles,
            sigma = sigma,
            logn_sigma = logn_sigma,
            regions = '-1',
            ignore_regions = None,
            geocode_area = 'No',
            mag_sigma_fix = 'No'#,
            # mag_sigma_val = 0.15
        )

        ######################################
        #region: Load configuration files

        self.logger.info('...Create Workflow Dict: start')

        ### parse input config file
        self.config = configparser.RawConfigParser()
        self.config.read(cfg)
        #endregion
        ######################################

        ### parse and enrich input workflow dict
        self.workflow_dict = ptf_mix_utilities.create_workflow_dict(args   = self.args,
                                                                    Config = self.config,
                                                                    logger = self.logger)

        self.logger.info('...Create Workflow Dict: finish')

        # loading intensity thresholds (mih)
        self.thresholds, self.intensity_measure = ptf_preload.load_intensity_thresholds(cfg = self.config)
        # loading fcp
        self.fcp = np.load(self.workflow_dict['pois_to_fcp'], allow_pickle=True).item()
        with open (self.config.get('Files','fcp_json'), 'r') as f:
            self.fcp_dict = {x['name']: x['id'] for x in json.load(f)['data']}

        #TODO this block of code loading and setting POIs can be improved (RobT)
        POIs = ptf_preload.load_pois(cfg    = self.config, 
                                     logger = self.logger)

        pois_d = dict()
        if self.workflow_dict['ptf_version'] == 'neam':
            POIs = ptf_mix_utilities.select_pois_cat_area(pois_dictionary = POIs)
        
            pois_d['pois_coords'] = POIs['selected_coords']
            pois_d['pois_depth'] = POIs['selected_depth']
            pois_d['pois_labels'] = np.array(POIs['selected_pois'])

            if self.workflow_dict['TEWmode']:
                pois_idx_for_fcp = ptf_preload.select_pois_for_fcp(fcp  = self.fcp, 
                                                                   pois = POIs['selected_pois'])
                pois_index = sorted(pois_idx_for_fcp)
                pois_d['pois_index'] = pois_index
                pois_d['pois_coords'] = pois_d['pois_coords'][pois_index]
                pois_d['pois_depth'] = pois_d['pois_depth'][pois_index]
                pois_d['pois_labels'] = pois_d['pois_labels'][pois_index]
                
            else:
                pois_d['pois_index'] = list(range(len(POIs['selected_pois'])))
        else:
            pois_d['pois_coords'] = np.column_stack((POIs['lon'], POIs['lat']))
            pois_d['pois_depth'] = np.array(POIs['dep'])
            pois_d['pois_labels'] = POIs['name']
            pois_d['pois_index'] = list(range(len(POIs['name'])))
        
        pois_d['pois_depth'] = - pois_d['pois_depth'] # this is because the depth is needed positive in water for simulations
        # print(type(pois_d['pois_depth']), np.mean(pois_d['pois_depth']), len(pois_d['pois_depth']))
        self.pois_d = pois_d

        ### If the parameter in_memory is set to True, we pre-load in memory all data used in ptf steps
        # NOTE in memory is convenient if we work with forecast point (TEWmode=True)
        
        if in_memory:

            self.logger.info('...Load in Memory Input Data: Start')

            ##################################
            #region dati step1
            PSBarInfo = ptf_preload.load_PSBarInfo(cfg = self.config)
            Scenarios_PS = ptf_preload.load_Scenarios_Reg(cfg     = self.config, 
                                                          type_XS = 'PS', 
                                                          logger  = self.logger)
            Mesh = ptf_preload.load_meshes(cfg    = self.config, 
                                           logger = self.logger)
            Regionalization = ptf_preload.load_regionalization(cfg    = self.config, 
                                                               logger = self.logger)
            Discretization = ptf_preload.load_discretization(cfg    = self.config, 
                                                             logger = self.logger)
            Model_Weights = ptf_preload.load_Model_Weights(cfg     = self.config,
                                                           ps_type = self.args.ps_type, 
                                                           logger  = self.logger)
            Region_files = ptf_preload.load_region_files(cfg    = self.config,
                                                         Npoly  = Regionalization['Npoly'],
                                                         Ttype  = Regionalization['Ttypes'], 
                                                         logger = self.logger)

            LongTermInfo = {}
            LongTermInfo['Regionalization'] = Regionalization
            LongTermInfo['Discretizations'] = Discretization
            LongTermInfo['Model_Weights'] = Model_Weights
            LongTermInfo['vecID'] = 10. ** np.array([8, 5, 2, 0, -2, -4, -6])
            #endregion
            ##################################

            ##################################
            #region dati step2

            Scenarios_BS = ptf_preload.load_Scenarios_Reg(cfg     = self.config, 
                                                          type_XS = 'BS', 
                                                          logger  = self.logger)

            #TODO: The conversion pf PS and BS can be done at file level 'una tantum'. We can also associate the scenario id directly to mihs files. However to do so, one should convert all files in tabular version, this will also improve performances of step.3
            Scenarios_PS_df = self._convert_Scenarios_PS_df(scenarios_dict = Scenarios_PS,
                                                            type_df        = self.args.type_df)
            Scenarios_BS_df = self._convert_Scenarios_BS_df(scenarios_dict = Scenarios_BS,
                                                            type_df        = self.args.type_df)

            ### convert Scenarios_BS and Scenarios_PS in a dictionary of dataframe
            mih_values = self._load_mihs_inmemory()

            #endregion
            ##################################

            self.dataset = {
                'PSBarInfo': PSBarInfo,
                'Scenarios_PS': Scenarios_PS,
                'Mesh': Mesh,
                'POIs': POIs,
                'Region_files': Region_files,
                'LongTermInfo': LongTermInfo,
                'Scenarios_BS': Scenarios_BS,
                'Scenarios_PS_df': Scenarios_PS_df,
                'Scenarios_BS_df': Scenarios_BS_df,
                'type_df': self.args.type_df,
                'mih_values': mih_values
            }
            self.logger.info('...Load in Memory Input Data: Finished')
        else:
            self.dataset = None
        
        endtime = time.time()
        self.logger.info(f"Initialized pyPTF class: total time {endtime - starttime} sec")



    def _create_logger(self):

        # create logger with current application

        #log_name = Path(__file__).stem
        logger = logging.getLogger(self.__class__.__name__)
        numeric_level = getattr(logging, 'DEBUG', None)
        logger.setLevel(numeric_level)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                    datefmt='%Y-%m-%dT%H:%M:%S')

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.info("Logger has been initialized.")
        return logger


    def _convert_Scenarios_PS_df(self, scenarios_dict,type_df='pandas'):
        """
        Convert Dictionary Scenario PS from numpy array to pandas or polars dataframe
        """

        scenarios_dict_pd = {}
        if(type_df in ('pandas','polars')):
            for ireg in scenarios_dict:
                scenarios_reg = scenarios_dict[ireg]
                #df_scenarios_reg = pd.DataFrame(scenarios_reg['ID'])
                #df_scenarios_reg = df_scenarios_reg.reset_index(names='ind_ps')
                #df_scenarios_reg = df_scenarios_reg.set_index(0)
                df_scenarios_reg = pl.DataFrame([item[0] for item in scenarios_reg['ID']]).rename({"column_0": "ID"})
                df_scenarios_reg = df_scenarios_reg.with_row_index(name='ind_ps')
                if(type_df=='pandas'):
                    df_scenarios_reg = df_scenarios_reg.to_pandas().set_index('ID')
                self.logger.info(f"...Converted Scenarios Region {ireg} to {type_df}: {len(df_scenarios_reg)} scenarios")
                scenarios_dict_pd[ireg] = df_scenarios_reg
        else:
            self.logger.info(f"Type of dataframe = {type_df}, nothing to convert")
        return scenarios_dict_pd


    def _convert_Scenarios_BS_df(self, scenarios_dict, type_df='pandas'):
        """
        Convert Dictionary Scenario BS from numpy array to pandas or polars dataframe
        """

        scenarios_dict_pd = {}
        if(type_df in ('pandas','polars')):
            for ireg in scenarios_dict:
                scenarios_reg = scenarios_dict[ireg]
                #df_scenarios_reg = pd.DataFrame(scenarios_reg)
                #df_scenarios_reg['ID'] = df_scenarios_reg.apply(lambda x: "_".join(str(i) for i in x),axis=1)
                #df_scenarios_reg = df_scenarios_reg.reset_index(names='ind_bs')
                #df_scenarios_reg = df_scenarios_reg.set_index('ID')
                #self.logger.warning("We selected a limited number of regions for test purpose")
                if(len(scenarios_reg)>0):
                    df_scenarios_reg = pl.DataFrame(scenarios_reg)
                    df_scenarios_reg = df_scenarios_reg.with_columns(pl.concat_str([pl.col("*")],separator='_').alias("ID"))
                    df_scenarios_reg = df_scenarios_reg.with_row_index(name='ind_bs')
                    if(type_df=='pandas'):
                        df_scenarios_reg = df_scenarios_reg.to_pandas().set_index('ID')
                    self.logger.info(f"...Converted Scenarios Region {ireg} to {type_df}: {len(df_scenarios_reg)} scenarios")
                    scenarios_dict_pd[ireg] = df_scenarios_reg
                else:
                    self.logger.warning(f"No scenarios for region {ireg}")
                    scenarios_dict_pd[ireg] = []
        else:
            self.logger.info(f"Type of dataframe = {type_df}, nothing to convert")
        return scenarios_dict_pd


    #TODO: consider the possibility to save mih_tmp[pois,:], it is relevant in case we perform steps for fcp and not all pois
    def _load_mihs_inmemory(self):

        mih_files_bs = ptf_preload.load_mih_from_linear_combinations(cfg = self.config,
                                                                     seis_type = 'bs')
        mih_files_ps = ptf_preload.load_mih_from_linear_combinations(cfg = self.config,
                                                                     seis_type = 'ps')
        mih_files = {**mih_files_bs, **mih_files_ps}
        
        mih_values = {'gl_ps':{},
                      'gl_bs':{}}
        pois_idx = self.pois_d['pois_index']
        for gl_type in mih_values.keys():
            for i in range(len(mih_files[gl_type])):
                if('empty' in mih_files[gl_type][i]):
                    # numpy cannot load empty files
                    self.logger.info("......Region file {}: empty".format(mih_files[gl_type][i]))
                    mih_tmp = None                    
                else:
                    # this if statement is needed for performances reason only
                    if self.workflow_dict['TEWmode']:
                        mih_tmp = np.load(mih_files[gl_type][i])[pois_idx,:]
                    else:
                        mih_tmp = np.load(mih_files[gl_type][i])    
                    self.logger.info("......Region file {}: loaded".format(mih_files[gl_type][i]))
                mih_values[gl_type][i+1] = mih_tmp
        return mih_values


    ### method to run ptf steps on a given input event_dict from file
    def run_from_file(self, event_file, event_format='json'):

        if not os.path.exists(event_file):
            raise Exception(f'Event file {event_file} not found. Exit')
        
        jsn_object = ptf_load_event.read_event_parameters(event_file   = event_file,
                                                          event_format = event_format)

        self.run(jsn_object)


    ### method to run ptf steps on a given input event_dict
    def run(self, event_dict, run_sigma=None, run_hazard_mode=None):

        ### we create a local copy of workflow dict and add all information related to the input event
        workflow_dict = self.workflow_dict.copy()

        workflow_dict['eventID'] = str(event_dict['features'][0]['properties']['eventId'])
        workflow_dict['originID'] = str(event_dict['features'][0]['properties']['originId'])
        workflow_dict['version'] = str(event_dict['features'][0]['properties']['version'])
        workflow_dict['uniqueID'] = workflow_dict['eventID'] + '_' + workflow_dict['originID'] + '_' + workflow_dict['version']

        # create unique file names for each event
        ptf_mix_utilities.create_output_names(workflow_dict = workflow_dict)

        self.logger.info(f"Create workflow dict for event: {workflow_dict['eventID']}")

        if (run_sigma):
            self.logger.warning(f"Modified sigma value for the current run from {self.workflow_dict['sigma']} to {run_sigma}")
            workflow_dict['sigma'] = run_sigma
            workflow_dict['sigma_inn'] = workflow_dict['sigma']
            workflow_dict['sigma_out'] = workflow_dict['sigma'] + 0.5
            workflow_dict['negligible_prob'] = 2*norm.cdf(-1. * workflow_dict['sigma'])

        if (run_hazard_mode):
            self.logger.warning(f"Modified hazard_mode for the current run from {self.workflow_dict['hazard_mode']} to {run_hazard_mode}")
            workflow_dict['hazard_mode'] = run_hazard_mode

        #workflow_dict['workdir'] = os.path.join(workflow_dict['workpath'], workflow_dict['eventID'])
        workflow_dict['workdir'] = os.path.join(workflow_dict['workpath'], 
                                   workflow_dict['eventID'] + f"__s{workflow_dict['sigma']}" + f"__{workflow_dict['hazard_mode']}" + f"__{workflow_dict['ptf_version']}")


        if workflow_dict['sampling_mode'] != 'None':
            workflow_dict['workdir'] = os.path.join(workflow_dict['workdir'],
                                       'sampling_' + workflow_dict['sampling_mode'] + '_' + workflow_dict['sampling_type'])

        if not os.path.exists(workflow_dict['workdir']):
            os.makedirs(workflow_dict['workdir'])
            self.logger.info("Create Work Directory: {}".format(workflow_dict['workdir']))
        else:
            self.logger.info('This event has been already processed; '
                             'previous results will be overwritten.')

        # init of ptfResults class
        ptf_results = pyptf.ptf_save_results.ptfResult(event_dict = event_dict, 
                                                       workflow_dict = workflow_dict,
                                                       pois_d = self.pois_d,
                                                       thresholds = self.thresholds,
                                                       logger = self.logger)
        try:
            ### we parse input event dict
            event_parameters = ptf_load_event.create_event_dict(cfg           = self.config,
                                                                workflow_dict = workflow_dict,
                                                                geocode_area  = self.args.geocode_area,
                                                                jsn_object    = event_dict,
                                                                logger        = self.logger)
            ### we print the final event dict
            ptf_load_event.print_event_parameters(event_dict = event_parameters,
                                                  logger     = self.logger)

            # global version
            if workflow_dict['ptf_version'] == 'global':
                if workflow_dict['user_gridname'] is not None:
                    # copy user-provided topo-bathymetric grid
                    self.logger.info('Going to use user-provided topo-bathymetric grid')
                    cp = shutil.copy(workflow_dict['user_gridname'], os.path.join(workflow_dict['workdir'], workflow_dict['bathy_filename']))
                else:
                    # create simulation grid
                    ptf_mix_utilities.create_simulation_grid(longitude = event_parameters['lon'], 
                                                             latitude = event_parameters['lat'], 
                                                             distance = workflow_dict['dist_from_epi'],
                                                             gridfile_input = workflow_dict['grid_global'],
                                                             gridfile_output =  os.path.join(workflow_dict['workdir'], workflow_dict['bathy_filename']))
                # geographical selection of POIs based on the event location
                self.pois_d = ptf_mix_utilities.select_pois_from_epicenter(pois_d = self.pois_d,
                                                                           longitude = event_parameters['lon'], 
                                                                           latitude = event_parameters['lat'], 
                                                                           distance = workflow_dict['dist_from_epi'],
                                                                           user_gridname = workflow_dict['user_gridname'],
                                                                           logger = self.logger)

            run_steps.main(cfg           = self.config,
                           args          = self.args,
                           workflow_dict = workflow_dict,
                           event_dict    = event_parameters,
                           pois_d        = self.pois_d,
                           fcp           = self.fcp,
                           fcp_dict      = self.fcp_dict,
                           thresholds    = self.thresholds,
                           ptf_results   = ptf_results,
                           logger        = self.logger,
                           dataset       = self.dataset)

        except PyPTFException as e:
            self.logger.warning(f'PyPTFException: {str(e)}')
            ptf_results.set_status(status=str(e))

        except Exception as e:
            self.logger.error(f'Exception: {str(e)}')
            ptf_results.set_status(status=str(e))   

        if self.save:
            ptf_results.save_results()

        return ptf_results



    def __str__(self):
        pyptf_str = f"""
pyPTF Class: 
    - cfg_file: {self.args.cfg}
    - workflow_input: {self.args.input_workflow}
    - hazard_mode: {self.args.hazard_mode}
    - sigma: {self.args.sigma}
    - pre-loaded datasets: {self.dataset is not None}
    - type_df: {self.workflow_dict['type_df']}
"""
        return pyptf_str
