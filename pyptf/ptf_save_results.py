import os
import numpy as np
import scipy
import xarray as xr
import shutil
import polars as pl
import glob

class ptfResultGlobal:
    def __init__(self, workflow_dict):
        self.workflow_dict = workflow_dict

        self.results = {
            'step1': {},
            'step2': {},
            'step3': {},
            'step4': {},
            'step5': {},
        }

        self.alert_level_float2str = {
            '0.0': '',
            '1.0': 'information',
            '2.0': 'advisory',
            '3.0': 'watch',
            '0': '',
            '1': 'information',
            '2': 'advisory',
            '3': 'watch'
        }

    def set_result(self, step_name, result_key, result_value):     
        self.results[step_name][result_key] = result_value

    '''
        return alert level  for all pois for a given percentile (or 'matrix') in json format
        pois are returned also with name, lon, lat
    '''
    def get_pois_alert_levels(self, type='mean'): 
        al_string = [{'alert_level': self.alert_level_float2str[str(al_float)]} for al_float in self.results['step4']['pois_alert_levels'][type]]
            
        alert_levels_dataframe = pl.DataFrame(
            al_string,
            schema={"alert_level": pl.String}
        )

        pois_coordinates = pl.DataFrame(
            self.pois_d['pois_coords'],
            schema={"lon": pl.Float32, "lat": pl.Float32}
        )

        pois_labels = pl.DataFrame(
            self.pois_d['pois_labels'],
            schema={"name": pl.String}
        )

        final_dataframe = pl.concat([alert_levels_dataframe, pois_labels, pois_coordinates], how="horizontal") 

        return final_dataframe.write_json()

    '''
        return wave_weight for all pois for a given percentile in json format
        pois are returned also with name, lon, lat 
    '''
    def get_hc(self, type='mean'):            
        hc_dataframe = pl.DataFrame(
            self.results['step3']['hc_d'][type],
            schema={"wave_weight": pl.Float32}
        )

        pois_coordinates = pl.DataFrame(
            self.pois_d['pois_coords'],
            schema={"lon": pl.Float32, "lat": pl.Float32}
        )

        pois_labels = pl.DataFrame(
            self.pois_d['pois_labels'],
            schema={"name": pl.String}
        )

        final_dataframe = pl.concat([hc_dataframe, pois_labels, pois_coordinates], how="horizontal") 

        return final_dataframe.write_json()

class ptfResult(ptfResultGlobal):
    #def __init__(self, event_dict, workflow_dict, pois_d, thresholds, logger):
    def __init__(self, event_dict, pois_d, thresholds, logger, **kwargs):
        super().__init__(**kwargs) 

        self.event_dict = event_dict
        # self.workflow_dict = workflow_dict
        self.logger = logger
        self.pois_d = pois_d
        self.thresholds = thresholds
        self.status = None

    def update_workflow_dict(self, new_workflow_dict):
        self.workflow_dict = new_workflow_dict
       
    # def set_results(self, step_name, results):
    #     self.results[step_name] = results


    # def get_results(self, step_name):
    #     results_step = self.results[step_name]
    #     return results_step

    # def get_result(self, step_name, result_key):
    #     return self.results[step_name].get(result_key,None)

    # def get_status(self):
    #     return self.status
    
    def set_status(self, status):
        self.status = status

    def get_alert_levels(self, type='mean'):            
        #return {fcp_name: {'coordinates': fcp_coordinate,'alert_level': fcp_lev} for fcp_lev, fcp_name, fcp_coordinate in zip(self.results['step4']['fcp_alert_levels'][type], self.results['step4']['fcp_names'], self.results['step4']['fcp_coordinates'])}
    
        if not self.results['step4']:
            return []
        #TODO: creare 'alert_level_float2str' a partire dal file di configuratione e all'interno dell'init della pypytf e inserendolo nel workflow_dict

        return [ 
            {
            'id': fcp_id, 
            'lat': fcp_coordinate[1], 
            'lon': fcp_coordinate[0], 
            'alert_level': self.alert_level_float2str[str(fcp_lev)], 
            'name': fcp_name
            } for fcp_id, fcp_lev, fcp_name, fcp_coordinate in zip(
                self.results['step4']['fcp_ids'], 
                self.results['step4']['fcp_alert_levels'][type],
                self.results['step4']['fcp_names'], 
                self.results['step4']['fcp_coordinates']
            )
        ]


    # def saveAsJson():
    #     pass

    # def save_probability_scenarios(prob_scenarios_bs, prob_scenarios_ps, workflow_dict):
    #     """
    #     Write the list of parameters for each selected BS scenario in a text file
    #     formatted as required by the HySea code input
    #     """        
    #     np.save(os.path.join(workflow_dict['workdir'], workflow_dict['step1_prob_BS']), prob_scenarios_bs)
    #     np.save(os.path.join(workflow_dict['workdir'], workflow_dict['step1_prob_PS']), prob_scenarios_ps)


    def save_mih_precomputed(self, n_scenarios, n_pois, filename_out, workdir, mih):
        '''
        '''

        # convert mih to cm
        mih = np.rint(mih * 100.)

        outfile = os.path.join(workdir, filename_out)

        ds = xr.Dataset(
                # data_vars={'ts_max_gl': (['scenarios', 'pois'], np.transpose(mih))},
                data_vars={'ts_max_gl': (['scenarios', 'pois'], mih)},
                coords={'pois': range(n_pois), 'scenarios': range(n_scenarios)},
                attrs={'description': outfile, 'unit': 'cm', 'format': 'int16'})

        # encode = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 
        encode = {'zlib': True, 'complevel': 5, 'dtype': 'int16', 
                '_FillValue': False}
        encoding = {var: encode for var in ds.data_vars}
        ds.to_netcdf(outfile, format='NETCDF4', encoding=encoding)


    def save_results(self):

        self.logger.info("Save Results")
        #
        self.logger.info("..save selected POIs")
        outfile = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['pois'])
        np.save(outfile, self.pois_d, allow_pickle=True)
        n_pois = len(self.pois_d['pois_index'])

        self.logger.info("..STEP.1")
        #TODO: check all exceptions
        if (self.results['step1'] != {}):
            self.logger.info("....save scenarios list")
            # bs
            par_scenarios_bs = self.results['step1']['par_scen_bs']
            file_bs_list = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step1_list_BS'])
            np.savetxt(file_bs_list, par_scenarios_bs, delimiter=" ", fmt='%.6f')
            # if (par_scenarios_bs is not None):
            #     #note fmt='%.6f' is the equivalent of fmt="{:f}" in save_bs_scenarios_list
            #     np.savetxt(file_bs_list, par_scenarios_bs, delimiter=" ", fmt='%.6f')
            #     #save_bs_scenarios_list(par_scenarios_bs = self.results['step1']['par_scenarios_bs'],
            #     #                       file_bs_list = file_bs_list)
            # else:
            #     # when there are no BS scenarios we create an empty file
            #     f_list_bs = open(file_bs_list, 'w')
            #     f_list_bs.close()
            # ps
            par_scenarios_ps = self.results['step1']['par_scen_ps']
            file_ps_list = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step1_list_PS'])
            np.savetxt(file_ps_list, par_scenarios_ps, delimiter=" ",fmt="%s")
            # if (par_scenarios_ps is not None):
            #     np.savetxt(file_ps_list, par_scenarios_ps, delimiter=" ",fmt="%s")
            # else:
            #     f_list_ps = open(file_ps_list, 'w')
            #     f_list_ps.close()
            #
            # sbs for global
            par_scenarios_sbs = self.results['step1']['par_scen_sbs']
            file_sbs_list = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step1_list_SBS'])
            np.savetxt(file_sbs_list, par_scenarios_sbs, delimiter=" ", fmt='%.6f')

            self.logger.info("....save scenarios probability")
            # bs
            prob_scenarios_bs = self.results['step1']['prob_scen_bs']
            file_bs_prob = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step1_prob_BS'])
            np.save(file_bs_prob, prob_scenarios_bs)
            # ps
            prob_scenarios_ps = self.results['step1']['prob_scen_ps']
            file_ps_prob = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step1_prob_PS'])
            np.save(file_ps_prob, prob_scenarios_ps)
            # sbs
            prob_scenarios_sbs = self.results['step1']['prob_scen_sbs']
            file_sbs_prob = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step1_prob_SBS'])
            np.save(file_sbs_prob, prob_scenarios_sbs)

            # save most probable scenario
            most_prob_file = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step1_most_prob_filename'])
            np.save(most_prob_file, self.workflow_dict['most_probable_scenario'])
    
            # save_probability_scenarios(prob_scenarios_bs = self.results['step1']['prob_scenarios_bs'], 
            #                            prob_scenarios_ps = self.results['step1']['prob_scenarios_ps'],
            #                            workflow_dict     = self.workflow_dict)


        self.logger.info("..STEP.2")
        if (self.results['step2'] !={} and self.workflow_dict['tsu_sim'] == 'precomputed'):
            #
            self.logger.info("....save mihs value for bs scenarios")
            mih_bs = self.results['step2']['mih_bs']
            if (mih_bs.size > 0):
                self.save_mih_precomputed(n_scenarios  = mih_bs.shape[0],
                                          n_pois       = n_pois,
                                          filename_out = self.workflow_dict['step2_hmax_pre_BS'],
                                          workdir      = self.workflow_dict['workdir'],
                                          mih          = mih_bs)

            self.logger.info("....save mihs value for ps scenarios")
            mih_ps = self.results['step2']['mih_ps']
            if (mih_ps.size > 0):
                self.save_mih_precomputed(n_scenarios  = mih_ps.shape[0],
                                          n_pois       = n_pois,
                                          filename_out = self.workflow_dict['step2_hmax_pre_PS'],
                                          workdir      = self.workflow_dict['workdir'],
                                          mih          = mih_ps)


        self.logger.info("..STEP.3")
        if (self.results['step3'] != {}):
            #
            self.logger.info('....saving Hazard Curves (BS+PS) and Percentiles')
            hazard_curves_pois = self.results['step3']['hc_pois']
            # hazard_curves_pois_sparse = scipy.sparse.csr_matrix(hazard_curves_pois.astype(np.float16))
            hazard_curves_pois_sparse = scipy.sparse.csr_matrix(hazard_curves_pois.astype(np.float32))
            scipy.sparse.save_npz(os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step3_hc'] + '.npz'), hazard_curves_pois_sparse)
            #
            self.logger.info('....saving percentiles')
            hazard_curves_d = self.results['step3']['hc_d']
            np.save(os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step3_hc_perc'] + '.npy'), hazard_curves_d, allow_pickle=True)
            #
            if self.workflow_dict['compute_pdf']:
                self.logger.info('....saving PDF pois')
                pdf_pois = self.results['step3']['pdf_pois']
                np.save(os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step3_hazard_pdf']), pdf_pois)
            
            if self.workflow_dict['save_nc']:
                # TODO:thresholds missing
                outfile_hc = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step3_hc'] + '.nc')
                outfile_perc = os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step3_hc_perc'] + '.nc')

                ds = xr.Dataset(
                        data_vars={'hazard_curves': (['pois', 'thresholds'], hazard_curves_pois)},
                        coords={'pois': range(n_pois), 'thresholds': self.thresholds},
                        attrs={'description': outfile_hc})
                encode = {'zlib': True, 'complevel': 5, 'dtype': 'float32', '_FillValue': False}
                encoding = {var: encode for var in ds.data_vars}
                ds.to_netcdf(outfile_hc, format='NETCDF4', encoding=encoding)

                ds = xr.Dataset(
                        data_vars={k: (['pois'], v) for k, v in hazard_curves_d.items()},
                        coords={'pois': range(n_pois)},
                        attrs={'description': outfile_perc})

                encode = {'zlib': True, 'complevel': 5, 'dtype': 'float32', '_FillValue': False}
                encoding = {var: encode for var in ds.data_vars}
                ds.to_netcdf(outfile_perc, format='NETCDF4', encoding=encoding)


        self.logger.info("..STEP.4")
        if(self.results['step4'] !={}):

            self.logger.info('....saving alert levels over pois')
            pois_alert_levels = self.results['step4']['pois_alert_levels']
            np.save(os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step4_alert_levels_POI']), pois_alert_levels, allow_pickle=True)

            self.logger.info('....saving alert levels over fcp')
            fcp_alert_levels = self.results['step4']['fcp_alert_levels']
            np.save(os.path.join(self.workflow_dict['workdir'], self.workflow_dict['step4_alert_levels_FCP']), fcp_alert_levels, allow_pickle=True)
       
        # copy json file in the output folder
        if self.event_dict.get('event_file'):
            cp = shutil.copy(self.event_dict['event_file'], self.workflow_dict['workdir'])
            self.logger.info("..saving input event")

        self.logger.info("..saving workflow dictionary and status")
        np.save(os.path.join(self.workflow_dict['workdir'], self.workflow_dict['workflow_dictionary']), self.workflow_dict, allow_pickle=True)
        # saving exit status of pyptf
        with open(os.path.join(self.workflow_dict['workdir'], self.workflow_dict['status_file']), 'w') as f:
            f.write(self.status)

class PtfResultPostProcess(ptfResultGlobal):
    def __init__(self, resultDataDirectory, **kwargs):
        super().__init__(**kwargs) 

        self.files = os.listdir(resultDataDirectory)
        
        self.pois_d = np.load(os.path.join(resultDataDirectory, 'pois.npy'), allow_pickle=True).item()

        alert_levels_POI_file = glob.glob(os.path.join(resultDataDirectory, "step4_alert_levels_POI_*.npy"))
        if alert_levels_POI_file:
            alert_levels_POI = np.load(alert_levels_POI_file[0], allow_pickle=True).item()
            self.set_result('step4', 'pois_alert_levels', alert_levels_POI)

        hazard_curves_file = glob.glob(os.path.join(resultDataDirectory, "step3_hazard_curves_percentiles_*.npy"))
        if hazard_curves_file:
            hc = np.load(hazard_curves_file[0], allow_pickle=True).item()
            self.set_result('step3', 'hc_d', hc)

        alert_levels_FCP_file = glob.glob(os.path.join(resultDataDirectory, "step4_alert_levels_POI_*.npy"))
        if alert_levels_FCP_file:
            alert_levels_FCP = np.load(alert_levels_FCP_file[0], allow_pickle=True).item()
            pass
            #self.set_result('step4', 'pois_alert_levels', alert_levels_POI)
    