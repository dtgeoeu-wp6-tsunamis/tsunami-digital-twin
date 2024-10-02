import os
import sys
import configparser
import numpy as np
import scipy
import h5py
import xarray as xr
from ptf_preload import load_intensity_thresholds
import pandas as pd
from pprint import pprint

def compute_hazard_curves(**kwargs):

    mih = kwargs.get('mih', None)
    n_scen = kwargs.get('n_scen', None)
    prob_scenarios = kwargs.get('prob_scenarios', None)
    n_pois = kwargs.get('n_pois', None)
    thresholds = kwargs.get('thresholds', None)
    sigma = kwargs.get('sigma', None)
    hazard_mode = kwargs.get('hazard_mode', None)
    compute_pdf = kwargs.get('compute_pdf', None)

    n_thr = len(thresholds)

    hazard_curves_pois = np.zeros((n_pois, n_thr))
    # hazard_curves_pois_m = np.zeros((n_pois))
    hazard_pdf_pois = np.zeros((n_pois, n_thr))

    if hazard_mode == 'lognormal':
        for ip in range(n_pois):

            mih_at_poi = mih[:,ip]
            ind_tmp = np.array(mih_at_poi == 0)
            mih_at_poi[ind_tmp] = 1.e-12

            mu = mih_at_poi
            mu = mu.reshape(len(mu), 1)

            cond_hazard_curve_tmp = 1 - scipy.stats.lognorm.cdf(thresholds, sigma, scale=mu).transpose()
            hazard_curves_pois[ip,:] = np.sum(prob_scenarios*cond_hazard_curve_tmp, axis=1)
            # print(cond_hazard_curve_tmp.shape, prob_scenarios.shape)

            # cond_hazard_curve_tmp_mean = np.exp(np.log(mu) + 0.5*sigma**2)
            # hazard_curves_pois_m[ip] = np.sum(prob_scenarios*cond_hazard_curve_tmp_mean.transpose()[0])

            if compute_pdf:
                pdf_hazard_curve_tmp = scipy.stats.lognorm.pdf(thresholds, sigma, scale=mu).transpose()
                hazard_pdf_pois[ip,:] = np.sum(prob_scenarios*pdf_hazard_curve_tmp, axis=1)
    
    # hazard curve with lognorm with pandas (ie sparse matrix)
    elif hazard_mode == 'lognormal_v1':
        
        print("HazMode = {}".format(hazard_mode))
        ### convert mih-array to a coo-sparse matrix (id_scen,id_poi,mih_scen_poi)
        mih_coo = scipy.sparse.coo_array(mih)
        #print("Number of non-zero elements = {}".format(len(mih_coo.data)))
        ### convert coo-matrix to a pandas dataframe
        df_mihs = pd.DataFrame({"id_scen":mih_coo.row,"id_poi":mih_coo.col,"mih_value":mih_coo.data})
        #pprint(df_mihs.head(15))
        ### convert scenario probabilities array into a pandas dataframe (id_sce,prob_scen) 
        df_prob_scenarios = pd.DataFrame(prob_scenarios)\
                                    .reset_index()\
                                    .rename(columns={'index':'id_scen',0:'prob_scen'})
        ### associate a probability to each scenario (id_scen,id_poi,mih_scen_poi,prob_scen)
        df_mihs = df_mihs.merge(df_prob_scenarios,how='left',left_on='id_scen', right_on='id_scen')
        #pprint(df_mihs.sample(frac=1,random_state=1).head(15))

        ### loop over the thresholds
        for ith,threshold in enumerate(thresholds[:]):
            ## copy mih dataframe 
            df_mihs_thr = df_mihs.copy(deep=True)
            ## for each (scen,poi): compute lognorm for each mih_value and moltiply it by corresponding scenarios probability
            col_name = 'prob_lognorm_{}'.format(threshold)
            df_mihs_thr[col_name] = 1-scipy.stats.lognorm.cdf(threshold, sigma, scale=df_mihs_thr['mih_value']).transpose()
            #pprint(df_mihs_thr.sample(frac=1,random_state=1).head())
            df_mihs_thr[col_name] = df_mihs_thr[col_name]*df_mihs_thr['prob_scen']
            #pprint(df_mihs_thr.sample(frac=1,random_state=1).head())
            ## sum scenario probabilities for each poi (id_poi,sum(prob_lognorm over id_scen))
            df_mihs_thr = df_mihs_thr.groupby(by='id_poi').agg({col_name:'sum'})
            #pprint(df_mihs_thr.sample(frac=1,random_state=1).head())
            ## store result in a (dense) numpy array
            hazard_curves_pois[df_mihs_thr.index,ith]=df_mihs_thr[col_name].to_numpy()

            #TODO(andrea): adattarlo se necessario
            if compute_pdf:
                #pdf_hazard_curve_tmp = scipy.stats.lognorm.pdf(thresholds, sigma, scale=mu).transpose()
                #hazard_pdf_pois[ip,:] = np.sum(prob_scenarios*pdf_hazard_curve_tmp, axis=1)
                # print("Not Done Yet")
                pass
        """
        #########################################
        #region TEST lognormal_v1 vs lognormal
        print("Test lognormal_v1...")
        # compute lognormal 
        test_hazard_pdf_pois = np.zeros((n_pois, n_thr))
        for ip in range(n_pois):
            print("Step = ",ip)
            mih_at_poi = mih[:,ip]
            ind_tmp = np.array(mih_at_poi == 0)
            mih_at_poi[ind_tmp] = 1.e-12
            mu = mih_at_poi
            mu = mu.reshape(len(mu), 1)
            cond_hazard_curve_tmp = 1 - scipy.stats.lognorm.cdf(thresholds, sigma, scale=mu).transpose()
            test_hazard_pdf_pois[ip,:] = np.sum(prob_scenarios*cond_hazard_curve_tmp, axis=1)
        # difference of hazard curves
        precison = 11
        delta_hazmap = np.subtract(np.around(test_hazard_pdf_pois,precison),np.around(hazard_curves_pois,precison))
        res = np.any(delta_hazmap)
        if(res==False):
            print("...All elements are zero (up to {}th decimal) :D".format(precison))
        else:
            print("...There are non-zero elements (max delta={}) :(".format(np.amax(np.absolute(delta_hazmap)))) 
        print("DONE")
        #endregion
        #########################################
        """

    elif hazard_mode == 'no_uncertainty':
        
        scenarioYN = np.ones((n_scen), dtype=bool)

        for ith, threshold in enumerate(thresholds):
            lambdaCumTmp = np.zeros((n_pois))
            
            for iscen in range(n_scen):

                if scenarioYN[iscen]:
                    ipois = np.where(mih[iscen,:] >= threshold)
                    if len(ipois) != 0:
                        lambdaCumTmp[ipois] += prob_scenarios[iscen]
                    else:
                        scenarioYN[iscen] = False
            hazard_curves_pois[:,ith] = lambdaCumTmp

    else:
        sys.exit('{0} is not an option for computing hazard curves'.format(hazard_mode))

    return hazard_curves_pois, hazard_pdf_pois


def main(**kwargs):

    cfg_file = kwargs.get('cfg_file', None)                       # Configuration file
    workflow_dict = kwargs.get('workflow_dict', None)             # workflow dictionary
    # args          = kwargs.get('args', None)

    Config = configparser.RawConfigParser()
    Config.read(cfg_file)

    sigma = float(Config.get('Settings', 'hazard_curve_sigma'))

    workdir = workflow_dict['workdir']
    tsu_sim = workflow_dict['tsu_sim']
    percentiles = workflow_dict['percentiles']
    n_percentiles = len(percentiles)
    # nr_bs   = workflow_dict['nr_bs_scenarios']
    # nr_ps   = workflow_dict['nr_ps_scenarios']
    hazard_mode = workflow_dict['hazard_mode']
    compute_pdf = workflow_dict['compute_pdf']
    save_nc = workflow_dict['save_nc']

    # loading intensity thresholds (mih)
    thresholds, intensity_measure = load_intensity_thresholds(cfg = Config)
    
    # loading pois
    # pois = np.load(os.path.join(workdir, 'pois.npy'), allow_pickle=True).item()['pois_coords']
    # n_pois, _ = pois.shape
    pois = np.load(os.path.join(workdir, workflow_dict['pois']), allow_pickle=True).item()['pois_index']
    n_pois = len(pois)

    #inpfileBS = os.path.join(workdir, 'Step1_scenario_list_BS.txt')
    #inpfilePS = os.path.join(workdir, 'Step1_scenario_list_PS.txt')
    inpfileBS = os.path.join(workdir, workflow_dict['step1_list_BS'])
    inpfilePS = os.path.join(workdir, workflow_dict['step1_list_PS'])

    imax_scenarios_bs = False
    imax_scenarios_ps = False
    prob_scenarios_bs = False
    prob_scenarios_ps = False

    # BS
    if os.path.getsize(inpfileBS) > 0:
        prob_scenarios_bs = np.load(os.path.join(workdir, workflow_dict['step1_prob_BS']))
        sum_prob_bs = np.sum(prob_scenarios_bs)
      
        if tsu_sim == "to_run":
            ptf_measure_type = Config.get('Settings', 'ptf_measure_type')
            try:
                imax_scenarios_bs = h5py.File(os.path.join(workdir, workflow_dict['step2_hmax_sim_BS']), 'r')
                mih_bs = np.array(imax_scenarios_bs[ptf_measure_type])
            except:
                sys.exit('File Step2_BS_hmax_sim.nc not found')

        elif tsu_sim == "precomputed":
            ptf_measure_type = 'ts_max_gl'
            try:
                imax_scenarios_bs = h5py.File(os.path.join(workdir, workflow_dict['step2_hmax_pre_BS']), 'r')
                # convert hmax from cm to m
                mih_bs = np.array(imax_scenarios_bs[ptf_measure_type]) * 0.01
#                print(mih_bs[:,24])

            except:
                sys.exit('File Step2_BS_hmax_pre.nc not found')
        
        else:
            sys.exit('check tsu_sim input variable in the .json file')
    
        # POIs
        n_bs, _ = mih_bs.shape
        print('Computing hazard curves from BS scenarios')
        hc_pois_bs, pdf_pois_bs = compute_hazard_curves(mih            = mih_bs,
                                                        n_scen         = n_bs,
                                                        prob_scenarios = prob_scenarios_bs,
                                                        n_pois         = n_pois,
                                                        thresholds     = thresholds,
                                                        sigma          = sigma,
                                                        hazard_mode    = hazard_mode,
                                                        compute_pdf    = compute_pdf)


    else:
        print('No BS scenarios for this event.')
        hc_pois_bs = np.zeros((n_pois, len(thresholds)))
        hc_pois_bs_mean = np.zeros((n_pois))
        pdf_pois_bs = np.zeros((n_pois, len(thresholds)))
        sum_prob_bs = 0.

    # PS
    if os.path.getsize(inpfilePS) > 0:
        
        prob_scenarios_ps = np.load(os.path.join(workdir, workflow_dict['step1_prob_PS']))
        sum_prob_ps = np.sum(prob_scenarios_ps)

        if tsu_sim == "to_run":
            ptf_measure_type = Config.get('Settings', 'ptf_measure_type')
            try:
                imax_scenarios_ps = h5py.File(os.path.join(workdir, workflow_dict['step2_hmax_sim_PS']), 'r')
                mih_ps = np.array(imax_scenarios_ps[ptf_measure_type])
            except:
                sys.exit('File Step2_PS_hmax_sim.nc not found')
    
        elif tsu_sim == "precomputed":
            ptf_measure_type = 'ts_max_gl'
            try:
                imax_scenarios_ps = h5py.File(os.path.join(workdir, workflow_dict['step2_hmax_pre_PS']), 'r')
                # convert hmax from cm to m
                mih_ps = np.array(imax_scenarios_ps[ptf_measure_type]) * 0.01
            except:
                sys.exit('File Step2_PS_hmax_pre.nc not found')
        
        else:
            sys.exit('check tsu_sim input variable in the .json file')
    
        # POIs
        n_ps, _ = mih_ps.shape
        print('Computing hazard curves from PS scenarios')
        hc_pois_ps, pdf_pois_ps = compute_hazard_curves(mih            = mih_ps,
                                                        n_scen         = n_ps,
                                                        prob_scenarios = prob_scenarios_ps,
                                                        n_pois         = n_pois,
                                                        thresholds     = thresholds,
                                                        sigma          = sigma,
                                                        hazard_mode    = hazard_mode,
                                                        compute_pdf    = compute_pdf)
    else:
        print('No PS scenarios for this event.')
        hc_pois_ps = np.zeros((n_pois, len(thresholds)))
        hc_pois_ps_mean = np.zeros((n_pois))
        pdf_pois_ps = np.zeros((n_pois, len(thresholds)))
        sum_prob_ps = 0.

     
    hazard_curves_pois = hc_pois_bs + hc_pois_ps
    #hazard_curves_pois_mean = hc_pois_bs_mean + hc_pois_ps_mean
    expected_values = scipy.integrate.simpson(hazard_curves_pois, thresholds)
    pdf_pois = (pdf_pois_bs + pdf_pois_ps) / (sum_prob_bs + sum_prob_ps)
    
    mih_percentiles = np.zeros((n_pois, n_percentiles)) 
    for i in range(n_pois):
        mih_percentiles[i,:] = np.interp(percentiles, hazard_curves_pois[i,::-1], thresholds[::-1])

    # defining saved dictionary
    hc = dict()
    #hc['mean'] = hazard_curves_pois_mean
    hc['mean'] = expected_values

    # percentiles' labels are expressed as 1-p since they represent the survival function (not exceedance)
    for ip, percentile in enumerate(percentiles): 
        hc["p" +  "{:.0f}".format((1-percentile)*100).zfill(2)] = mih_percentiles[:,ip]

    print('Saving Hazard Curves (BS+PS) and Percentiles')
    #np.save(os.path.join(workdir, 'Step3_hazard_curves.npy'), hazard_curves_pois)
    hazard_curves_pois_sparse = scipy.sparse.csr_matrix(hazard_curves_pois.astype(np.float16))
    scipy.sparse.save_npz(os.path.join(workdir, workflow_dict['step3_hc'] + '.npz'), hazard_curves_pois_sparse)
    np.save(os.path.join(workdir, workflow_dict['step3_hc_perc'] + '.npy'), hc, allow_pickle=True)

    if compute_pdf:

        np.save(os.path.join(workdir, workflow_dict['step3_hazard_pdf']), pdf_pois)

    # mih_all = np.concatenate((mih_bs,mih_ps), axis=0)
    # mih_poi = mih_all[:,24]
    # prob_all = np.concatenate((prob_scenarios_bs,prob_scenarios_ps), axis=0)
    # import matplotlib.pyplot as plt
    # plt.hist(mih_poi, bins=30, density=True, alpha=0.5, color='b')
    # plt.hist(mih_poi, bins=30, density=True, alpha=0.5, color='b', weights = prob_all)
    ## plt.show()

    if save_nc:

        pois_index = range(n_pois)
        outfile_hc = os.path.join(workdir, workflow_dict['step3_hc'] + '.nc')
        outfile_perc = os.path.join(workdir, workflow_dict['step3_hc_perc'] + '.nc')

        ds = xr.Dataset(
                data_vars={'hazard_curves': (['pois', 'thresholds'], hazard_curves_pois)},
                coords={'pois': pois_index, 'thresholds': thresholds},
                attrs={'description': outfile_hc})
        encode = {'zlib': True, 'complevel': 5, 'dtype': 'float32', '_FillValue': False}
        encoding = {var: encode for var in ds.data_vars}
        ds.to_netcdf(outfile_hc, format='NETCDF4', encoding=encoding)

        ds = xr.Dataset(
                data_vars={k: (['pois'], v) for k, v in hc.items()},
                coords={'pois': pois_index},
                attrs={'description': outfile_perc})

        encode = {'zlib': True, 'complevel': 5, 'dtype': 'float32', '_FillValue': False}
        encoding = {var: encode for var in ds.data_vars}
        ds.to_netcdf(outfile_perc, format='NETCDF4', encoding=encoding)


if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
