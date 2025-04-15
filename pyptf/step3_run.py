import os
import sys
import numpy as np
import scipy
import h5py
import pandas as pd
import polars as pl
# import xarray as xr

# from pyptf.ptf_preload import load_intensity_thresholds

def mylognorm_cdf(mean,sigma, threshold):
    ### CDF in terms of error function, see https://en.wikipedia.org/wiki/Log-normal_distribution
    return (0.5*(1+scipy.special.erf((np.log(threshold)-np.log(mean))/(sigma*np.sqrt(2)))))
    #return (np.log(threshold)-np.log(mean))/(sigma*np.sqrt(2))

def compute_hazard_curves(**kwargs):

    mih = kwargs.get('mih', None)
    n_scen = kwargs.get('n_scen', None)
    prob_scenarios = kwargs.get('prob_scenarios', None)
    n_pois = kwargs.get('n_pois', None)
    thresholds = kwargs.get('thresholds', None)
    sigma = kwargs.get('sigma', None)
    hazard_mode = kwargs.get('hazard_mode', None)
    compute_pdf = kwargs.get('compute_pdf', None)
    logger   = kwargs.get('logger', None)

    n_thr = len(thresholds)

    hazard_curves_pois = np.zeros((n_pois, n_thr))
    # hazard_curves_pois_m = np.zeros((n_pois))
    hazard_pdf_pois = np.zeros((n_pois, n_thr))

    # print(n_pois, n_thr, n_scen)
    # print(mih.shape, prob_scenarios.shape)

    if hazard_mode == 'lognormal':
        for ip in range(n_pois):

            mih_at_poi = mih[:,ip]
            ind_tmp = np.array(mih_at_poi == 0)
            mih_at_poi[ind_tmp] = 1.e-12

            mu = mih_at_poi
            mu = mu.reshape(len(mu), 1)

            cond_hazard_curve_tmp = 1 - scipy.stats.lognorm.cdf(thresholds, sigma, scale=mu).transpose()
            hazard_curves_pois[ip,:] = np.sum(prob_scenarios*cond_hazard_curve_tmp, axis=1)
            # logger.info(cond_hazard_curve_tmp.shape, prob_scenarios.shape)

            # cond_hazard_curve_tmp_mean = np.exp(np.log(mu) + 0.5*sigma**2)
            # hazard_curves_pois_m[ip] = np.sum(prob_scenarios*cond_hazard_curve_tmp_mean.transpose()[0])

            if compute_pdf:
                pdf_hazard_curve_tmp = scipy.stats.lognorm.pdf(thresholds, sigma, scale=mu).transpose()
                hazard_pdf_pois[ip,:] = np.sum(prob_scenarios*pdf_hazard_curve_tmp, axis=1)
    
    # hazard curve with lognorm with pandas (ie sparse matrix)
    elif hazard_mode == 'lognormal_v1':
        
        logger.info("HazMode = {}".format(hazard_mode))
        ### convert mih-array to a coo-sparse matrix (id_scen,id_poi,mih_scen_poi)
        mih_coo = scipy.sparse.coo_array(mih)
        #logger.info("Number of non-zero elements = {}".format(len(mih_coo.data)))
        ### convert coo-matrix to a pandas dataframe
        df_mihs = pd.DataFrame({"id_scen":mih_coo.row,"id_poi":mih_coo.col,"mih_value":mih_coo.data})
        #plogger.info(df_mihs.head(15))
        ### convert scenario probabilities array into a pandas dataframe (id_sce,prob_scen) 
        df_prob_scenarios = pd.DataFrame(prob_scenarios)\
                                    .reset_index()\
                                    .rename(columns={'index':'id_scen',0:'prob_scen'})
        ### associate a probability to each scenario (id_scen,id_poi,mih_scen_poi,prob_scen)
        df_mihs = df_mihs.merge(df_prob_scenarios,how='left',left_on='id_scen', right_on='id_scen')
        #plogger.info(df_mihs.sample(frac=1,random_state=1).head(15))

        ### loop over the thresholds
        for ith,threshold in enumerate(thresholds[:]):
            ## copy mih dataframe 
            df_mihs_thr = df_mihs.copy(deep=True)
            ## for each (scen,poi): compute lognorm for each mih_value and moltiply it by corresponding scenarios probability
            col_name = 'prob_lognorm_{}'.format(threshold)
            df_mihs_thr[col_name] = 1-scipy.stats.lognorm.cdf(threshold, sigma, scale=df_mihs_thr['mih_value']).transpose()
            #plogger.info(df_mihs_thr.sample(frac=1,random_state=1).head())
            df_mihs_thr[col_name] = df_mihs_thr[col_name]*df_mihs_thr['prob_scen']
            #plogger.info(df_mihs_thr.sample(frac=1,random_state=1).head())
            ## sum scenario probabilities for each poi (id_poi,sum(prob_lognorm over id_scen))
            df_mihs_thr = df_mihs_thr.groupby(by='id_poi').agg({col_name:'sum'})
            #plogger.info(df_mihs_thr.sample(frac=1,random_state=1).head())
            ## store result in a (dense) numpy array
            hazard_curves_pois[df_mihs_thr.index,ith]=df_mihs_thr[col_name].to_numpy()

            #TODO(andrea): adattarlo se necessario
            if compute_pdf:
                #pdf_hazard_curve_tmp = scipy.stats.lognorm.pdf(thresholds, sigma, scale=mu).transpose()
                #hazard_pdf_pois[ip,:] = np.sum(prob_scenarios*pdf_hazard_curve_tmp, axis=1)
                # logger.info("Not Done Yet")
                pass

    elif hazard_mode == 'lognormal_pl':
        #t0 = time.time()
        ### convert mih-array to a coo-sparse matrix (id_scen,id_poi,mih_scen_poi)
        #t1 = time.time()
        mih_coo = scipy.sparse.coo_array(mih)
        #logger.info("Time elapsed to create mih_coo sparse matrix = {}".format(time.time()-t1))
        logger.info("Number of non-zero elements = {}".format(len(mih_coo.data)))
        ### convert coo-matrix to a pandas dataframe
        #t1 = time.time()
        #df_mihs = pl.LazyFrame({"id_scen":mih_coo.row,"id_poi":mih_coo.col,"mih_value":mih_coo.data})
        df_mihs = pl.DataFrame({"id_scen":mih_coo.row,"id_poi":mih_coo.col,"mih_value":mih_coo.data})
        #logger.info("Time elapsed to create mih_coo polar dataframe = {}".format(time.time()-t1))
        #
        ### convert scenario probabilities array into a pandas dataframe (id_sce,prob_scen)
        #t1 = time.time()
        df_pl_prob_scenarios = pl.DataFrame(prob_scenarios,schema=['prob_scen'])\
                                 .with_row_index(name='scenario')\
                                 .select(pl.col("scenario").cast(pl.Int32),"prob_scen")
        #logger.info("Time elapsed to create scenarios probabilities polar dataframe = {}".format(time.time()-t1))
        ### associate a probability to each scenario (id_scen,id_poi,mih_scen_poi,prob_scen)
        #t1 = time.time()
        df_mihs = df_mihs.join(df_pl_prob_scenarios,how='left',left_on='id_scen', right_on='scenario')
        #logger.info("Time elapsed to join mihs and scenarios probabilities = {}".format(time.time()-t1))
        ### for each threshold we create a columns with lognormal cdf moltiplied with scenarios probability
        #t1 = time.time()
        df_mihs = df_mihs.with_columns([((1-mylognorm_cdf(mean=pl.col('mih_value'),sigma=sigma,threshold=thr))*pl.col('prob_scen')).alias('prob_lognorm_>{}'.format(thr)) for thr in thresholds])
        #logger.info("Time elapsed to compute lognormal cdf = {}".format(time.time()-t1))
        ### aggregate lognormal probability for each poi: first we create the different aggregation then we compute the groupby
        #t1 = time.time()
        agg_exprs = [pl.sum('prob_lognorm_>{}'.format(thr)).alias(f"prob_scen_{thr}") for thr in thresholds]
        df_results = df_mihs.group_by("id_poi").agg(agg_exprs)
        #logger.info("Time elapsed to compute hazard curves over pois = {}".format(time.time()-t1))
        #
        ### parse results in a 2d array
        row_indices = df_results["id_poi"].to_numpy().reshape(-1,1)
        data_values = df_results.select(pl.col('^prob_scen_.*$')).to_numpy()
        hazard_curves_pois[row_indices,np.arange(len(thresholds))] = data_values
        #for ith,thr in enumerate(thresholds):
        #    hazard_curves_pois[df_results['id_poi'].to_list(),ith] = df_results[f"prob_scen_{thr}"]
        #logger.info("Total time elapsed to compute Lognorm with polars = {}".format(time.time()-t0))


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
        raise Exception('{0} is not an option for computing hazard curves'.format(hazard_mode))

    return hazard_curves_pois, hazard_pdf_pois


def main(**kwargs):

    workflow_dict = kwargs.get('workflow_dict', None)             # workflow dictionary
    mih_bs   = kwargs.get('mih_bs', None)
    mih_ps   = kwargs.get('mih_ps', None)
    mih_sbs  = kwargs.get('mih_sbs', None)
    prob_bs  = kwargs.get('prob_bs', None)
    prob_ps  = kwargs.get('prob_ps', None)
    prob_sbs = kwargs.get('prob_sbs', None)
    pois_d   = kwargs.get('pois_d', None)
    thresholds = kwargs.get('thresholds', None)
    logger   = kwargs.get('logger', None)

    sigma = workflow_dict['logn_sigma']
    workdir = workflow_dict['workdir']
    tsu_sim = workflow_dict['tsu_sim']
    percentiles = workflow_dict['percentiles']
    n_percentiles = len(percentiles)
    hazard_mode = workflow_dict['hazard_mode']
    compute_pdf = workflow_dict['compute_pdf']
    # save_nc = workflow_dict['save_nc']

    # loading pois
    pois = pois_d['pois_index']
    n_pois = len(pois)

    # BS
    if prob_bs is None:
        file_bs = os.path.join(workdir, workflow_dict['step1_prob_BS'])
        try:
            prob_bs = np.load(file_bs)
        except:
            raise Exception(f"Error reading file: {file_bs}")
    
    n_bs = prob_bs.size
    
    if n_bs > 0:
        sum_prob_bs = np.sum(prob_bs)
      
        if tsu_sim == "to_run":
            ptf_measure_type = workflow_dict['ptf_measure_type']
            if mih_bs is None:
                file_hmax_sim_bs = os.path.join(workdir, workflow_dict['step2_hmax_sim_BS'])
                try:
                    imax_scenarios_bs = h5py.File(file_hmax_sim_bs, 'r')
                    mih_bs = np.array(imax_scenarios_bs[ptf_measure_type])
                except:
                    raise Exception(f"Error reading file: {file_hmax_sim_bs}")

        elif tsu_sim == "precomputed":
            ptf_measure_type = 'ts_max_gl'
            if mih_bs is None:
                file_hmax_pre_bs = os.path.join(workdir, workflow_dict['step2_hmax_pre_BS'])
                try:
                    imax_scenarios_bs = h5py.File(file_hmax_pre_bs, 'r')
                    # convert hmax from cm to m
                    mih_bs = np.array(imax_scenarios_bs[ptf_measure_type]) * 0.01
                except:
                    raise Exception(f"Error reading file: {file_hmax_pre_bs}")
        
        else:
            raise Exception('check tsu_sim input variable in the .json file')

        logger.info('Computing hazard curves from BS scenarios')
        hc_pois_bs, pdf_pois_bs = compute_hazard_curves(mih            = mih_bs,
                                                        n_scen         = n_bs,
                                                        prob_scenarios = prob_bs,
                                                        n_pois         = n_pois,
                                                        thresholds     = thresholds,
                                                        sigma          = sigma,
                                                        hazard_mode    = hazard_mode,
                                                        compute_pdf    = compute_pdf,
                                                        logger         = logger)
    else:
        logger.info('No BS scenarios for this event.')
        hc_pois_bs = np.zeros((n_pois, len(thresholds)))
        # hc_pois_bs_mean = np.zeros((n_pois))
        pdf_pois_bs = np.zeros((n_pois, len(thresholds)))
        sum_prob_bs = 0.

    # PS
    if prob_ps is None:
        file_ps = os.path.join(workdir, workflow_dict['step1_prob_PS'])
        try:
            prob_ps = np.load(file_ps)
        except:
            raise Exception(f"Error reading file: {file_ps}")
    
    n_ps = prob_ps.size

    if n_ps > 0:
        sum_prob_ps = np.sum(prob_ps)

        if tsu_sim == "to_run":
            ptf_measure_type = workflow_dict['ptf_measure_type']
            if mih_ps is None:
                file_hmax_sim_ps = os.path.join(workdir, workflow_dict['step2_hmax_sim_PS'])
                try:
                    imax_scenarios_ps = h5py.File(file_hmax_sim_ps, 'r')
                    mih_ps = np.array(imax_scenarios_ps[ptf_measure_type])
                except:
                    raise Exception(f"Error reading file: {file_hmax_sim_ps}")
    
        elif tsu_sim == "precomputed":
            ptf_measure_type = 'ts_max_gl'
            if mih_ps is None:
                file_hmax_pre_ps = os.path.join(workdir, workflow_dict['step2_hmax_pre_PS'])
                try:
                    imax_scenarios_ps = h5py.File(file_hmax_pre_ps, 'r')
                    # convert hmax from cm to m
                    mih_ps = np.array(imax_scenarios_ps[ptf_measure_type]) * 0.01
                except:
                    raise Exception(f"Error reading file: {file_hmax_pre_ps}")
        
        else:
            raise Exception('check tsu_sim input variable in the .json file')
    
        logger.info('Computing hazard curves from PS scenarios')
        hc_pois_ps, pdf_pois_ps = compute_hazard_curves(mih            = mih_ps,
                                                        n_scen         = n_ps,
                                                        prob_scenarios = prob_ps,
                                                        n_pois         = n_pois,
                                                        thresholds     = thresholds,
                                                        sigma          = sigma,
                                                        hazard_mode    = hazard_mode,
                                                        compute_pdf    = compute_pdf,
                                                        logger         = logger)
    else:
        logger.info('No PS scenarios for this event.')
        hc_pois_ps = np.zeros((n_pois, len(thresholds)))
        # hc_pois_ps_mean = np.zeros((n_pois))
        pdf_pois_ps = np.zeros((n_pois, len(thresholds)))
        sum_prob_ps = 0.

    # SBS
    if prob_sbs is None:
        file_sbs = os.path.join(workdir, workflow_dict['step1_prob_SBS'])
        try:
            prob_sbs = np.load(file_sbs)
        except:
            raise Exception(f"Error reading file: {file_sbs}")
    
    n_sbs = prob_sbs.size
    
    if n_sbs > 0:
        sum_prob_sbs = np.sum(prob_sbs)
      
        if tsu_sim == "to_run":
            ptf_measure_type = workflow_dict['ptf_measure_type']
            if mih_sbs is None:
                file_hmax_sim_sbs = os.path.join(workdir, workflow_dict['step2_hmax_sim_SBS'])
                try:
                    imax_scenarios_sbs = h5py.File(file_hmax_sim_sbs, 'r')
                    mih_sbs = np.array(imax_scenarios_sbs[ptf_measure_type])
                except:
                    raise Exception(f"Error reading file: {file_hmax_sim_sbs}")
        else:
            raise Exception('check tsu_sim input variable in the .json file')

        logger.info('Computing hazard curves from SBS scenarios')
        hc_pois_sbs, pdf_pois_sbs = compute_hazard_curves(mih            = mih_sbs,
                                                          n_scen         = n_sbs,
                                                          prob_scenarios = prob_sbs,
                                                          n_pois         = n_pois,
                                                          thresholds     = thresholds,
                                                          sigma          = sigma,
                                                          hazard_mode    = hazard_mode,
                                                          compute_pdf    = compute_pdf,
                                                          logger         = logger)
    else:
        logger.info('No SBS scenarios for this event.')
        hc_pois_sbs = np.zeros((n_pois, len(thresholds)))
        # hc_pois_sbs_mean = np.zeros((n_pois))
        pdf_pois_sbs = np.zeros((n_pois, len(thresholds)))
        sum_prob_sbs = 0.

    hc_pois = hc_pois_bs + hc_pois_ps + hc_pois_sbs
    # hazard_curves_pois_mean = hc_pois_bs_mean + hc_pois_ps_mean + hc_pois_sbs_mean
    expected_values = scipy.integrate.simpson(hc_pois, thresholds)
    pdf_pois = (pdf_pois_bs + pdf_pois_ps + pdf_pois_sbs) / (sum_prob_bs + sum_prob_ps + sum_prob_sbs)
    
    mih_percentiles = np.zeros((n_pois, n_percentiles)) 
    for i in range(n_pois):
        mih_percentiles[i,:] = np.interp(percentiles, hc_pois[i,::-1], thresholds[::-1])

    # defining saved dictionary
    hc_d = dict()
    # hc_d['mean'] = hazard_curves_pois_mean
    hc_d['mean'] = expected_values

    # percentiles' labels are expressed as 1-p since they represent the survival function (not exceedance)
    for ip, percentile in enumerate(percentiles):
        if percentile != 0:
            hc_d['p' +  '{:.0f}'.format((1-percentile)*100).zfill(2)] = mih_percentiles[:,ip]
        else:
            hc_d['envelope'] = mih_percentiles[:,ip]

    #most probable scenario
    all_mih = [mih_bs, mih_ps, mih_sbs] 
    idx_seis_type = workflow_dict['most_probable_scenario']['idx_seis_type']
    idx_max_prob_scen = workflow_dict['most_probable_scenario']['idscen'] - 1 #python array starts from zero
    hc_d['most_probable_scenario'] = all_mih[idx_seis_type][idx_max_prob_scen, :]

    return hc_pois, hc_d, pdf_pois

if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
