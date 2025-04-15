import os
import sys
# from scipy.stats import norm
# from ismember import ismember
import numpy as np
# from random import seed
# from random import random
from scipy import stats
# from numpy.linalg import inv
from math import radians, cos, sin, asin, sqrt

# from pyptf.ptf_probability_scenarios import save_bs_scenarios_list
# from pyptf.ptf_probability_scenarios import save_ps_scenarios_list
# from pyptf.ptf_probability_scenarios import save_probability_scenarios


def compute_ensemble_sampling_SDE(**kwargs):

    workflow_dict  = kwargs.get('workflow_dict', None)
    LongTermInfo   = kwargs.get('LongTermInfo', None)
    negl_prob      = kwargs.get('negligible_prob', None)
    pre_selection  = kwargs.get('pre_selection', None)
    short_term     = kwargs.get('short_term', None)
    regions        = kwargs.get('regions', None)
    probability_scenarios = kwargs.get('proba_scenarios', None)
    SDE_samp_scen         = kwargs.get('samp_scen', None)
    samp_type             = kwargs.get('samp_type', None)
    logger                = kwargs.get('logger', None)

    TotProbBS_all = np.sum(probability_scenarios['ProbScenBS'])
    TotProbPS_all = np.sum(probability_scenarios['ProbScenPS'])
    sampled_ensemble = {}

    ### Begining of the creation of the Nth sampled ensemble ###
    N=int(SDE_samp_scen)
    NBS= int(TotProbBS_all*N) # NBS: number of scenarios sampled from the BS ensemble
    NPS= N-NBS                # NPS: number of scenarios sampled from the PS ensemble
    # print(NBS,NPS)            # NBS and NPS are proportionnal to the probability of PS and BS
    sampled_ensemble = proscen.set_if_compute_scenarios(short_term    = short_term,
                                                        negl_prob     = negl_prob)
    
    prob_len_BS = len(probability_scenarios['ProbScenBS'])
    prob_len_PS = len(probability_scenarios['ProbScenPS'])

    ### Creation of the array of cumulated probability intervals associated to the initial ensemble ###
    intervals_ensemble_BS = np.zeros(prob_len_BS)
    probBSnorm = probability_scenarios['ProbScenBS']/np.sum(probability_scenarios['ProbScenBS'])
    prob_cum = 0
    for i in range(prob_len_BS):
        prob_cum=prob_cum+probBSnorm[i]
        intervals_ensemble_BS[i]= prob_cum

    ### Creation of the array of cumulated probability intervals associated to the initial ensemble ###
    intervals_ensemble_PS = np.zeros(prob_len_PS)
    probPSnorm = probability_scenarios['ProbScenPS']/np.sum(probability_scenarios['ProbScenPS'])
    prob_cum = 0
    for i in range(prob_len_PS):
        prob_cum=prob_cum+probPSnorm[i]
        intervals_ensemble_PS[i]= prob_cum

    ### Initialization of the dictionnaries ###
    sampled_ensemble['prob_scenarios_bs_fact'] = np.zeros( (NBS,  5) )
    sampled_ensemble['prob_scenarios_bs'] = np.zeros( (NBS) )
    sampled_ensemble['prob_scenarios_ang'] = np.zeros( (NBS) )
    sampled_ensemble['par_scenarios_bs'] = np.zeros(  (NBS, 11) )
    sampled_ensemble['prob_scenarios_ps_fact'] = np.zeros( (NPS,  5) )
    sampled_ensemble['prob_scenarios_ps'] = np.zeros( (NPS) )
    sampled_ensemble['par_scenarios_ps'] = np.zeros(  (NPS,  7) )
    sampled_ensemble['iscenbs']=np.zeros(NBS)
    sampled_ensemble['iscenps']=np.zeros(NPS)

    sampled_ensemble = bs_probability_scenarios(short_term         = short_term,
                                                pre_selection      = pre_selection,
                                                regions_files      = regions,
                                                prob_scenes        = probability_scenarios,
                                                samp_ens           = sampled_ensemble,
                                                Discretizations    = LongTermInfo['Discretizations'],
    				                	        NBS                = NBS,
                                                intervals_ensemble = intervals_ensemble_BS,
                                                samp_type          = samp_type,
                                                logger             = logger)

    sampled_ensemble = ps_probability_scenarios(short_term         = short_term,
                                                prob_scenes        = probability_scenarios,
                                                samp_ens           = sampled_ensemble,
                                                NPS                = NPS,
                                                samp_type          = samp_type,
                                                intervals_ensemble = intervals_ensemble_PS,
                                                logger             = logger)
    
    if(sampled_ensemble == False):
        return False

    # Re-Normalize scenarios (to manage events outside nSigma, BS for large events, ...)
    ProbScenBS = sampled_ensemble['prob_scenarios_bs_fact'].prod(axis=1)
    ProbScenPS = sampled_ensemble['prob_scenarios_ps_fact'].prod(axis=1)
    #logger.info(len(ProbScenBS))
    TotProbBS_preNorm = np.sum(ProbScenBS)
    TotProbPS_preNorm = np.sum(ProbScenPS)
    TotProb_preNorm   = TotProbBS_preNorm + TotProbPS_preNorm

    # No scenarios bs or ps possible
    if(TotProb_preNorm == 0):
        return False
    elif(TotProb_preNorm != 0):
        ProbScenBS = ProbScenBS / TotProb_preNorm
        ProbScenPS = ProbScenPS / TotProb_preNorm
        logger.info(' --> Total BS scenarios probability pre-renormalization: {:.5f}'.format(TotProbBS_preNorm))
        logger.info(' --> Total PS scenarios probability pre-renormalization: {:.5f}'.format(TotProbPS_preNorm))
        logger.info('     --> Total BS and PS probabilty renormalized to 1')

    TotBS=len(ProbScenBS)
    TotPS=len(ProbScenPS)
    Tot=TotBS+TotPS
    ProbScenBS = np.ones(TotBS)
    ProbScenPS = np.ones(TotPS)

    ######### Re-initialisation of the probability ######
    ## A uniform probability is attributed to all the new scenarios ##
    ## The probability is then modified proportionnaly to the number of repetitions ##
    sampled_ensemble['ProbScenBS'] = np.ones(TotBS)*1./Tot
    sampled_ensemble['ProbScenPS'] = np.ones(TotPS)*1./Tot
    
    prob_angles_sum = np.sum(sampled_ensemble['prob_scenarios_ang'])
    prob_angles_tot = sampled_ensemble['prob_scenarios_ang']/prob_angles_sum

    ######### Duplication of scenarios beginning ##########
    ## The numbers of dupplicated scenarios are saved, then mutliplied by their respective probability 
    ## and the duplicates erased from the ensemble

    sample_unique_bs, test, counts_bs = np.unique(sampled_ensemble['iscenbs'],return_index=True,return_counts=True) 
    unique_par = sampled_ensemble['par_scenarios_bs'][test,:]
    unique_fact = sampled_ensemble['prob_scenarios_bs_fact'][test,:]
    unique_prob = sampled_ensemble['ProbScenBS'][test]
    unique_ang = sampled_ensemble['prob_scenarios_ang'][test]
    unique_name = sampled_ensemble['iscenbs'][test]
    ProbScenBS = np.ones(len(unique_prob))*1./Tot
    for itmp in range(len(unique_prob)):
       iscenbs=unique_name[itmp]
       indexbs=np.where(sample_unique_bs == iscenbs)
       ProbScenBS[itmp]=unique_prob[itmp]*counts_bs[indexbs]

    sampled_ensemble['par_scenarios_bs'] = unique_par
    sampled_ensemble['prob_scenarios_bs_fact'] = unique_fact
    sampled_ensemble['ProbScenBS'] = ProbScenBS #/np.sum(ProbScenBS)
    sampled_ensemble['relevant_scenarios_bs'] = np.unique(sampled_ensemble['par_scenarios_bs'][:,0])
    ProbScenBS = sampled_ensemble['ProbScenBS'] 

    sample_unique_ps, test, counts_ps = np.unique(sampled_ensemble['iscenps'],return_index=True,return_counts=True)
    unique_par = sampled_ensemble['par_scenarios_ps'][test,:]
    unique_fact = sampled_ensemble['prob_scenarios_ps_fact'][test,:]
    unique_prob = sampled_ensemble['ProbScenPS'][test]
    unique_name = sampled_ensemble['iscenps'][test]
    ProbScenPS = np.ones(len(unique_prob))*1./Tot
    for itmp in range(len(unique_prob)):
       iscenps=unique_name[itmp]
       indexps=np.where(sample_unique_ps == iscenps)
       ProbScenPS[itmp]=unique_prob[itmp]*counts_ps[indexps]

    sampled_ensemble['par_scenarios_ps'] = unique_par
    sampled_ensemble['prob_scenarios_ps_fact'] = unique_fact
    sampled_ensemble['ProbScenPS'] = ProbScenPS
    sampled_ensemble['iscenps'] = unique_name
    sampled_ensemble['relevant_scenarios_ps'] = np.unique(sampled_ensemble['par_scenarios_ps'][:,0])

    ### Re-normalization of all the probabilities ###
    TotProbBS = np.sum(ProbScenBS)
    TotProbPS = np.sum(ProbScenPS)

    logger.info(TotProbBS)
    logger.info(' --> Relevant Scenarios BS : {}'.format(sampled_ensemble['relevant_scenarios_bs']))
    logger.info(' --> Relevant Scenarios PS : {}'.format(sampled_ensemble['relevant_scenarios_ps']))

    logger.info('Number of sampled BS scenarios: {}'.format(len(sampled_ensemble['par_scenarios_bs'])))
    logger.info('Number of sampled PS scenarios: {}'.format(len(sampled_ensemble['par_scenarios_ps'])))
    
    # file_bs_list = os.path.join(workflow_dict['workdir'], workflow_dict['step1_list_BS'])
    # file_ps_list = os.path.join(workflow_dict['workdir'], workflow_dict['step1_list_PS'])

    # # saving list of bs scenarios parameters
    # save_bs_scenarios_list(par_scenarios_bs = sampled_ensemble['par_scenarios_bs'],
    #                        file_bs_list     = file_bs_list)
    
    # # saving list of ps scenarios
    # save_ps_scenarios_list(par_scenarios_ps =  sampled_ensemble['par_scenarios_ps'],
    #                        Discretizations  = LongTermInfo['Discretizations'],
    #                        Regionalization  = LongTermInfo['Regionalization'],
    #                        file_ps_list     = file_ps_list)

    # # saving probability of scenarios bs and ps
    # save_probability_scenarios(prob_scenarios_bs = sampled_ensemble['ProbScenBS'],
    #                            prob_scenarios_ps = sampled_ensemble['ProbScenPS'],
    #                            workflow_dict     = workflow_dict)

    try:
        max_idxBS = np.argmax(ProbScenBS)
    except:
        max_idxBS = -1
    try:
        max_ValBS = ProbScenBS[max_idxBS]
    except:
        max_ValBS = 0
    try:
        max_idxPS = np.argmax(ProbScenPS)
    except:
        max_idxPS = -1
    try:
        max_ValPS = ProbScenPS[max_idxPS]
    except:
        max_ValPS = 0

        max_ValPS = 0

        max_ValPS = 0

    sampled_ensemble['best_scenarios'] = {'max_idxBS':max_idxBS, 'max_idxPS':max_idxPS, 'max_ValBS':max_ValBS, 'max_ValPS':max_ValPS}

    logger.info('     --> Best BS scenario Idx and Value: {} {:.5e}'.format(max_idxBS, max_ValBS))
    logger.info('     --> Best PS scenario Idx and Value: {} {:.5e}'.format(max_idxPS, max_ValPS))
    
    return sampled_ensemble


def find_nearest(array, value):
    arr = np.asarray(array)
    idx = 0
    diff = arr-value
    diff[diff<1e-26]=100.0
    idx=diff.argmin()
    return idx,array[idx]


def bs_probability_scenarios(**kwargs):

    short_term      = kwargs.get('short_term', None)
    prob_scenes     = kwargs.get('prob_scenes', None)
    samp_ens        = kwargs.get('samp_ens', None)
    pre_selection   = kwargs.get('pre_selection', None)
    Discretizations = kwargs.get('Discretizations', None)
    region_files    = kwargs.get('regions_files', None)
    NBS	            = kwargs.get('NBS', None)
    int_ens         = kwargs.get('intervals_ensemble', None)
    samp_type       = kwargs.get('samp_type', None)
    logger          = kwargs.get('logger', None)

    region_info     = dict()

    if (samp_ens['BScomputedYN'] == False or short_term['BS_computed_YN'] == False or pre_selection['BS_scenarios'] == False):
        samp_ens['nr_bs_scenarios'] = 0
        return samp_ens
    regions_nr = []

    ### Generation of an array (size of the new ensemble) of random probability 
    if samp_type=='MC':
       random_value = np.random.random(NBS)
    if samp_type=='LH':
       sampler = stats.qmc.LatinHypercube(d=1)
       random_value = sampler.random(n=NBS)
    
    iscenbs=0
    for i in random_value:
        ### Each value is associated to a scenario that can be retrieved from the cumulative probability function
        idx, proba = find_nearest(int_ens,i)
        ### samp_ens corresponds to the new ensemble where the identification nb of each scenario in 
        ### the initial ensemble is saved in iscenbs, and the parameters and the probability as well
        samp_ens['iscenbs'][iscenbs]=idx
        samp_ens['prob_scenarios_bs'][iscenbs]=prob_scenes['ProbScenBS'][idx]
        for j in range(5):
            samp_ens['prob_scenarios_bs_fact'][iscenbs,j]=prob_scenes['prob_scenarios_bs_fact'][idx,j]
        for j in range(11):
            samp_ens['par_scenarios_bs'][iscenbs,j]=prob_scenes['par_scenarios_bs'][idx,j] 
 
        # Inside the original code the strike/dip/rake
        # do not depend of the magnitude and position
        latlon_1=prob_scenes['par_scenarios_bs'][idx,2]
        latlon_0=prob_scenes['par_scenarios_bs'][idx,3]
        bs2_pos = len(pre_selection['BS2_Position_Selection_inn'])
        d_latlon=np.zeros((bs2_pos,2))
        d_diff=np.zeros((bs2_pos))
        for val in range(bs2_pos):
                tmp_idx = pre_selection['BS2_Position_Selection_inn'][val]
                d_latlon[val,:] = Discretizations['BS-2_Position']['Val'][tmp_idx].split()
                d_diff[val] = haversine(latlon_1, latlon_0, d_latlon[val,0], d_latlon[val,1])
        ipos_idx = int(np.argmin(d_diff))
        ipos = pre_selection['BS2_Position_Selection_inn'][ipos_idx]
        ireg = Discretizations['BS-2_Position']['Region'][ipos] #prob_scenes['par_scenarios_bs'][idx,0] #Discretizations['BS-2_Position']['Region'][ipos]
        
        if(ireg not in regions_nr):
            #logger.info("...............................", region_files)
            region_info = load_region_infos(ireg         = ireg,
                                            region_info  = region_info,
                                            region_files = region_files)
            regions_nr.append(ireg)
        
        RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_valNorm']
        
        if(RegMeanProb_BS4.size == 0):
             logger.info(' --> WARNING: region info %d is empty!!! {}'.format(ireg))
        
        ipos_reg = np.where(region_info[ireg]['BS4_FocMech_iPosInRegion'] == ipos+1)[1]
        tmpProbAngles = RegMeanProb_BS4[ipos_reg[0]]
        id_sel = 0
        val_angles = prob_scenes['par_scenarios_bs'][idx,5]+100*prob_scenes['par_scenarios_bs'][idx,6]+10000*prob_scenes['par_scenarios_bs'][idx,7]
        for angles_id in range(len(Discretizations['BS-4_FocalMechanism']['Val'])):
            str_val,dip_val,rak_val = Discretizations['BS-4_FocalMechanism']['Val'][angles_id].split()
            val_check = float(str_val)+100*float(dip_val)+10000*float(rak_val)
            if abs(val_angles-val_check)<0.0001:
               #logger.info("One matching value for:",iscenbs,angles_id,tmpProbAngles[angles_id])
               #logger.info(str_val,dip_val,rak_val)
               id_sel=angles_id
        ProbAngles = tmpProbAngles[id_sel]
        samp_ens['prob_scenarios_ang'][iscenbs]=ProbAngles

        iscenbs=iscenbs+1

    samp_ens['nr_bs_scenarios'] = np.shape(samp_ens['prob_scenarios_bs_fact'])[0]
    return samp_ens


def ps_probability_scenarios(**kwargs):

    short_term         = kwargs.get('short_term', None)
    prob_scenes        = kwargs.get('prob_scenes', None)
    samp_ens           = kwargs.get('samp_ens', None)
    NPS                = kwargs.get('NPS', None)
    int_ens            = kwargs.get('intervals_ensemble', None)
    samp_type          = kwargs.get('samp_type', None)
    logger             = kwargs.get('logger', None)

    samp_ens['PScomputedYN'] == False
    
    if(samp_ens['PScomputedYN'] == False or short_term['PS_computed_YN'] == False):

        #fbfix 2021-11-26
        samp_ens['PScomputedYN']    = False
        short_term['PS_computed_YN']   = False
        samp_ens['nr_ps_scenarios'] = 0

        return samp_ens

    ### Generation of an array (size of the new ensemble) of random probability 
    if samp_type=='MC':
       random_value = np.random.random(NPS)
    if samp_type=='LH':
       sampler = stats.qmc.LatinHypercube(d=1)
       random_value = sampler.random(n=NPS)

    iscenps=0
    for i in random_value:
        ### Each value is associated to a scenario that can be retrieved from the cumulative probability function
        idx,proba = find_nearest(int_ens,i)
        ### samp_ens corresponds to the new ensemble where the identification nb of each scenario in 
        ### the initial ensemble is saved in iscenbs, and the parameters and the probability as well
        samp_ens['iscenps'][iscenps]=idx
        samp_ens['prob_scenarios_ps'][iscenps]=prob_scenes['ProbScenPS'][idx]
        for j in range(5):
            samp_ens['prob_scenarios_ps_fact'][iscenps,j]=prob_scenes['prob_scenarios_ps_fact'][idx,j]
        for j in range(7):
            samp_ens['par_scenarios_ps'][iscenps,j]=prob_scenes['par_scenarios_ps'][idx,j]
        iscenps=iscenps+1

    samp_ens['nr_ps_scenarios'] = np.shape(samp_ens['prob_scenarios_ps_fact'])[0]

    return samp_ens


def load_region_infos(**kwargs):

    ireg        = kwargs.get('ireg', None)
    files       = kwargs.get('region_files', None)
    region_info = kwargs.get('region_info', None)

    info = np.load(files['ModelsProb_Region_files'][ireg-1], allow_pickle=True).item()
    region_info[ireg] = info

    return region_info

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

