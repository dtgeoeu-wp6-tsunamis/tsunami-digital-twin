import os
import sys
import ast
import time
import numpy as np
from scipy.stats import norm

from ismember import ismember


def compute_probability_scenarios(**kwargs):

    Config           = kwargs.get('cfg', None)
    workflow_dict    = kwargs.get('workflow_dict', None)
    workdir          = kwargs.get('workdir', None)
    ee               = kwargs.get('event_parameters', None)
    negl_prob        = kwargs.get('negligible_prob', None)
    args             = kwargs.get('args', None)
    LongTermInfo     = kwargs.get('LongTermInfo', None)
    PSBarInfo        = kwargs.get('PSBarInfo', None)
    lambda_bsps      = kwargs.get('lambda_bsps', None)
    pre_selection    = kwargs.get('pre_selection', None)
    short_term       = kwargs.get('short_term', None)
    regions          = kwargs.get('regions', None)
    Scenarios_PS     = kwargs.get('Scenarios_PS', None)
    Regionalization  = kwargs.get('Regionalization', None)
    samp_mode        = kwargs.get('samp_mode', None)
    
    # Check if Xs schenarios prob must be computed and inizialize dictionary
    # probability_scenarios = set_if_compute_scenarios(negl_prob     = negl_prob,
    #                                                 short_term    = short_term)

    probability_scenarios = dict()
    probability_scenarios['nr_ps_scenarios'] = 0
    probability_scenarios['nr_bs_scenarios'] = 0
    
    #file_bs_list = os.path.join(workdir, 'Step1_scenario_list_BS.txt')
    #file_ps_list = os.path.join(workdir, 'Step1_scenario_list_PS.txt')
    file_bs_list = os.path.join(workdir, workflow_dict['step1_list_BS'])
    file_ps_list = os.path.join(workdir, workflow_dict['step1_list_PS'])

    # if (probability_scenarios['BScomputedYN'] == True and short_term['BS_computed_YN'] == True and pre_selection['BS_scenarios'] == True):
    if (short_term['BS_computed_YN'] == True and pre_selection['BS_scenarios'] == True):
    
        probability_scenarios['par_scenarios_bs']       = np.zeros( (int(short_term['Total_BS_Scenarios']), 11) )
        probability_scenarios['prob_scenarios_bs_fact'] = np.zeros( (int(short_term['Total_BS_Scenarios']), 5) )
        probability_scenarios = bs_probability_scenarios(cfg              = Config,
                                                         short_term       = short_term,
                                                         pre_selection    = pre_selection,
                                                         regions_files    = regions,
                                                         prob_scenes      = probability_scenarios,
                                                         Discretizations  = LongTermInfo['Discretizations'])

        probability_scenarios['relevant_scenarios_bs'] = np.unique(probability_scenarios['par_scenarios_bs'][:,0])
        # print(' --> Unload empty regions')
        # probability_scenarios['relevant_scenarios_bs'] = remove_empty_scenarios(scenarios=probability_scenarios['relevant_scenarios_bs'], cfg=Config)

        print(' --> Used regions for BS : ', probability_scenarios['relevant_scenarios_bs'])

        # Re-Normalize scenarios (to manage events outside nSigma, BS for large events, ...)
        ProbScenBS        = probability_scenarios['prob_scenarios_bs_fact'].prod(axis=1)
        TotProbBS_preNorm = np.sum(ProbScenBS)

        # saving list of bs scenarios parameters
        if samp_mode == 'None':
            save_bs_scenarios_list(par_scenarios_bs = probability_scenarios['par_scenarios_bs'],
                                   file_bs_list     = file_bs_list)
    else:
        print('No BS scenarios')
        ProbScenBS = 0
        TotProbBS_preNorm = 0
        # saving empty file
        f_list_bs = open(file_bs_list, 'w')
        f_list_bs.close()

    # if (probability_scenarios['PScomputedYN'] == True and short_term['PS_computed_YN'] == True):
    if (short_term['PS_computed_YN'] == True):

        # Some inizializations
        probability_scenarios['par_scenarios_ps']       = np.zeros( (int(short_term['Total_PS_Scenarios']), 7) )
        probability_scenarios['prob_scenarios_ps_fact'] = np.zeros( (int(short_term['Total_PS_Scenarios']), 5) )

        probability_scenarios = ps_probability_scenarios(PSBarInfo        = PSBarInfo,
                                                         short_term       = short_term,
                                                         pre_selection    = pre_selection,
                                                         prob_scenes      = probability_scenarios,
                                                         region_ps        = regions['region_listPs'], #        = LongTermInfo['region_listPs'],
                                                         Discretizations  = LongTermInfo['Discretizations'],
                                                         Model_Weights    = LongTermInfo['Model_Weights'],
                                                         Scenarios_PS     = Scenarios_PS,
                                                         ps1_magnitude    = LongTermInfo['Discretizations']['PS-1_Magnitude'],
                                                         lambda_bsps      = lambda_bsps)

        probability_scenarios['relevant_scenarios_ps'] = np.unique(probability_scenarios['par_scenarios_ps'][:,0])
        print(' --> Used regions for PS : ', probability_scenarios['relevant_scenarios_ps'])
        # Re-Normalize scenarios (to manage events outside nSigma, BS for large events, ...)
        ProbScenPS        = probability_scenarios['prob_scenarios_ps_fact'].prod(axis=1)
        TotProbPS_preNorm = np.sum(ProbScenPS)
        # saving list of ps scenarios
        if samp_mode == 'None':
            save_ps_scenarios_list(par_scenarios_ps = probability_scenarios['par_scenarios_ps'],
                                   Discretizations  = LongTermInfo['Discretizations'],
                                   Regionalization  = Regionalization,
                                   file_ps_list     = file_ps_list)

    else:
        print('No PS scenarios')
        #probability_scenarios['PScomputedYN']  = False
        #short_term['PS_computed_YN'] = False
        ProbScenPS = 0
        TotProbPS_preNorm = 0
        # saving empty file
        f_list_ps = open(file_ps_list, 'w')
        f_list_ps.close()


    if (probability_scenarios == False):
        return False

    TotProb_preNorm = TotProbBS_preNorm + TotProbPS_preNorm

    # No scenarios bs or ps possible
    if (TotProb_preNorm == 0):
        return False

    if (TotProb_preNorm < 1.0):
        ProbScenBS = ProbScenBS / TotProb_preNorm
        ProbScenPS = ProbScenPS / TotProb_preNorm
        print(' --> Total BS scenarios probability pre-renormalization: %.5f' % TotProbBS_preNorm)
        print(' --> Total PS scenarios probability pre-renormalization: %.5f' % TotProbPS_preNorm)
        print('     --> Total BS and PS probabilty renormalized to 1')

    probability_scenarios['ProbScenBS'] = ProbScenBS
    probability_scenarios['ProbScenPS'] = ProbScenPS

    # saving probability of scenarios bs and ps
    if samp_mode == 'None':
        save_probability_scenarios(prob_scenarios_bs = probability_scenarios['ProbScenBS'], 
                                   prob_scenarios_ps = probability_scenarios['ProbScenPS'],
                                   workflow_dict     = workflow_dict,
                                   workdir           = workdir)
 
    # check on the nr scenarios computed into the two section. Should be identical
    check_bs = 'OK'
    check_ps = 'OK'

    print(' --> Total_BS_Scenarios:      %7d' % probability_scenarios['nr_bs_scenarios'])
    print(' --> Total_PS_Scenarios:      %7d' % probability_scenarios['nr_ps_scenarios'])
 
    if (probability_scenarios['nr_bs_scenarios'] != short_term['Total_BS_Scenarios']):
        check_bs = 'WARNING'
        print(' --> Check Nr BS scenarios: %7d  <--> %7d --> %s' % (probability_scenarios['nr_bs_scenarios'], short_term['Total_BS_Scenarios'], check_bs))
    if (probability_scenarios['nr_ps_scenarios'] != short_term['Total_PS_Scenarios']):
        check_ps = 'WARNING'
        print(' --> Check Nr PS scenarios: %7d  <--> %7d --> %s' % (probability_scenarios['nr_ps_scenarios'], short_term['Total_PS_Scenarios'], check_ps))


#    try:
#        max_idxBS = np.argmax(ProbScenBS)
#    except:
#        max_idxBS = -1
#    try:
#        max_ValBS = ProbScenBS[max_idxBS]
#    except:
#        max_ValBS = 0
#    try:
#        max_idxPS = np.argmax(ProbScenPS)
#    except:
#        max_idxPS = -1
#    try:
#        max_ValPS = ProbScenPS[max_idxPS]
#    except:
#        max_ValPS = 0

#    probability_scenarios['best_scenarios'] = {'max_idxBS':max_idxBS, 
#                                               'max_idxPS':max_idxPS, 
#                                               'max_ValBS':max_ValBS, 
#                                               'max_ValPS':max_ValPS}

#    print('     --> Max prob BS scenario Idx and Value: %6d    %.5e' % (max_idxBS, max_ValBS))
#    print('     --> Max prob PS scenario Idx and Value: %6d    %.5e' % (max_idxPS, max_ValPS))

    return probability_scenarios


def bs_probability_scenarios(**kwargs):

    Config          = kwargs.get('cfg', None)
    short_term      = kwargs.get('short_term', None)
    prob_scenes     = kwargs.get('prob_scenes', None)
    pre_selection   = kwargs.get('pre_selection', None)
    Discretizations = kwargs.get('Discretizations', None)
    region_files    = kwargs.get('regions_files', None)

    region_info = dict()
    regions_nr = []

    empty_scenarios = np.array(ast.literal_eval(Config.get('ScenariosList', 'bs_empty')))
    empty_scenarios = empty_scenarios.astype('float64')

    iScenBS = 0
    sel_mag = len(pre_selection['sel_BS_Mag_idx'])
    bs2_pos = len(pre_selection['BS2_Position_Selection_common'])
    foc_ids = len(Discretizations['BS-4_FocalMechanism']['ID'])

    for i1 in range(sel_mag):
        imag = pre_selection['sel_BS_Mag_idx'][i1]

        for i2 in range(bs2_pos):
            ipos = pre_selection['BS2_Position_Selection_common'][i2]
            ireg = Discretizations['BS-2_Position']['Region'][ipos]
            
            if ireg in empty_scenarios:
                continue

            if (ireg not in regions_nr):
                region_info = load_region_infos(ireg         = ireg,
                                                region_info  = region_info,
                                                region_files = region_files)

                regions_nr.append(ireg)
                # print("...............................", ireg, region_info, region_files)


            RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_valNorm']
            # print(type(RegMeanProb_BS4))
            # print(ireg, RegMeanProb_BS4.size)


            if(RegMeanProb_BS4.size == 0):
                 print(' --> WARNING: region info %d is empty!!!' % (ireg) )


            ipos_reg = np.where(region_info[ireg]['BS4_FocMech_iPosInRegion'] == ipos+1)[1]
            tmpProbAngles = RegMeanProb_BS4[ipos_reg[0]]

            len_depth_valvec = len(Discretizations['BS-3_Depth']['ValVec'][imag][ipos])

            # I3 (depth) AND I4 (angles) ENUMERATE ALL RELEVANT SCENARIOS FOR EACH MAG AND POS (Equivalent to compute_scenarios_prefixes)
            for i3 in range(len_depth_valvec):

                for i4 in range(foc_ids):

                    mag               = Discretizations['BS-1_Magnitude']['Val'][imag]
                    lon, lat          = Discretizations['BS-2_Position']['Val'][ipos].split()
                    depth             = Discretizations['BS-3_Depth']['ValVec'][imag][ipos][i3]
                    strike, dip, rake = Discretizations['BS-4_FocalMechanism']['Val'][i4].split()
                    area              = Discretizations['BS-5_Area']['ValArea'][ireg-1, imag, i4]
                    length            = Discretizations['BS-5_Area']['ValLen'][ireg-1, imag, i4]
                    slip              = Discretizations['BS-6_Slip']['Val'][ireg-1, imag, i4]

                    prob_scenes['par_scenarios_bs'][iScenBS][0]  = int(ireg)
                    prob_scenes['par_scenarios_bs'][iScenBS][1]  = float(mag)
                    prob_scenes['par_scenarios_bs'][iScenBS][2]  = float(lon)
                    prob_scenes['par_scenarios_bs'][iScenBS][3]  = float(lat)
                    prob_scenes['par_scenarios_bs'][iScenBS][4]  = float(depth)
                    prob_scenes['par_scenarios_bs'][iScenBS][5]  = float(strike)
                    prob_scenes['par_scenarios_bs'][iScenBS][6]  = float(dip)
                    prob_scenes['par_scenarios_bs'][iScenBS][7]  = float(rake)
                    prob_scenes['par_scenarios_bs'][iScenBS][8]  = float(length)
                    prob_scenes['par_scenarios_bs'][iScenBS][9]  = float(area)
                    prob_scenes['par_scenarios_bs'][iScenBS][10] = float(slip)

                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][0] = short_term['magnitude_probability'][imag]
                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][1] = short_term['PosProb'][i1, i2]
                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][2] = short_term['RatioBSonTot'][imag]
                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][3] = short_term['DepProbScenes'][i1, i2][i3]
                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][4] = tmpProbAngles[i4]

                    iScenBS = iScenBS + 1

    prob_scenes['par_scenarios_bs'] = prob_scenes['par_scenarios_bs'][:iScenBS,:]
    prob_scenes['prob_scenarios_bs_fact'] = prob_scenes['prob_scenarios_bs_fact'][:iScenBS,:]
    prob_scenes['nr_bs_scenarios'] = np.shape(prob_scenes['prob_scenarios_bs_fact'])[0]

    return prob_scenes


def load_region_infos(**kwargs):

    ireg        = kwargs.get('ireg', None)
    files       = kwargs.get('region_files', None)
    region_info = kwargs.get('region_info', None)

    info = np.load(files['ModelsProb_Region_files'][ireg-1], allow_pickle=True).item()
    region_info[ireg] = info
    return region_info


def ps_probability_scenarios(**kwargs):

    short_term      = kwargs.get('short_term', None)
    prob_scenes     = kwargs.get('prob_scenes', None)
    pre_selection   = kwargs.get('pre_selection', None)
    Model_Weights   = kwargs.get('Model_Weights', None)
    Discretizations = kwargs.get('Discretizations', None)
    regions         = kwargs.get('regions', None)
    PSBarInfo       = kwargs.get('PSBarInfo', None)
    region_ps       = kwargs.get('region_ps', None)
    Scenarios_PS    = kwargs.get('Scenarios_PS', None)
    ps1_magnitude   = kwargs.get('ps1_magnitude', None)
    lambda_bsps     = kwargs.get('lambda_bsps', None)

#    if (prob_scenes['PScomputedYN'] == False or short_term['PS_computed_YN'] == False):

#        # fbfix 2021-11-26
#        prob_scenes['PScomputedYN']  = False
#        short_term['PS_computed_YN'] = False
#        prob_scenes['nr_ps_scenarios'] = 0
#        return prob_scenes

    iScenPS = 0

    # print("Select PS Probability Scenarios")

    sel_mag = len(pre_selection['sel_PS_Mag_idx'][0])
    sel_imod = len(Model_Weights['PS2_Bar']['Wei'])

    par_scenarios_ps       = np.zeros((0,7))
    prob_scenarios_ps_fact = np.zeros((0,5))
    nrireg = 0

    for i1 in range(sel_mag):

        imag = pre_selection['sel_PS_Mag_idx'][0][i1]
        Imag = pre_selection['sel_PS_Mag_val'][i1]
        IMAG = imag+1

        tmp_b = np.where(short_term['PS_model_YN'][imag] == 1)

        # fb fixed 2023-02-23: from  sel_imod-1 to sel_imod
        for imod in range(sel_imod):

            ps_models = pre_selection['Inside_in_BarPSperModel'][imag][imod]['inside']

            if (len(ps_models) == 0):
                continue

            for i2 in range(len(ps_models)):

                # ibar = -1
                ibar = pre_selection['Inside_in_BarPSperModel'][imag][imod]['inside'][i2]

                try:
                    ireg = PSBarInfo['BarPSperModelReg'][imag][imod][ibar][0]
                except:
                    try:
                        ireg = PSBarInfo['BarPSperModelReg'][imag][imod][ibar]
                    except:
                        ireg = np.int64(PSBarInfo['BarPSperModelReg'][imag][imod])

                try:
                    nr_reg = region_ps[ireg-1]
                except:
                    continue

                tmp_a = int(lambda_bsps['regionsPerPS'][ireg-1])

                if (short_term['sel_RatioPSonPSTot'][nr_reg-1] > 0):

                    selected_maPsIndex        = np.where(Scenarios_PS[ireg]['magPSInd'] == imag+1)  #### Gli indici che iniziano da 1
                    selected_SlipDistribution = np.take(Scenarios_PS[ireg]['SlipDistribution'], selected_maPsIndex)
                    slipVal                   = np.unique(selected_SlipDistribution)
                    nScen                     = len(slipVal)
                    locScen                   = np.array(range(iScenPS +1, iScenPS + nScen))
                    vectmp                    = np.ones(nScen)

                    if (len(vectmp) == 0):
                        return False

                    tmp_par_scenarios_ps       = np.zeros((len(vectmp), 7))
                    tmp_prob_scenarios_ps_fact = np.zeros((len(vectmp), 5))


                    for k in range(len(vectmp)):

                        # fb 2022-04-21
                        # To fix approximation errors: use round(x,5) for 2 and 3 elements
                        # THIS may be very weak. Better to think new algorithm
                        tmp_par_scenarios_ps[k][0] = vectmp[k] * ireg
                        tmp_par_scenarios_ps[k][1] = vectmp[k] * ps1_magnitude['Val'][imag]
                        tmp_par_scenarios_ps[k][2] = round(vectmp[k] * PSBarInfo['BarPSperModel'][imag][imod]['pos_xx'][ibar],5)
                        tmp_par_scenarios_ps[k][3] = round(vectmp[k] * PSBarInfo['BarPSperModel'][imag][imod]['pos_yy'][ibar],5)
                        tmp_par_scenarios_ps[k][4] = vectmp[k] * imod + 1 # matlab start from 1
                        tmp_par_scenarios_ps[k][5] = slipVal[k]

                        try:
                            tmp_par_scenarios_ps[k][6] = vectmp[k] * PSBarInfo['BarPSperModelDepth'][imag][imod][ibar]
                        except:
                            tmp_par_scenarios_ps[k][6] = vectmp[k] * PSBarInfo['BarPSperModelDepth'][imag][imod]

                        tmp_prob_scenarios_ps_fact[k][0] = vectmp[k] * short_term['magnitude_probability'][imag]
                        tmp_prob_scenarios_ps_fact[k][1] = vectmp[k] * short_term['BarProb'][imod][imag][i2]
                        tmp_prob_scenarios_ps_fact[k][2] = vectmp[k] * short_term['RatioPSonTot'][imag] * short_term['RatioPSonTot'][tmp_a]
                        tmp_prob_scenarios_ps_fact[k][3] = vectmp[k] * Model_Weights['PS2_Bar']['Wei'][imod] / np.sum(Model_Weights['PS2_Bar']['Wei'][tmp_b])
                        tmp_prob_scenarios_ps_fact[k][4] = vectmp[k] / nScen

                    prob_scenarios_ps_fact = np.concatenate((prob_scenarios_ps_fact, tmp_prob_scenarios_ps_fact), axis=0)
                    par_scenarios_ps       = np.concatenate((par_scenarios_ps, tmp_par_scenarios_ps), axis=0)

                    iScenPS = iScenPS + nScen

    prob_scenes['prob_scenarios_ps_fact'] = prob_scenarios_ps_fact
    prob_scenes['par_scenarios_ps']       = par_scenarios_ps
    prob_scenes['nr_ps_scenarios']        = np.shape(prob_scenes['prob_scenarios_ps_fact'])[0]

    return prob_scenes

def save_bs_scenarios_list(**kwargs):
    """
    Write the list of parameters for each selected BS scenario in a text file
    formatted as required by the HySea code input
    """
    par_scenarios_bs = kwargs.get('par_scenarios_bs', None)
    file_bs_list     = kwargs.get('file_bs_list', None)

    len_scenarios_bs, len_pars = par_scenarios_bs.shape

    # fmt = "{:.4f}".format
    fmt = "{:f}".format
    with open(file_bs_list, 'w') as f_list_bs:
        for ic in range(len_scenarios_bs):
            pars = " ".join(fmt(par_scenarios_bs[ic, item]) for item in range(1, len_pars))
            ireg = "{:.0f}".format(par_scenarios_bs[ic, 0])
            f_list_bs.write("{:s} {:s} {:s}\n".format(str(ic+1), ireg, pars))

def save_ps_scenarios_list(**kwargs):
    """
    Write the list of ids for each selected PS scenario in a text file
    linked to PS initial conditions (for HySea)
    """
    par_scenarios_ps = kwargs.get('par_scenarios_ps', None)
    Discretizations  = kwargs.get('Discretizations', None)
    Regionalization  = kwargs.get('Regionalization', None)
    file_ps_list     = kwargs.get('file_ps_list', None)

    len_scenarios_ps, _ = par_scenarios_ps.shape

    # region labels
    reg_pos = [int(ireg-1) for ireg in par_scenarios_ps[:,0]]
    sel_reg_id = [Regionalization['ID'][ireg] for ireg in reg_pos]
    # magnitude labels
    magnitude_labels = Discretizations['PS-1_Magnitude']['ID']
    magnitude_vals = Discretizations['PS-1_Magnitude']['Val']
    #for magval in par_scenarios_ps[:,1]:
        #print(magval)
        #print(magnitude_vals.index(magval))
        #print(magnitude_labels[magnitude_vals.index(magval)])
    sel_mag_id = [magnitude_labels[magnitude_vals.index(magval)] for magval in par_scenarios_ps[:,1]]
    # position labels
    position_labels = Discretizations['PS-2_PositionArea']['ID']
    position_vals = [(x,y) for x,y in zip(Discretizations['PS-2_PositionArea']['Val_x'], Discretizations['PS-2_PositionArea']['Val_y'])]
    sel_pos_id = [position_labels[position_vals.index((posx,posy))].split('_')[1] for posx,posy in zip(par_scenarios_ps[:,2],par_scenarios_ps[:,3])]
    # model alternative labels
    model_vals = [int(mval) for mval in par_scenarios_ps[:,4]]
    model_labels = ['Str_PYes_Hom', 'Str_PNo_Hom', 'Mur_PYes_Hom', 'Mur_PNo_Hom', 'Str_PYes_Var', 'Str_PNo_Var', 'Mur_PYes_Var', 'Mur_PNo_Var']
    sel_model_id = [model_labels[k-1] for k in model_vals]
    # slip labels
    sel_slip_id = ['S00' + str(int(sval)) for sval in par_scenarios_ps[:,5]]

    with open(file_ps_list, 'w') as f_list_ps:
        for ic in range(len_scenarios_ps):
            id_scenario_ps = "".join([sel_reg_id[ic],'-PS-',sel_model_id[ic] ,'-',sel_mag_id[ic],'_',sel_pos_id[ic],'_',sel_slip_id[ic]])
            ireg = "{:.0f}".format(par_scenarios_ps[ic, 0])
            f_list_ps.write("{0} {1} {2}.txt\n".format(str(ic+1), ireg, id_scenario_ps))


def save_probability_scenarios(**kwargs):
    """
    Write the list of parameters for each selected BS scenario in a text file
    formatted as required by the HySea code input
    """
    prob_scenarios_bs = kwargs.get('prob_scenarios_bs', None)
    prob_scenarios_ps = kwargs.get('prob_scenarios_ps', None)
    workflow_dict     = kwargs.get('workflow_dict', None)
    workdir           = kwargs.get('workdir', None)
    
    np.save(os.path.join(workdir, workflow_dict['step1_prob_BS']), prob_scenarios_bs)
    np.save(os.path.join(workdir, workflow_dict['step1_prob_PS']), prob_scenarios_ps)
    

def set_if_compute_scenarios(**kwargs):
 
    short_term = kwargs.get('short_term', None)
    negl_prob = kwargs.get('negl_prob', None)
 
    out = dict()
 
    # Some inizialization
    out['nr_ps_scenarios'] = 0
    out['nr_bs_scenarios'] = 0
 
    BScomputedYN = False
    PScomputedYN = False

    tmpbs = (short_term['magnitude_probability'] * short_term['RatioBSonTot']).sum()
    tmpps = (short_term['magnitude_probability'] * short_term['RatioPSonTot']).sum()
    if(tmpbs > negl_prob):
        BScomputedYN = True
    if(tmpps > negl_prob):
        PScomputedYN = True

    out['BScomputedYN'] = BScomputedYN
    out['PScomputedYN'] = PScomputedYN

    print(short_term['magnitude_probability'])
    print(short_term['RatioBSonTot'])
    print(short_term['RatioPSonTot'])

    print(' --> Negligible Probability: %.4f' % negl_prob)
    print(' --> Sum of probability BS scenarios: %.4f  --> Compute Probability Scenarios: %r' % (tmpbs, BScomputedYN))
    print(' --> Sum of probability PS scenarios: %.4f  --> Compute Probability Scenarios: %r' % (tmpps, PScomputedYN))

    return out

#def remove_empty_scenarios(**kwargs):

#    Config    = kwargs.get('cfg', None)
#    scenarios = kwargs.get('scenarios', None)

#    empty_scenarios = np.array(ast.literal_eval(Config.get('ScenariosList','bs_empty')))
#    empty_scenarios = empty_scenarios.astype('float64')

#    commons, idx = ismember(empty_scenarios, scenarios)
#    scenarios = np.delete(scenarios, idx)
#    return scenarios
