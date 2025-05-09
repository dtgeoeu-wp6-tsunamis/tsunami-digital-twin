import sys
import copy
import numpy as np
import numpy.matlib as npm
from scipy.stats import norm

from pyptf.ptf_mix_utilities import ray_tracing_method
from pyptf.ptf_mix_utilities import NormMultiDvec
from pyptf.ptf_scaling_laws  import correct_BS_vertical_position
from pyptf.ptf_scaling_laws  import correct_BS_horizontal_position
from pyptf.ptf_scaling_laws  import correct_PS_horizontal_position

def short_term_probability_distribution(**kwargs):

    Config         = kwargs.get('cfg', None)
    ee             = kwargs.get('event_parameters', None)
    negl_prob      = kwargs.get('negligible_prob', None)
    LongTermInfo   = kwargs.get('LongTermInfo', None)
    PSBarInfo      = kwargs.get('PSBarInfo', None)
    lambda_bsps    = kwargs.get('lambda_bsps', None)
    pre_selection  = kwargs.get('pre_selection', None)
    logger         = kwargs.get('logger', None)

    short_term = dict()
    short_term['DepProbPoints'] = dict()   #b questo poi lo chiamiamo short_term_depth_prob_points
    short_term['DepProbTemps'] = dict()   #b questo poi non serve piu apparentemente
    short_term['DepProbScenes'] = dict()   #b questo poi lo chiamiamo short_term_depth_prob_scenes

    x = np.size(pre_selection['sel_BS_Mag_val'])
    y = np.size(pre_selection['BS2_Position_Selection_inn'])
    short_term['DepProbScenesN'] = np.zeros((x,y))

    x = np.size(LongTermInfo['Model_Weights']['PS2_Bar']['Wei'])
    y = np.size(LongTermInfo['Discretizations']['PS-1_Magnitude']['Val'])

    short_term['BarProb'] = [[0 for j in range(y)] for i in range(x)]
    short_term['PS_model_YN'] = np.ones([y,x], dtype = int)

    #COMPUTE INTEGRAL FOR MAGNITUDES
    short_term = compute_distribution_for_magnitudes(Discretizations  = LongTermInfo['Discretizations'],
                                                     short_term       = short_term,
                                                     event_parameters = ee,
                                                     logger           = logger)

    # Compute Short term Prob with respect PS and BS (total)
    short_term = find_short_term_prob_for_psbs(short_term                  = short_term,
                                               negl_prob                   = negl_prob,
                                               lambda_bsps                 = lambda_bsps,
                                               len_discretization_PS1_Mag = len(LongTermInfo['Discretizations']['PS-1_Magnitude']['Val']),
                                               pre_selection               = pre_selection,
                                               cfg                         = Config,
                                               logger                      = logger)

    if(short_term['BS_computed_YN'] == False and short_term['PS_computed_YN'] == False):
        return False

    # COMPUTE INTEGRAL FOR BS POS AND DEPTHS
    if(pre_selection['BS_scenarios'] == True):
        short_term = set_grid_integration(Discretizations  = LongTermInfo['Discretizations'],
                                          pre_selection    = pre_selection,
                                          short_term       = short_term,
                                          cfg              = Config,
                                          logger           = logger)

    # COMPUTE HYPOCENTRAL PROBABILITY DISTRIBUTION FOR BS, IF REQUIRED
    if(short_term['BS_computed_YN'] == True and pre_selection['BS_scenarios'] == True):
        short_term  = get_hypocentral_prob_for_bs(Discretizations  = LongTermInfo['Discretizations'],
                                                  pre_selection    = pre_selection,
                                                  short_term       = short_term,
                                                  ee               = ee)
    else:
        short_term['Total_BS_Scenarios'] = 0

    ##COMPUTE PS BAR PROBABILITY DISTRIBUTION FOR EACH PS MODEL, IF REQUIRED
    if(short_term['PS_computed_YN'] == True and pre_selection['PS_scenarios'] == True):
        short_term  = get_ps_bar_probability(Discretizations  = LongTermInfo['Discretizations'],
                                             Model_weight     = LongTermInfo['Model_Weights'],
                                             pre_selection    = pre_selection,
                                             lambda_bsps      = lambda_bsps,
                                             PSBarInfo        = PSBarInfo,
                                             short_term       = short_term,
                                             ee               = ee)
    else:
        short_term['Total_PS_Scenarios'] = 0

    return short_term


def get_ps_bar_probability(**kwargs):

    Discretizations = kwargs.get('Discretizations', None)
    Model_weight    = kwargs.get('Model_weight', None)
    short_term      = kwargs.get('short_term', None)
    pre_selection   = kwargs.get('pre_selection', None)
    lambda_bsps     = kwargs.get('lambda_bsps', None)
    PSBarInfo       = kwargs.get('PSBarInfo', None)
    ee              = kwargs.get('ee', None)

    # initialize
    ps1_mag = len(Discretizations['PS-1_Magnitude']['Val'])
    ps2_bar = len(Model_weight['PS2_Bar']['Type'])
    sel_mag = len(pre_selection['sel_PS_Mag_idx'][0])

    ee_bar_prob = dict()

    tmpPSScen = np.zeros((ps1_mag,ps2_bar))
    PSmodelYN = np.ones((ps1_mag,ps2_bar))

    for imod in range(ps2_bar):
        for i1 in range(sel_mag):

            imag = pre_selection['sel_PS_Mag_idx'][0][i1]

            if(PSBarInfo['BarPSperModel'][imag][imod]['pos_xx'].size ==0):
                PSmodelYN[imag][imod] = 0
                continue

            tmpPSScen[imag][imod]                  = len(pre_selection['Inside_in_BarPSperModel'][imag][imod]['inside'])
            ee_bar_prob.setdefault(imag, {})[imod] = 2 #np.zeros(tmpPSScen[imag][imod])

            h_correction = correct_PS_horizontal_position(mag=Discretizations['PS-1_Magnitude']['Val'][imag])

            tmpCOV = copy.deepcopy(ee['ee_PosCovMat_2d']) * 1e6
            tmpCOV[0,0] = tmpCOV[0,0] + h_correction**2
            tmpCOV[1,1] = tmpCOV[1,1] + h_correction**2

            tmpMU = copy.deepcopy(ee['PosMean_2d'])

            tmp_bar_prob = []

            for i2 in range(int(tmpPSScen[imag][imod])):
                ibar = pre_selection['Inside_in_BarPSperModel'][imag][imod]['inside'][i2]
                try:
                    ireg    = PSBarInfo['BarPSperModelReg'][imag][imod][ibar]
                    utm_lon = PSBarInfo['BarPSperModel'][imag][imod]['utm_pos_lon'][ibar]
                    utm_lat = PSBarInfo['BarPSperModel'][imag][imod]['utm_pos_lat'][ibar]
                except:
                    ireg    = PSBarInfo['BarPSperModelReg'][imag][imod]
                    utm_lon = PSBarInfo['BarPSperModel'][imag][imod]['utm_pos_lon']
                    utm_lat = PSBarInfo['BarPSperModel'][imag][imod]['utm_pos_lat']
                # Coordinata utm del baricentro
                tmpVAR = np.array([utm_lat, utm_lon])

                tmp = lambda_bsps['regionsPerPS'][ireg-1]

                if(not np.isnan(tmp) and lambda_bsps['lambda_ps_on_ps_tot'][int(tmp)]>0):
                   tmp_norm = NormMultiDvec(x = tmpVAR.transpose(), mu = tmpMU.transpose(), sigma = tmpCOV)
                   tmp_norm = tmp_norm[0]
                else:
                   tmp_norm = 0

                tmp_bar_prob.append(tmp_norm)

            # NORMALIZE OVER BARICENTRES WITH EQUAL MAGNITUDE AND MODEL
            if (np.sum(tmp_bar_prob) != 0):
                tmp_bar_prob = np.array(tmp_bar_prob)
                tmp_bar_prob = tmp_bar_prob / np.sum(tmp_bar_prob)

            short_term['BarProb'][imod][imag] = tmp_bar_prob

    short_term['Total_PS_Scenarios'] = tmpPSScen.sum()

    return short_term


def get_hypocentral_prob_for_bs(**kwargs):

    Discretizations = kwargs.get('Discretizations', None)
    short_term      = kwargs.get('short_term', None)
    pre_selection   = kwargs.get('pre_selection', None)
    ee              = kwargs.get('ee', None)

    # initialize
    bs1_mag = len(Discretizations['BS-1_Magnitude']['Val'])
    pre_bs  = len(pre_selection['BS2_Position_Selection_inn'])

    tmp_BS_scenarios_val = np.zeros((bs1_mag, pre_bs))
    pre_selection['sel_BS_Mag_idx'] = pre_selection['sel_BS_Mag_idx'][0]

    # SET SPATIAL PROBABILITY TO 1, SINCE ALREADY INCLUDED IN 3D INTEGRATION
    short_term['PosProb'] = np.ones((len(pre_selection['sel_BS_Mag_idx']),
                                    len(pre_selection['BS2_Position_Selection_inn'])))

    for i in range(len(pre_selection['sel_BS_Mag_idx'])):
        #get magnitude
        v_mag   = pre_selection['sel_BS_Mag_val'][i]
        i_mag   = pre_selection['sel_BS_Mag_idx'][i]

        # Compute vertical half_width with respect the magnitude
        v_hwidth = correct_BS_vertical_position(mag = v_mag)
        h_hwidth = correct_BS_horizontal_position(mag = v_mag)

        mu      = ee['PosMean_3d']
        co      = copy.deepcopy(ee['PosCovMat_3dm'])

        # Correct  Covariance matrix
        co[0,0] = co[0,0] + h_hwidth**2
        co[1,1] = co[1,1] + h_hwidth**2
        co[2,2] = co[2,2] + v_hwidth**2

        for j in range(len(pre_selection['BS2_Position_Selection_inn'])):
            #get position
            j_pos_inn_idx = pre_selection['BS2_Position_Selection_inn'][j]
            # COUNT NUMBER OF SCENARIOS TO TREAT
            try:
                a = len(Discretizations['BS-3_Depth']['ValVec'][i_mag][j_pos_inn_idx])
            except:
                a = 0
                raise Exception('!!!! Error in data: no depth defined !!!!')

            b = len(Discretizations['BS-4_FocalMechanism']['ID'])
            tmp_BS_scenarios_val[i, j] = a*b
            j_pos_out = np.where(pre_selection['BS2_Position_Selection_out'] == j_pos_inn_idx)

            # SELECT POINTS ABOVE MOHO IN THIS CELL
            j_sel = np.where((short_term['grid_3d'][2,:]  > v_hwidth) &
                             (short_term['dist_3d_idx']  == j_pos_out[0][0]) &
                             (short_term['grid_3d'][2,:] <= -1000 * Discretizations['BS-2_Position']['DepthMoho'][j_pos_inn_idx] + v_hwidth))

            tmp_depth = 1000 * Discretizations['BS-3_Depth']['ValVec'][i_mag][j_pos_inn_idx] + v_hwidth

            b = np.array([])
            for n in range(len(tmp_depth)):
                a = np.linalg.norm(tmp_depth[n] - short_term['grid_3d'][2,j_sel], axis=0)
                b = np.append(b, a, axis=0)

            n = len(tmp_depth)
            m = int(len(b)/n)
            b = b.reshape(n,m)
            # refDepthSel3D_val = b.min(axis=0)
            refDepthSel3D_idx = np.where(b == np.amin(b, axis=0))[0]

            # COMPUTE PROBABILITY WITHOUT INTEGRAL
            tmp_idx = pre_selection['BS2_Position_Selection_inn'][j]
            a = pre_selection['BS2_Position_Selection_inn']
            x = Discretizations['BS-2_Position']['utm_x'][tmp_idx]
            y = Discretizations['BS-2_Position']['utm_y'][tmp_idx]

            tmp_pt     = npm.repmat([x,y], n,1)
            tmp_pt     = np.append(tmp_pt, tmp_depth.reshape(n,1), axis=1)
            tmp_grid3d = np.array([short_term['grid_3d'][0][j_sel],short_term['grid_3d'][1][j_sel],short_term['grid_3d'][2][j_sel]])

            # compute probability for each depth point (depending on the size of the fault) and scenarios
            short_term_prob_points = NormMultiDvec(x = tmp_pt, mu = mu, sigma = co) # line 536
            short_term_prob_temps  = NormMultiDvec(x = tmp_grid3d.transpose(), mu = mu, sigma = co)

            # Store short_term_prob_points
            short_term['DepProbPoints'][i,j] = copy.deepcopy(short_term_prob_points)
            short_term['DepProbTemps'][i,j]  = copy.deepcopy(short_term_prob_temps)

            selection_sum = np.zeros(len(Discretizations['BS-3_Depth']['ValVec'][i_mag][j_pos_inn_idx]))

            for k in range(len(Discretizations['BS-3_Depth']['ValVec'][i_mag][j_pos_inn_idx])):
                selection = np.where(refDepthSel3D_idx == k)
                selection_sum[k] = np.sum(short_term_prob_temps[selection])

            short_term['DepProbScenes'][i,j] = copy.deepcopy(selection_sum)


        keys = [z for z in short_term['DepProbScenes'] if z[0] == i]
        vals = [short_term['DepProbScenes'][x] for x in keys]
        NormFact = np.sum(np.hstack(vals))

        keys = [z for z in short_term['DepProbPoints'] if z[0] == i]
        vals = [short_term['DepProbPoints'][x] for x in keys]
        NormFactPoints = np.sum(np.hstack(vals))

        for p in range(len(pre_selection['BS2_Position_Selection_inn'])):
            short_term['DepProbScenes'][i,p] = short_term['DepProbScenes'][i,p]/NormFact
            short_term['DepProbPoints'][i,p] = short_term['DepProbPoints'][i,p]/NormFactPoints

    short_term['Total_BS_Scenarios'] = tmp_BS_scenarios_val.sum()

    return short_term


def set_grid_integration(**kwargs):

    Discretizations = kwargs.get('Discretizations', None)
    short_term      = kwargs.get('short_term', None)
    pre_selection   = kwargs.get('pre_selection', None)
    Config          = kwargs.get('cfg', None)
    logger          = kwargs.get('logger', None)

    z_to_xyfact = float(Config.get('Settings','Z2XYfact'))
    space_bin   = float(Config.get('Settings','Space_Bin'))
    space_grid  = z_to_xyfact * space_bin
    all_depth   = np.array([])

    minx = min(Discretizations['BS-2_Position']['utm_x'][pre_selection['BS2_Position_Selection_out']])
    maxx = max(Discretizations['BS-2_Position']['utm_x'][pre_selection['BS2_Position_Selection_out']])
    miny = min(Discretizations['BS-2_Position']['utm_y'][pre_selection['BS2_Position_Selection_out']])
    maxy = max(Discretizations['BS-2_Position']['utm_y'][pre_selection['BS2_Position_Selection_out']])

    tmp = Discretizations['BS-3_Depth']['ValVec'][pre_selection['sel_BS_Mag_idx']]
    for i in range(0, np.shape(tmp)[0]):
        all_depth = np.concatenate((all_depth, tmp[i][pre_selection['BS2_Position_Selection_out']]), axis=None)

    all_depth_bs_3 = np.concatenate(all_depth)*1000
    all_depth_moho = np.array(Discretizations['BS-2_Position']['DepthMoho'])[pre_selection['BS2_Position_Selection_out']]* (-1000.0)

    # check if we have BS positions aligned in x or y or both
    if (minx == maxx):
        x_grid = np.array(minx)
    else:
        x_grid = np.arange(minx, maxx, space_grid)
    if (miny == maxy):
        y_grid = np.array(miny)
    else:
        y_grid = np.arange(miny, maxy, space_grid)

    z_grid = np.arange(min(all_depth_bs_3), max(all_depth_moho), space_bin)

    xx_2d, yy_2d = np.meshgrid(x_grid, y_grid, indexing='xy')
    xx_2d = xx_2d.flatten('F')
    yy_2d = yy_2d.flatten('F')
    grid_2d = np.array([xx_2d, yy_2d])

    xx_3d, yy_3d, zz_3d = np.meshgrid(x_grid, y_grid, z_grid, indexing='xy')
    xx_3d = xx_3d.flatten('F')
    yy_3d = yy_3d.flatten('F')
    zz_3d = zz_3d.flatten('F')
    grid_3d = np.array([xx_3d, yy_3d, zz_3d])

    # inizialize
    dist_2d_idx = np.zeros(len(yy_2d))
    dist_2d_val = np.zeros(len(yy_2d))
    dist_3d_idx = np.zeros(len(yy_3d))
    dist_3d_val = np.zeros(len(yy_3d))

    ## Make array with BS-2_Position
    tmp = [Discretizations['BS-2_Position']['utm_y'][pre_selection['BS2_Position_Selection_out']],
           Discretizations['BS-2_Position']['utm_x'][pre_selection['BS2_Position_Selection_out']]]
    tmp = np.array(tmp).transpose()

    # Mapping: get distances and idx
    for i in range(len(yy_2d)):
        a = np.linalg.norm(tmp - np.array([yy_2d[i], xx_2d[i]]), axis=1)
        idx = np.where(a == np.amin(a))
        dist_2d_idx[i] = idx[0][0]
        dist_2d_val[i] = np.amin(a)

    uu = []
    for i in range(len(yy_3d)):
        a = np.linalg.norm(tmp - np.array([yy_3d[i], xx_3d[i]]), axis=1)
        uu.append(a)
        idx = np.where(a == np.amin(a))
        dist_3d_idx[i] = idx[0][0]
        dist_3d_val[i] = np.amin(a)

    short_term['grid_3d'] = grid_3d
    short_term['grid_2d'] = grid_2d
    short_term['dist_2d_idx'] = dist_2d_idx
    short_term['dist_2d_val'] = dist_2d_val
    short_term['dist_3d_idx'] = dist_3d_idx
    short_term['dist_3d_val'] = dist_3d_val

    logger.info(' --> Set grid Integration')

    return short_term

def find_short_term_prob_for_psbs(**kwargs):

    lambda_bsps    = kwargs.get('lambda_bsps', None)
    short_term     = kwargs.get('short_term', None)
    negl_prob      = kwargs.get('negl_prob', None)
    pre_selection  = kwargs.get('pre_selection', None)
    len_PS_Mag     = kwargs.get('len_discretization_PS1_Mag', None)
    Config         = kwargs.get('cfg', None)
    logger         = kwargs.get('logger', None)

    max_BS_mag     = float(Config.get('Settings','Mag_BS_Max'))
    max_PS_mag     = float(Config.get('Settings','Mag_PS_Max'))

    short_term['BS_computed_YN'] = False
    short_term['PS_computed_YN'] = False

    vec_ps = np.ones(len_PS_Mag)  #ones or zeros? should be 1 only for mag>max_mag_BS
    vec_bs = np.zeros(len_PS_Mag)

    if (lambda_bsps['lambda_ps'] != 0.):
        sel_RatioPSonPSTot = np.array(lambda_bsps['lambda_ps_sub']) / lambda_bsps['lambda_ps']
    else:
        sel_RatioPSonPSTot = np.array(lambda_bsps['lambda_ps_sub'])

    pxBS = lambda_bsps['lambda_bs'] / (lambda_bsps['lambda_ps'] + lambda_bsps['lambda_bs'])
    pxPS = 1 - pxBS

    if(pre_selection['BS_scenarios'] == True):
        for i in range(len(pre_selection['sel_BS_Mag_idx'][0])):

            if (pre_selection['sel_BS_Mag_val'][i] <= max_BS_mag):
                vec_bs[pre_selection['sel_BS_Mag_idx'][0][i]] = pxBS

    if(pre_selection['PS_scenarios'] == True):
        for i in range(len(pre_selection['sel_PS_Mag_idx'][0])):

            if (pre_selection['sel_PS_Mag_val'][i] <= max_PS_mag):
                vec_ps[pre_selection['sel_PS_Mag_idx'][0][i]] = pxPS

    short_term['RatioPSonTot'] = vec_ps
    short_term['RatioBSonTot'] = vec_bs
    short_term['sel_RatioPSonPSTot'] = sel_RatioPSonPSTot

    # Check, if probability BS/PS larger than Prob Negligible, then compute PS BS
    tempbs = np.sum(np.multiply(short_term['magnitude_probability'][pre_selection['sel_PS_Mag_idx'][0]], pxBS))
    tempps = np.sum(np.multiply(short_term['magnitude_probability'][pre_selection['sel_PS_Mag_idx'][0]], pxPS))
    if(tempbs > negl_prob):
        short_term['BS_computed_YN'] = True
    if(tempps > negl_prob):
        short_term['PS_computed_YN'] = True

    logger.info(' --> Negligible Probability: %.4f' % negl_prob)
    logger.info(' --> Probability BS = %.4e --> compute BS = %r' % (tempbs, short_term['BS_computed_YN']))
    logger.info(' --> Probability PS = %.4e --> compute PS = %r' % (tempps, short_term['PS_computed_YN']))

    return short_term

def compute_distribution_for_magnitudes(**kwargs):

    Discretizations = kwargs.get('Discretizations', None)
    short_term      = kwargs.get('short_term', None)
    ee              = kwargs.get('event_parameters', None)
    logger          = kwargs.get('logger', None)

    a     = Discretizations['PS-1_Magnitude']['Val'][0:-1]
    b     = Discretizations['PS-1_Magnitude']['Val'][1:]
    c     = np.add(a, b) * 0.5

    lower = np.insert(c, 0, -np.inf)
    upper = np.insert(c, c.size, np.inf)

    lower_probility_norm  = norm.cdf(lower, ee['mag_percentiles']['p50'], ee['MagSigma'])
    upper_probility_norm  = norm.cdf(upper, ee['mag_percentiles']['p50'], ee['MagSigma'])

    short_term['magnitude_probability'] = np.subtract(upper_probility_norm, lower_probility_norm)
    logger.info(' --> Compute magnitude cumulative distribution')

    return short_term
