import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
# from datetime                  import datetime
import utm
from scipy.stats import norm

from pyptf.ptf_mix_utilities    import NormMultiDvec
from pyptf.ptf_mix_utilities    import get_focal_mechanism
from pyptf.ptf_lambda_bsps_load import load_lambda_BSPS
from pyptf.ptf_ellipsoids       import build_location_ellipsoid_objects
from pyptf.ptf_scaling_laws     import scalinglaw_WC
from pyptf.ptf_scaling_laws     import correct_BS_horizontal_position
from pyptf.ptf_scaling_laws     import correct_BS_vertical_position


def define_ensemble_global(**kwargs):

    #np.set_printoptions(threshold=sys.maxsize)

    workflow_dict = kwargs.get('workflow_dict', None)
    Config = kwargs.get('cfg', None)
    event_parameters = kwargs.get('event_parameters', None)
    logger = kwargs.get('logger', None)
    # POIs              = kwargs.get('POIs', None)

    workdir         = workflow_dict['workdir']
    sigma           = workflow_dict['sigma']
    # negligible_prob = workflow_dict['negligible_prob']
    # samp_mode       = workflow_dict['sampling_mode'] 
    # samp_scen       = workflow_dict['number_of_scenarios'] 
    # samp_type       = workflow_dict['sampling_type'] 

   
    logger.info('Creating discretized ensemble')
    ensemble = dict()

    # discretized magnitude
    ensemble['magnitude'], ensemble['magnitude_idx'] = select_magnitude(event_parameters = event_parameters, 
                                                                        workflow_dict    = workflow_dict,
                                                                        logger           = logger)
                                                              
    # discretized positions in geographic coordinates
    location_ellipse_2d = build_location_ellipsoid_objects(event           = event_parameters,
                                                           cfg             = Config,
                                                           sigma           = sigma,
                                                           seismicity_type = 'BS')

    utm_x, utm_y = discretized_position(ellipse     = location_ellipse_2d,
                                        # zone_number = event_parameters['ee_utm'][2],
                                        # zone_letter = event_parameters['ee_utm'][3],
                                        step        = workflow_dict['position_step'],
                                        logger      = logger)

    # plot ellipse 2d
    # plt.plot(location_ellipse_2d[:,1], location_ellipse_2d[:,0], linewidth=0, marker='o')
    # plt.plot(event_parameters['ee_utm'][0], event_parameters['ee_utm'][1], linewidth=0, marker='*')
    # plt.show()

    ensemble['position_utm_x'] = utm_x
    ensemble['position_utm_y'] = utm_y
    
    # discretized depth in km
    tmp = load_lambda_BSPS(cfg              = Config,
                           sigma            = sigma,
                           event_parameters = event_parameters,
                           logger           = logger)

    ellipsoid_3d = tmp['gaussian_ellipsoid']

    ensemble['depth'] = discretized_depth(z      = -ellipsoid_3d['zp'],
                                          step   = workflow_dict['depth_step'],
                                          logger = logger)

    grid_3d, lon, lat = create_grid3d(event_parameters = event_parameters,
                                      ensemble         = ensemble,
                                      logger           = logger)
    
    # fix for negative longitudes (Chile)
    lon[lon < 0] += 360.
                        
    ensemble['grid_3d'] = grid_3d
    ensemble['position_geo_lon'] = lon
    ensemble['position_geo_lat'] = lat

    # plot z
    # plt.plot(ellipsoid_3d['xp'], -ellipsoid_3d['zp'], linewidth=0, marker='o', color = 'blue')
    # plt.plot(event_parameters['ee_utm'][0], event_parameters['depth']*1000., linewidth=0, marker='*', color = 'red')
    # plot ellipse and grid 3d
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # ax.plot_surface(ellipsoid_3d['xp'],ellipsoid_3d['yp'],-ellipsoid_3d['zp'], alpha = 0.4)
    # # ax.scatter(event_parameters['ee_utm'][0], event_parameters['ee_utm'][1], event_parameters['depth']*1000., linewidth=0, marker='*', color = 'red')
    # ax.scatter(ensemble['position_geo_lon'], ensemble['position_geo_lat'], [grid_3d[i][2] for i in range(grid_3d.shape[0])], linewidth=0, marker='o', color = 'blue')
    # ax.scatter(event_parameters['lon'], event_parameters['lat'], event_parameters['depth']*1000., linewidth=0, marker='*', color = 'red')
    # plt.show()

    # focal_mechanism = get_focal_mechanism(event_parameters = event_parameters,
    #                                       workdir          = workdir,
    #                                       logger           = logger)
    # focal mechanism for Maule 2010 from USGS (https://earthquake.usgs.gov/earthquakes/eventpage/official20100227063411530_30/moment-tensor)
    focal_mechanism = {'np1': {'strike': 178.0, 'dip': 77.0, 'rake': 86.0},
                       'np2': {'strike': 17.0, 'dip': 14.0, 'rake': 108.0}}
    # focal mechanism for Tohoku 2011 from USGS (https://earthquake.usgs.gov/earthquakes/eventpage/official20110311054624120_30/moment-tensor)
    # focal_mechanism = {'np1': {'strike': 193.0, 'dip': 9.0, 'rake': 78.0},
    #                    'np2': {'strike': 25.0, 'dip': 81.0, 'rake': 92.0}}

    ensemble['focal_mechanism'] = discretized_mechanism(focal_mechanism  = focal_mechanism,
                                                        strike_step      = workflow_dict['strike_step'],
                                                        strike_sigma     = workflow_dict['strike_sigma'],
                                                        dip_step         = workflow_dict['dip_step'],
                                                        dip_sigma        = workflow_dict['dip_sigma'],
                                                        rake_step        = workflow_dict['rake_step'],
                                                        rake_sigma       = workflow_dict['rake_sigma'],
                                                        logger           = logger)
    print(ensemble['focal_mechanism'])

    ensemble['fault_size'] = compute_fault_size(magnitude = ensemble['magnitude'],
                                                logger    = logger)

    ensemble['slip'] = compute_slip(magnitude  = ensemble['magnitude'],
                                    fault_size = ensemble['fault_size'],
                                    rigidity   = workflow_dict['rigidity'],
                                    logger     = logger)

    logger.info('--> Computing probabilities')
    ensemble_probs = dict()
    ensemble_probs['magnitude'] = compute_mag_probability(#magnitude          = ensemble['magnitude'],
                                                          magnitude_idx      = ensemble['magnitude_idx'],
                                                          event_parameters   = event_parameters,
                                                          mag_discretization = workflow_dict['magnitude_values'],
                                                          logger             = logger)
  
    ensemble_probs['position'] = compute_pos_probability(event_parameters = event_parameters,
                                                         ensemble         = ensemble,
                                                         logger           = logger)
    
    ensemble_probs['focal_mechanism'] = compute_mech_probability(ensemble = ensemble,  
                                                                 logger   = logger)

    # creation of scenarios parameters list and computing corresponding probabilities
    scen_params, scen_probs = scenarios_parameters_and_probabilities(ensemble_probs = ensemble_probs,
                                                                     ensemble       = ensemble,
                                                                     logger         = logger)

    # ensemble['scenarios_parameters'] = scen_params
    # ensemble_probs['scenarios_probabilities'] = scen_probs

    # save_scenario_lists(workflow_dict           = workflow_dict,
    #                     scenarios_parameters    = ensemble['scenarios_parameters'],
    #                     scenarios_probabilities = ensemble_probs['scenarios_probabilities'])
    # 
    scenarios = dict()
    scenarios['par_scenarios_sbs'] = scen_params
    scenarios['ProbScenSBS'] = scen_probs

    return scenarios


def select_magnitude(**kwargs):
    """
    Selecting magnitude values to create the ensemble.
    If magnitude distribution is provided in json file then the values 
    in the key [mag_values] are used. 
    Otherwise, magnitude ranges are defined from the uncertainty provided 
    in the key parameter [mag_percentiles]
    
    """
    event_parameters = kwargs.get('event_parameters', None)
    workflow_dict = kwargs.get('workflow_dict', None)
    logger = kwargs.get('logger', None)

    logger.info('--> Selecting magnitude values')

    if len(event_parameters['mag_values']) == 0:
        logger.info('    Using percentiles from the json event file [mag_percentiles]')

        magnitude_values = workflow_dict['magnitude_values']
        workdir = workflow_dict['workdir']
        sigma = workflow_dict['sigma']

        min_mag  = event_parameters['mag'] - event_parameters['MagSigma'] * sigma
        max_mag  = event_parameters['mag'] + event_parameters['MagSigma'] * sigma

        if max_mag < magnitude_values[0] or min_mag > magnitude_values[-1]:
            workflow_dict['exit message'] = 'The magnitude of the event is out of the limits. The range is {0}-{1}.'.format(magnitude_values[0], magnitude_values[-1])
            np.save(os.path.join(workdir, workflow_dict['workflow_dictionary']), workflow_dict, allow_pickle=True)
            sys.exit('The magnitude of the event is out of the limits. The range is {0}-{1}.'.format(magnitude_values[0], magnitude_values[-1]))
            
        else:
            magnitudes = magnitude_values[(magnitude_values >= min_mag) & (magnitude_values <= max_mag)]
            idx = np.where((magnitude_values >= min_mag) & (magnitude_values <= max_mag))
            if magnitudes.size == 0:
                idx = np.array((np.abs(magnitude_values-max_mag)).argmin())
                magnitudes = np.array([magnitude_values[idx]])

    else:
        logger.info('    Using magnitude values from the json event file [mag_values]')
        magnitudes = np.array(event_parameters['mag_values'])
        idx = None
      
    logger.info('    Number of magnitude values = ' + str(len(magnitudes)))

    return magnitudes, idx

def discretized_position(**kwargs):

    ellipse = kwargs.get('ellipse', None)
    # zone_number = kwargs.get('zone_number', None)
    # zone_letter = kwargs.get('zone_letter', None)
    step = kwargs.get('step', None)
    logger = kwargs.get('logger', None)

    logger.info('--> Discretizing positions')

    utm_x_min = np.amin(ellipse[:,1])
    utm_x_max = np.amax(ellipse[:,1])
    utm_y_min = np.amin(ellipse[:,0])
    utm_y_max = np.amax(ellipse[:,0])
    utm_x = np.arange(utm_x_min, utm_x_max + step, step)
    utm_y = np.arange(utm_y_min, utm_y_max + step, step)
    
    logger.info('    Number of points along x= ' + str(len(utm_x)))
    logger.info('    Number of points along y= ' + str(len(utm_y)))

    return utm_x, utm_y

def discretized_depth(**kwargs):
    z = kwargs.get('z', None)
    step = kwargs.get('step', None)
    logger = kwargs.get('logger', None)

    logger.info('--> Discretizing depths')

    min_depth = np.amin(z)
    max_depth = np.amax(z)

    depth = np.arange(min_depth, max_depth + step, step)
    logger.info('    Number of points along z= ' + str(len(depth)))
    
    return depth

def create_grid3d(**kwargs): 
    """
    Creation of grid 3d of position in utm (x, y, z) and conversion to geographic coordinates (lon, lat).
    In ensemble['grid_3d'] points are ordered as follows: x1,y1,z1; x1,y2,z1; ...; x2,y1,z1;...
    """
    event_parameters = kwargs.get('event_parameters', None)
    ensemble = kwargs.get('ensemble', None)
    logger = kwargs.get('logger', None)

    logger.info('--> Creating grid 3d')

    zone_number = event_parameters['ee_utm'][2]
    zone_letter = event_parameters['ee_utm'][3]

    position_x = ensemble['position_utm_x']
    position_y = ensemble['position_utm_y']
    depth      = ensemble['depth']

    xx_3d, yy_3d, zz_3d = np.meshgrid(position_x, position_y, depth, indexing='xy')
    xx_3d               = xx_3d.flatten('F')
    yy_3d               = yy_3d.flatten('F')
    zz_3d               = zz_3d.flatten('F')
    grid_3d             = np.array([xx_3d, yy_3d, zz_3d]).transpose()

    easting = copy.deepcopy(grid_3d[:,0])
    northing = copy.deepcopy(grid_3d[:,1])
    geo_coord = np.array(utm.to_latlon(easting, northing, zone_number, zone_letter))
    # geo_coord = np.array(utm.to_latlon(grid_3d[:,0], grid_3d[:,1], zone_number, zone_letter))
    # geo_coord = np.array(utm.to_latlon(grid_3d[:,0], grid_3d[:,1], zone_number, northern=True)) # northern=True if coordinates are exepressed with negative latitude in the south hemisphere
    # print(geo_coord.shape, grid_3d.shape)
    logger.info('    Number of grid points = ' + str(grid_3d.shape[0]))
    
    return grid_3d, geo_coord[1,:], geo_coord[0,:]


def discretized_mechanism(**kwargs):
    fm = kwargs.get('focal_mechanism', None)
    stk_step = kwargs.get('strike_step', None)
    stk_sigma = kwargs.get('strike_sigma', None)
    dip_step = kwargs.get('dip_step', None)
    dip_sigma = kwargs.get('dip_sigma', None)
    rake_step = kwargs.get('rake_step', None)
    rake_sigma = kwargs.get('rake_sigma', None)
    logger = kwargs.get('logger', None)
    
    logger.info('--> Discretizing strike, dip, and rake values')

    #discretizing strike angle
    stk_tmp = fm['np1']['strike']
    stk1 = np.arange(stk_tmp - stk_sigma, stk_tmp + stk_sigma + 1., stk_step)  #+1 is to include the last value
    ind1 = np.argwhere(stk1 < 0)
    stk1[ind1] = 180. - stk1[ind1]
    stk_tmp = fm['np2']['strike']
    stk2 = np.arange(stk_tmp - stk_sigma, stk_tmp + stk_sigma + 1., stk_step)
    ind2 = np.argwhere(stk2 < 0)
    stk2[ind2] = 180. - stk2[ind2]

    #discretizing dip angle
    #TODO BE CAREFUL TO NEGATIVE VALUES OF DIP
    dip_tmp = fm['np1']['dip']
    dip1 = np.arange(dip_tmp - dip_sigma, dip_tmp + dip_sigma + 1., dip_step)
    dip_tmp = fm['np2']['dip']
    dip2 = np.arange(dip_tmp - dip_sigma, dip_tmp + dip_sigma + 1., dip_step)

    #discretizing rake angle
    rake_tmp = fm['np1']['rake']
    rake1 = np.arange(rake_tmp - rake_sigma, rake_tmp + rake_sigma + 1., rake_step)
    rake_tmp = fm['np2']['rake']
    rake2 = np.arange(rake_tmp - rake_sigma, rake_tmp + rake_sigma + 1., rake_step)

    fm1 = np.array(np.meshgrid(stk1, dip1, rake1)).T.reshape(-1,3)
    fm2 = np.array(np.meshgrid(stk2, dip2, rake2)).T.reshape(-1,3)
    focal_mechanism = np.concatenate((fm1, fm2))
    # print(focal_mechanism)
    logger.info('    Number of angle combinations = ' + str(focal_mechanism.shape[0]))

    return focal_mechanism

def compute_fault_size(**kwargs):
    """
    Fault dimensions (length, width and area) are converted from km to m.
    """
    #TODO understand which scaling law to use in global
    magnitude = kwargs.get('magnitude', None)
    logger = kwargs.get('logger', None)

    logger.info('--> Computing fault size')

    length = scalinglaw_WC(mag=magnitude, type_scala='M2L')
    width = scalinglaw_WC(mag=magnitude, type_scala='M2W')
    area = length * width
    fault_size = np.vstack((length*1.e3, width*1.e3, area*1.e6)).T
   
    return fault_size

def compute_slip(**kwargs):
    """
    Calculation of slip from magnitude by scalar seismic moment.

    Scalar seismic moment: M0 = 10**(1.5*(magnitudo+10.7) 
                           Kanamori formula 1977 in dyne⋅cm (10−7 N⋅m)
    Slip on fault        : D(m) = M0(Pa*m3) / area(m2)*mu(Pa)

    """
    magnitude = kwargs.get('magnitude', None)
    fault_size = kwargs.get('fault_size', None)
    mu = kwargs.get('rigidity', None)  # TODO In TSUMAPS 33.0 GPa was used for BS, 30 GPa for PS forse
    logger = kwargs.get('logger', None)

    logger.info('--> Computing slip')

    area = fault_size[:,0]*fault_size[:,1]
    scalar_moment = 10.**(1.5*(magnitude+10.7)) * 1.e-7
    
    return scalar_moment/(area*mu)

def compute_mag_probability(**kwargs):
    """
    """
    # mag = kwargs.get('magnitude', None)
    idx = kwargs.get('magnitude_idx', None)
    event_parameters = kwargs.get('event_parameters', None)
    mag_discretization = kwargs.get('mag_discretization', None)
    logger = kwargs.get('logger', None)

    if len(event_parameters['mag_values']) == 0:
        ev_mag_sigma = event_parameters['MagSigma']
        ev_mag       = event_parameters['mag']

        logger.info('--> Computing magnitude cumulative distribution')

        a = mag_discretization[0:-1]
        b = mag_discretization[1:]
        #a = mag[0:-1]
        #b = mag[1:]
        c = np.add(a, b) * 0.5

        lower = np.insert(c, 0, -np.inf)
        upper = np.insert(c, c.size, np.inf)

        lower_probility_norm  = norm.cdf(lower, ev_mag, ev_mag_sigma)
        upper_probility_norm  = norm.cdf(upper, ev_mag, ev_mag_sigma)

        magnitude_probability = np.subtract(upper_probility_norm, lower_probility_norm)
        #plt.plot(mag_discretization,magnitude_probability)
        #plt.show()

        magnitude_probability = magnitude_probability[idx]


    else:
        mag_counts = np.array(event_parameters['mag_counts'])
        magnitude_probability = mag_counts/np.sum(mag_counts)
    
    return magnitude_probability

def compute_pos_probability(**kwargs):
    """
    """
    event_parameters = kwargs.get('event_parameters', None)
    ensemble = kwargs.get('ensemble', None)
    logger = kwargs.get('logger', None)

    logger.info('--> Computing position probabilities')

    magnitudes = ensemble['magnitude']
    grid_3d    = ensemble['grid_3d']

    mu = event_parameters['PosMean_3d']
    # co = event_parameters['PosCovMat_3dm']

    n_points = grid_3d.shape[0]
    n_mag = len(magnitudes)
    position_probability = np.zeros((n_mag, n_points))
    for imag,mag in enumerate(magnitudes):
        # Compute vertical half_width with respect the magnitude
        v_hwidth = correct_BS_vertical_position(mag = mag)
        h_hwidth = correct_BS_horizontal_position(mag = mag)

        co = copy.deepcopy(event_parameters['PosCovMat_3dm'])
        # Correct  Covariance matrix
        co[0,0] = co[0,0] + h_hwidth**2
        co[1,1] = co[1,1] + h_hwidth**2
        co[2,2] = co[2,2] + v_hwidth**2
        tmp_prob_points = NormMultiDvec(x = grid_3d, mu = mu, sigma = co)
        normfact = np.sum(tmp_prob_points)
        #TODO normalization for each mag (or normalize once outside of the loop?)
        position_probability[imag] = tmp_prob_points / normfact   
   
    return position_probability

def compute_mech_probability(**kwargs):

    ensemble = kwargs.get('ensemble', None)
    logger = kwargs.get('logger', None)

    logger.info('--> Computing focal mechanism probabilities')
    
    # TODO EQUIPROBABILI???
    n_scenarios, _ = ensemble['focal_mechanism'].shape
    mechanism_probabilities = np.ones((n_scenarios))/n_scenarios              

    return mechanism_probabilities

def scenarios_parameters_and_probabilities(**kwargs):
    """
    ordered list of scenarios parameters for t-hysea simulations:
       index, mag, lon, lat, depth, strike, dip, rake, length, area, slip
       nparams: number of parameters (assigned manually as 11)

    """
    ensemble_probs = kwargs.get('ensemble_probs', None)
    ensemble = kwargs.get('ensemble', None)
    logger = kwargs.get('logger', None)

    logger.info('--> Computing the total probability for each scenario')

    # for k, v in ensemble.items():
    #     print(k, v.shape)

    n_mag = len(ensemble['magnitude'])
    n_points = ensemble['grid_3d'].shape[0]
    n_foc_mech = ensemble['focal_mechanism'].shape[0]
    #print(n_mag, n_points, n_foc_mech)
    nscen = n_mag * n_points * n_foc_mech
    logger.info('Total number of scenarios = ' + str(nscen))

    nparams = 12
    scen_params = np.zeros((nscen, nparams))
    scen_probs = np.zeros((nscen,3))
    
    iscen = 0
    for imag in range(n_mag):
        for ipoint in range(n_points):
            for ifoc in range(n_foc_mech):
                scen_params[iscen,0] = iscen + 1
                scen_params[iscen,1] = 9999
                scen_params[iscen,2] = ensemble['magnitude'][imag]
                scen_params[iscen,3] = ensemble['position_geo_lon'][ipoint]
                scen_params[iscen,4] = ensemble['position_geo_lat'][ipoint]
                scen_params[iscen,5] = ensemble['grid_3d'][ipoint,2] / 1.e3    # m --> km
                scen_params[iscen,6] = ensemble['focal_mechanism'][ifoc,0]
                scen_params[iscen,7] = ensemble['focal_mechanism'][ifoc,1]
                scen_params[iscen,8] = ensemble['focal_mechanism'][ifoc,2]
                scen_params[iscen,9] = ensemble['fault_size'][imag,0] / 1.e3   # m --> km
                scen_params[iscen,10] = ensemble['fault_size'][imag,2] / 1.e6  # m2 --> km2
                scen_params[iscen,11] = ensemble['slip'][imag]

                scen_probs[iscen,0] = ensemble_probs['magnitude'][imag]
                scen_probs[iscen,1] = ensemble_probs['position'][imag,ipoint]
                scen_probs[iscen,2] = ensemble_probs['focal_mechanism'][ifoc]

                iscen += 1

    scenario_parameters = scen_params
    scenario_probability_pre_norm = scen_probs.prod(axis=1)

    # print('--> Normalizing scenario probabilities')
    #both prob_mag and prob_pos are normalized, normfact=1, maybe not needed!
    normfact = np.sum(scenario_probability_pre_norm) 
    scenario_probability = scenario_probability_pre_norm / normfact
    
    # print(np.sort(scenario_probability)) 
    
    return scenario_parameters, scenario_probability

# def save_scenario_lists(**kwargs):
#     wd = kwargs.get('workflow_dict', None)
#     scenarios_parameters = kwargs.get('scenarios_parameters', None)
#     scenarios_probabilities = kwargs.get('scenarios_probabilities', None)
#     logger = kwargs.get('logger', None)
# 
#     logger.info('--> Saving scenario list and probabilities')
# 
#     workdir = wd['workdir']
# 
#     fmt = '%d ' + '%f ' * (scenarios_parameters.shape[1] - 1)
#     np.savetxt(os.path.join(workdir, wd['step1_list']), scenarios_parameters, fmt=fmt)
#                
#     np.save(os.path.join(workdir, wd['step1_prob']), scenarios_probabilities)
# 
#     return
# 
