import sys
import utm
import scipy
import ast
import numpy as np
from itertools import chain
# import collections
# from numba import jit

#@jit(nopython=True)
def find_distances_tetra_mesh(**kwargs):

    mesh   = kwargs.get('mesh', None)
    tetra  = kwargs.get('tetra', None)
    buffer = kwargs.get('buffer', None)
    moho   = kwargs.get('moho', None)
    g_moho = kwargs.get('grid_moho', None)

    d = dict()

    dist   = np.zeros(len(tetra))
    m_dist = np.zeros(len(tetra))

    for i in range(len(tetra)):
        dist[i]   = np.amin(np.linalg.norm(mesh - tetra[i], axis=1))
        m_dist[i] = np.amin(np.linalg.norm(g_moho - tetra[i], axis=1))

    # Check if all below moho
    d['tetra_in_moho']        = True #default

    # Minimal distance
    d['distances_mesh_tetra'] = dist
    d['distance_min_value']   = np.amin(dist)
    d['distance_min_idx']     = np.argmin(dist)
    d['moho_d_mesh_tetra']    = m_dist
    d['moho_d_min_value']     = np.amin(m_dist)
    d['moho_d_min_idx']       = np.argmin(m_dist)


    # All distances min than buffer
    d['idx_less_then_buffer'] = np.where(dist <= buffer)[0]
    d['idx_more_then_buffer'] = np.where(dist >  buffer)[0]

    ## Questa parte qui general 'errore'!!!!
    # select all indx below the surface for the ones into the slab (PS)
    tmp_tetra = np.take(tetra[:,2],d['idx_less_then_buffer'])
    d['tmp']  = np.where(tmp_tetra <= 0)[0]
    d['idx_less_then_buffer_effective'] = d['idx_less_then_buffer'][d['tmp']]

    # select all indx below the surface for the ones outside the slab (BS)
    tmp_tetra = np.take(tetra[:,2],d['idx_more_then_buffer'])
    d['tmp']  = np.where(tmp_tetra <= 0)[0]
    d['idx_more_then_buffer_effective'] = d['idx_more_then_buffer'][d['tmp']]

    tmp_tetra = np.take(tetra[:,2],d['idx_more_then_buffer_effective'])
    tmp_moho  = np.take(moho, d['idx_more_then_buffer_effective'])
    d['tmp']  = np.where((tmp_moho -1*tmp_tetra/1000) <=0)
    d['idx_more_then_buffer_effective'] = d['idx_more_then_buffer_effective'][d['tmp']]

    # Check if this is in moho
    if(len(d['idx_more_then_buffer_effective']) == 0):
        d['tetra_in_moho'] = False

    return d


def find_tetra_index_for_ps_and_bs(**kwargs):

    Config         = kwargs.get('cfg', None)
    ee             = kwargs.get('event_parameters', None)
    lambda_bsps    = kwargs.get('lambda_bsps', None)
    mesh           = kwargs.get('mesh', None)
    moho           = kwargs.get('moho', None)
    grid_moho      = kwargs.get('grid_moho', None)
    logger         = kwargs.get('logger', None)

    buffer = float(Config.get('lambda','subd_buffer'))

    tt = np.empty((0,3))

    # here make 1 grid moho in utm
    grid_moho_utm = utm.from_latlon(grid_moho[:,1], grid_moho[:,0], ee['ee_utm'][2])
    grid_moho     = np.column_stack((grid_moho_utm[0].transpose(), grid_moho_utm[1].transpose(), (1000*grid_moho[:,2]).transpose()))

    for keys in mesh:
        # Convert lat lon to utm for the baricenter
        mesh[keys]['bari']['utm'] = utm.from_latlon(mesh[keys]['bari']['lat'], mesh[keys]['bari']['lon'], ee['ee_utm'][2])

        tmp_mesh = np.column_stack((mesh[keys]['bari']['utm'][0].transpose(), \
                                    mesh[keys]['bari']['utm'][1].transpose(), \
                                    mesh[keys]['bari']['depth'].transpose()))
        tt_mesh  = np.concatenate((tt, tmp_mesh))

        mesh[keys]['d_dist'] = find_distances_tetra_mesh(mesh = tt_mesh, tetra = lambda_bsps['tetra_xyz'], buffer = buffer, moho = moho, grid_moho = grid_moho)
        logger.info('     --> Min distance from slab %s %10.3f [km]' % (mesh[keys]['name'], mesh[keys]['d_dist']['distance_min_value']/1000))
        logger.info('         --> Nr of PS tetra with dist.  < %4.1f [km] from slab %s : %d  (effective: %d)' % \
              (buffer/1000, mesh[keys]['name'], len(mesh[keys]['d_dist']['idx_less_then_buffer']), len(mesh[keys]['d_dist']['idx_less_then_buffer_effective'])))
        logger.info('         --> Nr of BS tetra with dist. >= %4.1f [km] from slab %s : %d  (effective: %d)' % \
              (buffer/1000, mesh[keys]['name'], len(mesh[keys]['d_dist']['idx_more_then_buffer']), len(mesh[keys]['d_dist']['idx_more_then_buffer_effective'])))

    return mesh


def compute_ps_bs_gaussians_general(**kwargs):

    vol            = kwargs.get('vol', None)
    ee             = kwargs.get('event_parameters', None)
    mesh           = kwargs.get('mesh', None)
    tetra          = kwargs.get('tetra', None)
    lambda_bsps    = kwargs.get('lambda_bsps', None)
    bar_depth_moho = kwargs.get('bar_depth_moho', None)
    logger         = kwargs.get('logger', None)

    hx             = ee['ee_utm'][0]
    hy             = ee['ee_utm'][1]
    hz             = ee['depth']* (1000.0)
    covariance     = ee['PosCovMat_3dm']
    xyz            = np.array([hx, hy, hz])

    # first merge index
    ps_idx    = []
    bs_idx    = []
    bs_ps_idx = []
    gauss_ps_eff = np.array([])
    gauss_bs_eff = np.array([])

    #distances min
    min_d_mesh = sys.float_info.max
    min_d_moho = sys.float_info.max

    n_tetra, _ = tetra.shape

    for keys in mesh:
        ps_idx.extend((mesh[keys]['d_dist']['idx_less_then_buffer_effective']).tolist())
        #bs_idx.extend((mesh[keys]['d_dist']['idx_more_then_buffer_effective']).tolist())
        # if (mesh[keys]['d_dist']['tetra_in_moho'] == True):
        #     inmoho = True
        if (mesh[keys]['d_dist']['moho_d_min_value'] <= min_d_moho):
            min_d_moho = mesh[keys]['d_dist']['moho_d_min_value']
        if (mesh[keys]['d_dist']['distance_min_value'] <= min_d_mesh):
            min_d_mesh = mesh[keys]['d_dist']['distance_min_value']

    ps_idx = np.array(list(set(ps_idx))).astype(int)
    bs_idx = np.setdiff1d(np.arange(n_tetra,dtype=int),ps_idx)

    tetra_dep = tetra[:,2] / 1000.
    bs_idx_moho = np.where((tetra_dep < 0) & (bar_depth_moho-tetra_dep <= 0))[0]
    bs_idx = np.intersect1d(bs_idx, bs_idx_moho);
    bs_ps_idx = np.concatenate((ps_idx, bs_idx))

    ps_tetra = tetra[ps_idx]
    bs_tetra = tetra[bs_idx]

    # # the ellipsoide do not intersect any available source (e.g. event too deep)
    # if not ps_tetra.size and not bs_tetra.size:
    #     return False
    
    bs_ps_tetra = tetra[bs_ps_idx]

    ps_tetra[:,2]    = ps_tetra[:,2]* -1
    bs_tetra[:,2]    = bs_tetra[:,2]* -1
    bs_ps_tetra[:,2] = bs_ps_tetra[:,2]* -1

    gauss_ps_eff = scipy.stats.multivariate_normal.pdf(ps_tetra, xyz, covariance)
    gauss_bs_eff = scipy.stats.multivariate_normal.pdf(bs_tetra, xyz, covariance)
    gauss_bs_ps_eff = scipy.stats.multivariate_normal.pdf(bs_ps_tetra, xyz, covariance)

    sum_bs_ps = np.sum(np.multiply(gauss_bs_ps_eff,vol[bs_ps_idx]))
    # sum_ps = np.sum(np.multiply(gauss_ps_eff,vol[ps_idx]))
    # sum_bs = np.sum(np.multiply(gauss_bs_eff,vol[bs_idx]))

    lambda_ps = np.sum(np.multiply(gauss_ps_eff,vol[ps_idx])) / sum_bs_ps
    lambda_bs = np.sum(np.multiply(gauss_bs_eff,vol[bs_idx])) / sum_bs_ps

    lambda_bsps['lambda_ps']  = lambda_ps
    lambda_bsps['lambda_bs']  = lambda_bs
    lambda_bsps['gauss_ps']   = gauss_ps_eff
    lambda_bsps['gauss_bs']   = gauss_bs_eff

    logger.info(" --> lambda PS: {:6.4e}  Volume PS: {:10.4e} [m^3]".format(lambda_ps, np.sum(vol[ps_idx])))
    logger.info(" --> lambda BS: {:6.4e}  Volume BS: {:10.4e} [m^3]".format(lambda_bs, np.sum(vol[bs_idx])))
    logger.info(" -->                     Volume BS-PS: {:10.4e} [m^3]".format(np.sum(vol[bs_ps_idx])))

    return lambda_bsps

def compute_ps_bs_gaussians_single_zone(**kwargs):

    vol            = kwargs.get('vol', None)
    ee             = kwargs.get('event_parameters', None)
    mesh           = kwargs.get('mesh', None)
    tetra          = kwargs.get('tetra', None)
    lambda_bsps    = kwargs.get('lambda_bsps', None)
    logger         = kwargs.get('logger', None)
    
    hx             = ee['ee_utm'][0]
    hy             = ee['ee_utm'][1]
    hz             = ee['depth']* (1000.0)
    covariance     = ee['PosCovMat_3dm']
    xyz            = np.array([hx, hy, hz])
    lambda_ps_sub  = []

    # logger.info('............................', lambda_bsps['lambda_mix'])
    # sys.exit()

    # if (lambda_bsps['lambda_mix'] == False):
    if (lambda_bsps['lambda_ps'] == 0):

        lambda_bsps['lambda_ps_sub']       = [0,0,0]
        lambda_bsps['lambda_ps_on_ps_tot'] = [0,0,0] # Fixed for lambda_mix == False (PS == 0)
        return lambda_bsps


    for keys in mesh:

        # ps_first = np.zeros(3)
        # bs_first = np.zeros(3)
        # pb_first = np.zeros(3)

        # first merge index
        ps_idx    = []
        bs_idx    = []
        # bs_ps_idx = []

        # first Compute general PS-BS
        #for keys in mesh:
        ps_idx.extend((mesh[keys]['d_dist']['idx_less_then_buffer_effective']).tolist())
        bs_idx.extend((mesh[keys]['d_dist']['idx_more_then_buffer_effective']).tolist())
        if(len(ps_idx) == 0):
            lambda_ps = 0.0
            lambda_ps_sub.append(lambda_ps)
            logger.info("     --> Single {} lambda PS: {:6.4e} Volume ps: {:10.4e} [m^3]".format(mesh[keys]['name'], lambda_ps, np.sum(vol[ps_idx])))

        else:
            ps_idx    = set(ps_idx)
            ps_idx    = np.array(list(ps_idx))
            ps_tetra    = tetra[ps_idx]
            ps_tetra[:,2]    = ps_tetra[:,2]* -1
            gauss_ps_eff     = scipy.stats.multivariate_normal.pdf(ps_tetra, xyz, covariance)

            sum_ps    = np.sum(np.multiply(gauss_ps_eff,vol[ps_idx]))
            lambda_ps = (np.sum(np.multiply(gauss_ps_eff,vol[ps_idx])) / sum_ps) * lambda_bsps['lambda_ps']
            lambda_ps_sub.append(lambda_ps)

            logger.info("     --> Single {} lambda PS: {:6.4e} Volume ps: {:10.4e} [m^3]".format(mesh[keys]['name'], lambda_ps, np.sum(vol[ps_idx])))

    lambda_bsps['lambda_ps_sub'] = lambda_ps_sub

    # Define LambdsPs for each reagion on total lambda PS
    lambda_bsps['lambda_ps_on_ps_tot'] =  np.array(lambda_bsps['lambda_ps_sub']) / lambda_bsps['lambda_ps']

    return lambda_bsps

def update_lambda_bsps_dict(**kwargs):

    Config          = kwargs.get('cfg', None)
    lambda_bsps     = kwargs.get('lambda_bsps', None)
    Regionalization = kwargs.get('Regionalization', None)

    mesh_zones      = ast.literal_eval(Config.get('lambda','mesh_zones'))

    """
    SettingsLambdaBSPS.regionsPerPS = nan(LongTermInfo.Regionalization.Npoly,1);
    SettingsLambdaBSPS.regionsPerPS([3,24,44,48,49])=1;
    SettingsLambdaBSPS.regionsPerPS([10,16,54])=2;
    SettingsLambdaBSPS.regionsPerPS([27,33,35,36])=3;

    config['lambda']['mesh_zones']   = '{\'0\':\'[3,24,44,48,49]\', \'1\':\'[10,16,54]\', \'2\':\'[27,33,35,36]\'}'
    """
    #logger.info(Regionalization['Npoly'])
    regionsPerPS    = np.empty(Regionalization['Npoly'])
    regionsPerPS[:] = np.NaN

    for key in mesh_zones:
        l = ast.literal_eval(mesh_zones[key])
        regionsPerPS[l] = int(key)

    lambda_bsps['regionsPerPS'] = regionsPerPS

    return lambda_bsps

def separation_lambda_BSPS(**kwargs):

    Config      = kwargs.get('cfg', None)
    ee          = kwargs.get('event_parameters', None)
    LongTerm    = kwargs.get('LongTermInfo', None)
    lambda_bsps = kwargs.get('lambda_bsps', None)
    mesh        = kwargs.get('mesh', None)
    logger      = kwargs.get('logger', None)
    
    moho_ll  = np.column_stack((LongTerm['Discretizations']['BS-2_Position']['Val_x'], LongTerm['Discretizations']['BS-2_Position']['Val_y']))
    tetra_ll = np.column_stack((lambda_bsps['tetra_bar']['lon'], lambda_bsps['tetra_bar']['lat']))

    bar_depth_moho = scipy.interpolate.griddata(moho_ll,
                                                LongTerm['Discretizations']['BS-2_Position']['DepthMoho'],
                                                tetra_ll)

    logger.info(' --> Distance between tetra and slabs:')
    mesh = find_tetra_index_for_ps_and_bs(event_parameters = ee,
                                          lambda_bsps      = lambda_bsps,
                                          mesh             = mesh,
                                          cfg              = Config,
                                          logger           = logger,
                                          moho             = bar_depth_moho,
                                          grid_moho        = LongTerm['Discretizations']['BS-2_Position']['grid_moho'])
    #print(mesh['mesh_0']['name'], mesh['mesh_0']['nodes']['lat'][:2], mesh['mesh_0']['nodes']['lon'][:2])
    #print(mesh['mesh_1']['name'], mesh['mesh_1']['nodes']['lat'][:2], mesh['mesh_1']['nodes']['lon'][:2])
    #print(mesh['mesh_2']['name'], mesh['mesh_2']['nodes']['lat'][:2], mesh['mesh_2']['nodes']['lon'][:2])

    lambda_bsps = compute_ps_bs_gaussians_general(tetra            = lambda_bsps['tetra_xyz'],
                                                  event_parameters = ee,
                                                  lambda_bsps      = lambda_bsps,
                                                  vol              = lambda_bsps['volumes_elements'],
                                                  mesh             = mesh,
                                                  bar_depth_moho   = bar_depth_moho,
                                                  logger           = logger)
    # if not lambda_bsps:
    #     return False

    lambda_bsps = compute_ps_bs_gaussians_single_zone(tetra            = lambda_bsps['tetra_xyz'],
                                                      event_parameters = ee,
                                                      lambda_bsps      = lambda_bsps,
                                                      vol              = lambda_bsps['volumes_elements'],
                                                      mesh             = mesh,
                                                      logger           = logger)

    #Update lambda_bsps zoones
    lambda_bsps = update_lambda_bsps_dict(cfg              = Config,
                                          lambda_bsps      = lambda_bsps,
                                          Regionalization  = LongTerm['Regionalization'])

    return lambda_bsps
