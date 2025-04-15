import sys
import numpy as np
from operator import itemgetter

from pyptf.ptf_mix_utilities import ray_tracing_method


def pre_selection_of_scenarios(**kwargs):

    sigma        = kwargs.get('sigma', None)
    ee           = kwargs.get('event_parameters', None)
    LongTermInfo = kwargs.get('LongTermInfo', None)
    PSBarInfo    = kwargs.get('PSBarInfo', None)
    ellipses     = kwargs.get('ellipses', None)
    logger       = kwargs.get('logger', None)

    pre_selection = dict()
    pre_selection = pre_selection_magnitudes(sigma            = sigma,
                                             event_parameters = ee,
                                             pre_selection    = pre_selection,
                                             logger           = logger,
                                             PS_mag           = LongTermInfo['Discretizations']['PS-1_Magnitude'],
                                             BS_mag           = LongTermInfo['Discretizations']['BS-1_Magnitude'])

    if (pre_selection['BS_scenarios'] == False and pre_selection['PS_scenarios'] == False):
        logger.info(" --> WARNING: Event magnitude out of the discretization used in PTF. Apply Decision Matrix")
        return False

    if (pre_selection['BS_scenarios'] == True):
        pre_selection = pre_selection_BS2_position(pre_selection  = pre_selection,
                                                   logger         = logger,
                                                   BS2_pos        = LongTermInfo['Discretizations']['BS-2_Position'],
                                                   ellipse_2d_inn = ellipses['location_ellipse_2d_BS_inn'],
                                                   ellipse_2d_out = ellipses['location_ellipse_2d_BS_out'])
    else:
        pre_selection['BS2_Position_Selection_inn'] = np.array([])

    if (pre_selection['PS_scenarios'] == True):
        pre_selection = pre_selection_PS2_position(pre_selection  = pre_selection,
                                                   logger         = logger,
                                                   PS2_pos        = LongTermInfo['Discretizations']['PS-2_PositionArea'],
                                                   ellipse_2d_inn = ellipses['location_ellipse_2d_PS_inn'],
                                                   ellipse_2d_out = ellipses['location_ellipse_2d_PS_out'])

        pre_selection = pre_selection_Bar_PS_Model(pre_selection  = pre_selection,
                                                   BarPSperModel  = PSBarInfo['BarPSperModel'],
                                                   ellipse_2d_inn = ellipses['location_ellipse_2d_PS_inn'])


    return pre_selection


def pre_selection_Bar_PS_Model(**kwargs):
    """
    This function uses a ray tracing method decorated with numba
    """

    pre_selection  = kwargs.get('pre_selection', None)
    BarPSperModel  = kwargs.get('BarPSperModel', None)
    ellipse_2d_inn = kwargs.get('ellipse_2d_inn', None)

    Selected_PS_Mag_idx = pre_selection['sel_PS_Mag_idx'][0]

    test_dict = dict()

    for i1 in range(len(Selected_PS_Mag_idx)):
        imag = Selected_PS_Mag_idx[i1]
        for imod in range(len(BarPSperModel[imag])):
            if('utm_pos_lat' in BarPSperModel[imag][imod]):
                if(BarPSperModel[imag][imod]['utm_pos_lat'].size >=2):
                    points     = zip(BarPSperModel[imag][imod]['utm_pos_lon'], BarPSperModel[imag][imod]['utm_pos_lat'])
                    inside_inn = [ray_tracing_method(point[0], point[1], ellipse_2d_inn) for point in points]
                elif(BarPSperModel[imag][imod]['utm_pos_lat'].size ==1 ):
                    inside_inn = ray_tracing_method(BarPSperModel[imag][imod]['utm_pos_lon'][0], BarPSperModel[imag][imod]['utm_pos_lat'][0], ellipse_2d_inn)
                else:
                    pass

                Inside_in_BarPSperModel = {'inside' : np.where(inside_inn)[0]}
                test_dict.setdefault(imag, {})[imod] = Inside_in_BarPSperModel

    pre_selection['Inside_in_BarPSperModel'] = test_dict

    return pre_selection


def pre_selection_PS2_position(**kwargs):
    """
    This function uses a ray tracing method decorated with cumba
    """

    pre_selection  = kwargs.get('pre_selection', None)
    PS2_pos        = kwargs.get('PS2_pos', None)
    ellipse_2d_inn = kwargs.get('ellipse_2d_inn', None)
    ellipse_2d_out = kwargs.get('ellipse_2d_out', None)
    logger         = kwargs.get('logger', None)

    # https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    points     = zip(PS2_pos['utm_y'], PS2_pos['utm_x'])
    inside_inn = [ray_tracing_method(point[0], point[1], ellipse_2d_inn) for point in points]

    points     = zip(PS2_pos['utm_y'], PS2_pos['utm_x'])
    inside_out = [ray_tracing_method(point[0], point[1], ellipse_2d_out) for point in points]

    # Map common indices
    bool_array       = np.in1d(np.where(inside_out)[0], np.where(inside_inn)[0])
    common_positions = np.where(bool_array)[0]

    # fill dictionary
    pre_selection['PS2_Position_Selection_inn']    = np.where(inside_inn)[0]
    pre_selection['PS2_Position_Selection_out']    = np.where(inside_out)[0]
    pre_selection['PS2_Position_Selection_common'] = np.take(pre_selection['PS2_Position_Selection_out'],common_positions)

    logger.info(" --> PS2_Position inner: {:4d} positions found".format(len(pre_selection['PS2_Position_Selection_inn'])))
    logger.info(" --> PS2_Position outer: {:4d} positions found".format(len(pre_selection['PS2_Position_Selection_out'])))

    if len(pre_selection['PS2_Position_Selection_inn']) == 0:
        pre_selection['PS_scenarios'] = False
        logger.info('Warning: PS are excluded since no scenarios are found in the inner positions (within sigma_inn)')

    return pre_selection


def pre_selection_BS2_position(**kwargs):
    """
    This function uses a ray tracing method decorated with cumba
    """

    pre_selection  = kwargs.get('pre_selection', None)
    BS2_pos        = kwargs.get('BS2_pos', None)
    ellipse_2d_inn = kwargs.get('ellipse_2d_inn', None)
    ellipse_2d_out = kwargs.get('ellipse_2d_out', None)
    logger         = kwargs.get('logger', None)

    # https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    points     = zip(BS2_pos['utm_y'], BS2_pos['utm_x'])
    inside_inn = [ray_tracing_method(point[0], point[1], ellipse_2d_inn) for point in points]

    points     = zip(BS2_pos['utm_y'], BS2_pos['utm_x'])
    inside_out = [ray_tracing_method(point[0], point[1], ellipse_2d_out) for point in points]

    # Map common indices
    bool_array       = np.in1d(np.where(inside_out)[0], np.where(inside_inn)[0])
    common_positions = np.where(bool_array)[0]

    # fill dictionary
    pre_selection['BS2_Position_Selection_inn']    = np.where(inside_inn)[0]
    pre_selection['BS2_Position_Selection_out']    = np.where(inside_out)[0]
    pre_selection['BS2_Position_Selection_common'] = np.take(pre_selection['BS2_Position_Selection_out'],common_positions)

    logger.info(" --> BS2_Position inner: {:4d} positions found".format(len(pre_selection['BS2_Position_Selection_inn'])))
    logger.info(" --> BS2_Position outer: {:4d} positions found".format(len(pre_selection['BS2_Position_Selection_out'])))

    # if len(pre_selection['BS2_Position_Selection_inn']) == 0 and len(pre_selection['BS2_Position_Selection_out']) != 0:
    if len(pre_selection['BS2_Position_Selection_inn']) == 0:
        pre_selection['BS_scenarios'] = False
        logger.info('Warning: BS are excluded since no scenarios are found in the inner positions (within sigma_inn)')

    return pre_selection

def pre_selection_magnitudes(**kwargs):

    sigma         = kwargs.get('sigma', None)
    ee            = kwargs.get('event_parameters', None)
    PS_mag        = kwargs.get('PS_mag', None)
    BS_mag        = kwargs.get('BS_mag', None)
    pre_selection = kwargs.get('pre_selection', None)
    logger        = kwargs.get('logger', None)

    val_PS = np.array(PS_mag['Val'])
    ID_PS  = list(PS_mag['ID'])
    val_BS = np.array(BS_mag['Val'])
    ID_BS  = list(BS_mag['ID'])

    # Magnitude range given by sigma
    min_mag = ee['mag'] - ee['MagSigma'] * sigma
    max_mag = ee['mag'] + ee['MagSigma'] * sigma

    # PS
    if(max_mag <= val_PS[0]):
        pre_selection['PS_scenarios'] = False
        sel_PS_Mag_val = np.array([])
        sel_PS_Mag_idx = (np.array([]),)
        sel_PS_Mag_IDs = []

    elif(min_mag >= val_PS[-1]):
        pre_selection['PS_scenarios'] = True
        sel_PS_Mag_val = np.array([val_PS[-1]])
        sel_PS_Mag_idx = (np.where(val_PS[-1]),)
        sel_PS_Mag_IDs = [ID_PS[-1]]

    else:
        pre_selection['PS_scenarios'] = True
        sel_PS_Mag_val = val_PS[(val_PS >= min_mag) & (val_PS <= max_mag)]
        sel_PS_Mag_idx = np.where((val_PS >= min_mag) & (val_PS <= max_mag))
        # To fix if mag uncertainty is too small for val_PS element intervals
        # Find closest magnitude
        if(len(sel_PS_Mag_idx[0]) == 0):
            idx = np.array((np.abs(val_PS-max_mag)).argmin())
            sel_PS_Mag_val = np.array([val_PS[idx]])
            sel_PS_Mag_idx = (np.array[idx],)
            sel_PS_Mag_IDs = list(itemgetter(idx)(ID_PS))
        else:
            sel_PS_Mag_IDs = list(itemgetter(*sel_PS_Mag_idx[0])(ID_PS))

    # BS
    if(max_mag <= val_BS[0]):
        pre_selection['BS_scenarios'] = False
        sel_BS_Mag_val = np.array([])
        sel_BS_Mag_idx = (np.array([]),)
        sel_BS_Mag_IDs = []

    elif(min_mag >= val_BS[-1]):
        # sel_BS_Mag_val = np.array([val_BS[-1]])
        pre_selection['BS_scenarios'] = False
        sel_BS_Mag_val = np.array([])
        sel_BS_Mag_idx = (np.array([]),)
        sel_BS_Mag_IDs = []

    else:
        pre_selection['BS_scenarios'] = True
        sel_BS_Mag_val = val_BS[(val_BS >= min_mag) & (val_BS <= max_mag)]
        sel_BS_Mag_idx = np.where((val_BS >= min_mag) & (val_BS <= max_mag))
        # To fix if mag uncertainty is too small for val_BS element intervals
        # Find closest magnitude
        if(len(sel_BS_Mag_idx[0]) == 0):
            idx = np.array((np.abs(val_BS-max_mag)).argmin())
            sel_BS_Mag_val = np.array([val_BS[idx]])
            sel_BS_Mag_idx = (np.array[idx],)
            sel_BS_Mag_IDs = list(itemgetter(idx)(ID_BS))
        else:
            sel_BS_Mag_IDs = list(itemgetter(*sel_BS_Mag_idx[0])(ID_BS))

    pre_selection['sel_PS_Mag_val'] = sel_PS_Mag_val
    pre_selection['sel_PS_Mag_idx'] = sel_PS_Mag_idx
    pre_selection['sel_PS_Mag_IDs'] = sel_PS_Mag_IDs
    pre_selection['sel_BS_Mag_val'] = sel_BS_Mag_val
    pre_selection['sel_BS_Mag_idx'] = sel_BS_Mag_idx
    pre_selection['sel_BS_Mag_IDs'] = sel_BS_Mag_IDs

    logger.info(" --> BS magnitude values: {}".format(sel_BS_Mag_val))
    logger.info(" --> PS magnitude values: {}".format(sel_PS_Mag_val))

    return pre_selection
