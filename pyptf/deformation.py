import ctypes
import csv
import sys
import numpy as np
import pandas as pd

def save_gnss_stations_list(gnss_df):
    '''
    '''
    gnss_stations_file = 'gnss.inp'
    gnss_df.to_csv(gnss_stations_file, header=None, index=None, sep=' ', columns=['SITE', 'Lon', 'Lat'], mode='w')
    return gnss_stations_file

def save_eq_parameters_bs(par_eq):
    '''
    '''
    eq_parameters_file = 'fault.inp'    
    with open(eq_parameters_file, 'w') as f:
        txt = ' '.join(['{:.2f}'.format(par) for par in par_eq])
        f.write(f'# eq params {txt} \n')
        f.write(f'-location {par_eq[3]:.2f} {par_eq[4]:.2f} {par_eq[5]:.2f} -strike  {par_eq[6]:.2f} -dip  {par_eq[7]:.2f} -rake  {par_eq[8]:.2f} -mw  {par_eq[2]:.2f} -ad')

    return eq_parameters_file

def save_eq_parameters_ps(par_eq):
    '''
    '''
    pass

# def create_deformation_d(file):
#     '''
#     '''
#     data_dict = {}
#     reader = csv.reader(file, delimiter='\t')
#     #df = pd.read_csv(file)
#     #print(df)
#     header = next(reader) # Assuming the first row is a header
#     print(header)
#     for i, row in enumerate(reader):
#         # Basic error handling in case of row length mismatch
#         if len(row) != len(header):
#             print(f"Warning: Row {i+2} has an incorrect number of elements. Skipping.")
#             continue
#         data_dict[i] = dict(zip(header, row))
#     return data_dict

def run_ruptgenkd(fileFlt, filePoi):
    '''
    Python wrapper for the ruptGenKd_call function.
    '''
    
    # Load the shared library
    libruptGenKd = ctypes.CDLL('/home/tonini/temp/GNSS_simulated/libruptGenKd.so')
    fileOut = filePoi + '.gps'
    fileFlt_b = fileFlt.encode('utf-8')
    filePoi_b = filePoi.encode('utf-8')

    libruptGenKd.ruptGenKd_call.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libruptGenKd.ruptGenKd_call.restype = ctypes.c_int

    ierr = libruptGenKd.ruptGenKd_call(fileFlt_b, filePoi_b)
    
    if ierr == 0:
        # print("ruptGenKd_call completed successfully!")
        try:
            with open(fileOut, 'r') as f:
                data_df = pd.read_csv(f)
                #data_dict = create_deformation_d(f)
            return ierr, data_df
        except Exception as e:
            print(f"Error reading or processing {fileOut}: {e}")
            return ierr, None
    else:
        print("ruptGenKd_call failed.")
        return ierr, None


def calculate(**kwargs):
    '''
    '''
    gnss_df = kwargs.get('gnss_df', None)
    par_bs = kwargs.get('par_bs', None)
    par_ps = kwargs.get('par_ps', None)
    par_sbs = kwargs.get('par_sbs', None)

    file_gnss_stations = save_gnss_stations_list(gnss_df)
    deformation_d = dict()
    if par_bs.shape[0] > 0:
        for i in range(par_bs.shape[0]):
            file_scenario_params = save_eq_parameters_bs(par_bs[i,:])
            ierr, data_df = run_ruptgenkd(file_scenario_params, file_gnss_stations)
            deformation_d[str(int(par_bs[i,0]))] = data_df
    if par_ps.shape[0] > 0:
        for i in range(par_ps.shape[0]):
            file_scenario_params = save_eq_parameters_ps(par_ps[i,:])
            #TODO
            #ierr, data_df = run_ruptgenkd(file_scenario_params, file_gnss_stations)
            #deformation_d[str(int(par_ps[i,0]))] = data_df
    if par_sbs.shape[0] > 0:
        for i in range(par_sbs.shape[0]):
            file_scenario_params = save_eq_parameters_bs(par_sbs[i,:])
            ierr, data_df = run_ruptgenkd(file_scenario_params, file_gnss_stations)
            deformation_d[str(int(par_sbs[i,0]))] = data_df

    return deformation_d
