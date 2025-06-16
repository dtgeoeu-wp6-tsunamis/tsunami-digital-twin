"""
Misfit evaluator  

Based on the GITEWS approach. See the misfit draft for more detail.
---
Note: This is a modified version of M. Bänsch's original misfit_synthetic.py, where I tried to modularize the code substantially for better maintainability and readability.

J. Behrens, 03/2025
"""

#------------------------------------------------------------------------------------------------------------------------------------------------
# Generic modules that are needed 
#------------------------------------------------------------------------------------------------------------------------------------------------
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import time
from math import radians, cos, sin, asin, sqrt

from pyptf.TypeMeasurement_for_misfit import TypeMeasurement 

#------------------------------------------------------------------------------------------------------------------------------------------------
# Function Definitions for modularization
#------------------------------------------------------------------------------------------------------------------------------------------------
def remove_masked(data):
    """
    Function to remove the masked entries from the synthetic data and save the indices that have been removed
    """
    maskedarray_indices = []
    tmp_data = []
    data_len = np.shape(data)[0]
    time_line = np.shape(data)[1]
    for idx in range(data_len):
        if np.ma.is_masked(data[idx]):
            maskedarray_indices.append(idx)
        else: 
            tmp_data.append(data[idx])
    nMask = len(maskedarray_indices)
    cleaned_data = np.zeros((data_len-nMask, time_line))
    for idx in range(data_len-nMask):
        cleaned_data[idx,:] = tmp_data[idx]

    return cleaned_data, maskedarray_indices

#------------------------------------------------------------------------------------------------------------------------------------------------
def get_gauge_synthetic(gauge_data_path, gauge_POI, statistical_misfit):
    """
    Function to read the syntethic gauge data and output them in an array
    """
    ncfile = os.path.join(gauge_data_path, 'grid_A_ts.nc')
    ds = Dataset(ncfile, 'r', format='NETCDF4')
    gauge_time = ds.variables['time'][:]
 
    # Get time and wave height data
    time_data = ds.variables['time'][:]
    tmp_wave_data = ds.variables['eta'][:]  
 
    # get number of gauges (which is equal to number of grid points for the synthetic data if not otherwise specified)
    idx_remove = 0 #23
    if (statistical_misfit is True):
        if (gauge_POI is None):
        # The first data points (23) need to be removed because they are observation and not PTF data
    
            ngauges = ds.dimensions['grid_npoints'].size - idx_remove
            gauge_POI = list(range(ngauges))
    
            tmp2_wave_data = np.transpose(tmp_wave_data[:, idx_remove:ngauges+idx_remove])
    
        # The whole syntethic data has many masked entries that need to be removed
            wave_data, maskedarray_indices = remove_masked(tmp2_wave_data)
        else: 
            ngauges = len(gauge_POI)
            wave_data, maskedarray_indices = remove_masked(np.transpose(tmp_wave_data[:, gauge_POI]))
    else:
        wave_data = np.transpose(tmp_wave_data[:,0:idx_remove])
        if (gauge_POI is None):
            gauge_POI = list(range(idx_remove))
        maskedarray_indices = []

    # Calculate new length for gauges
    ngauges = np.shape(wave_data)[0]

    return ngauges, gauge_POI, time_data, wave_data, maskedarray_indices

#------------------------------------------------------------------------------------------------------------------------------------------------
def get_scenario_waves(N, data_folders, indices):
    """
    Function to read the scenario data and output them in an array
    """
    # Check index size (mainly for synthetic gauge data)
    ncfile = os.path.join(data_folders[0], 'out_ts.nc')
    ds = Dataset(ncfile, 'r', format='NETCDF4')
    max_index = ds.dimensions['grid_npoints'].size
    # print(f"Maximum index for the scenario data is {max_index}.")
    if (indices[-1] > max_index): 
        indices = indices[0:max_index]
  
    wave_data = []
    scenario_min_height = []
    scenario_max_height = []
    for scenario in range(N):
        #if (scenario % 250 == 0):
        # print(f"Fetching scenario {scenario} out of {N} scenarios.")
        ncfile = os.path.join(data_folders[scenario], 'out_ts.nc')
        ds = Dataset(ncfile, 'r', format='NETCDF4')
        scenario_wave_amplitude = ds.variables['eta'][:]
        scenario_min_height.append(ds.variables['min_height'][indices])
        scenario_max_height.append(ds.variables['max_height'][indices])
        scenario_POIs_wave = scenario_wave_amplitude[:, indices]
        wave_data.append(scenario_POIs_wave)
        # print(wave_data)

    scenario_time = ds.variables['time'][:]

    return wave_data, scenario_time, scenario_min_height, scenario_max_height

#------------------------------------------------------------------------------------------------------------------------------------------------
def get_PTF_waveheights(ngauges, gauge_data):
    """
    Function to save the minimum and maximum PTF waveheight.
    """
    minimum_waveheight = np.zeros(ngauges)
    maximum_waveheight = np.zeros(ngauges)
    nan_indices = []
    for gauge in range(ngauges):
        minimum_waveheight[gauge] = np.min(gauge_data[gauge])
        maximum_waveheight[gauge] = np.max(gauge_data[gauge])
        if (np.isnan(maximum_waveheight[gauge]) or np.isnan(minimum_waveheight[gauge])):
            print("Gauge entry is a masked array. Setting min/max waveheight to zero.")
            minimum_waveheight[gauge] = 0.
            maximum_waveheight[gauge] = 0.

    return minimum_waveheight, maximum_waveheight, nan_indices
  
#------------------------------------------------------------------------------------------------------------------------------------------------
def get_PTF_time_indices(ngauges, gauge_times, scenario_time):
    """
    Function to compare gauge times and choose the closest PTF scenario time.
    """
  
    PTF_indices = []
    for gauge in range(ngauges):
        time_array = np.array(gauge_times[gauge])
        sub_indices = np.zeros(len(time_array), dtype=int)
        for index in range(len(time_array)):
            time_value = time_array[index]
            time_diff = np.abs(scenario_time - time_value)
            sub_indices[index] = np.argmin(time_diff)
        PTF_indices.append(sub_indices)

    return PTF_indices

#------------------------------------------------------------------------------------------------------------------------------------------------
def load_scenario_data(ngauges,scenario_data_path, gauge_POI, synthetic_gauge, maskedarray_indices):
    """
    Fetch PTF scenario data
    """

    # Get data folder names
    data_folders = []
    for sim_folder in scenario_data_path:
        if os.path.exists(sim_folder):
            data_folders += [os.path.join(sim_folder, each) for each in sorted(os.listdir(sim_folder)) if os.path.isdir(os.path.join(sim_folder, each)) and not each.startswith('.')]
    N = len(data_folders) #total number of scenarios
    # N = 10 #for testing only

    # Read and save the scenario results at the gauge POIs and time data
    print("Reading scenario data.")
    start = time.time()
    scenario_results, scenario_time, scenario_min_height, scenario_max_height = get_scenario_waves(N, data_folders, gauge_POI)
    stop = time.time()    
    print(f"Reading the data took {stop - start} s.\n")

    # Get time range and set up time arrays for the gauge data
    time_range = [np.min(scenario_time), np.max(scenario_time)]
    N_scenario_time = len(scenario_time)

    # Remove necessary indices if syntethic data is used (?)
    if (synthetic_gauge is True):
        # Determine length of masked array indices
        masked_length = len(maskedarray_indices)
  
        # Initialize new arrays
        tmp_scenario_results = []
        tmp_scenario_min_height = []
        tmp_scenario_max_height = []
        # Delete masked array indices from PTF data
        for scenario in range(N):
            tmp_scenario_min_height.append(np.delete(scenario_min_height[scenario], maskedarray_indices).tolist())
            tmp_scenario_max_height.append(np.delete(scenario_max_height[scenario], maskedarray_indices).tolist())
            for time_idx in range(N_scenario_time):
                tmp_scenario_results.append(np.delete(scenario_results[scenario][time_idx], maskedarray_indices).tolist())
  
        # Overwrite old data
        scenario_results = np.reshape(tmp_scenario_results, (N, N_scenario_time, ngauges))
        scenario_min_height = tmp_scenario_min_height
        scenario_max_height = tmp_scenario_max_height
    
    return scenario_results, scenario_time, scenario_min_height, scenario_max_height, N, time_range

#------------------------------------------------------------------------------------------------------------------------------------------------
def pick_arrival_times(ngauges,N,synthetic_gauge,gauge_data,\
        scenario_time,scenario_results,time_range,arrtime_percentage,scenario_min_height,scenario_max_height):
    """
    Pick arrival times (they have to be used in the time range picking)
    """

    start = time.time()
    # Overwrite gauge times and data so that they match the scenario time range
    if (not synthetic_gauge):
        # for gauge in range(ngauges):
        gauge = 0
        gauge_name = []
        data_list = []
        time_list = []
        for key, inner_dict in gauge_data.items():
            print(f"Key: {key}")
            gauge_name.append(key)
            df = inner_dict['data']
            data_list.append(df['sea_level'].values)
            time_list.append(df['sec'].values)
            data_list[gauge] = data_list[gauge][(time_list[gauge] >= time_range[0]) &
                                                            (time_list[gauge] <= time_range[1])]
            time_list[gauge] = time_list[gauge][(time_list[gauge] >= time_range[0]) &
                                                            (time_list[gauge] <= time_range[1])]
            gauge = gauge + 1
    # Or vice-versa if synthetic data is used
    else:
        N_new_time = len(scenario_time[scenario_time <= np.max(time_list)])
        new_scenario_results = np.zeros((N, N_new_time, ngauges))
        for scenario in range(N):
            for gauge in range(ngauges):
                new_scenario_results[scenario][:, gauge] = scenario_results[scenario][0:N_new_time, gauge]
                scenario_time = scenario_time[scenario_time <= np.max(time_list)]      
        scenario_results = new_scenario_results
        gauge_Ntime = len(time_list)
        time_list = np.ones((ngauges, gauge_Ntime)) * time_list
  
    # Get PTF min/max waveheight (and indices with NaNs for the synthetic data)
    min_waveheight, max_waveheight, nan_indices = get_PTF_waveheights(ngauges, data_list)

    # Arrival times for the gauges
    arrival_times = np.zeros(ngauges)
    for gauge in range(ngauges):
        absmax_waveheight = max(np.abs(min_waveheight[gauge]), np.abs(max_waveheight[gauge]))
        arrtime_trigger = absmax_waveheight * arrtime_percentage
        tmp_data = data_list[gauge][(time_list[gauge] >= time_range[0])]
        arrtime_index = np.argmax(np.array(np.abs(tmp_data)) >= arrtime_trigger)
        arrival_times[gauge] = time_list[gauge][arrtime_index]

    # Arrival times for the scenarios
    scenario_arrival_times = np.zeros((N, ngauges))
    for scenario in range(N):
        for gauge in range(ngauges):
            current_wave = scenario_results[scenario][:,gauge]
            scenario_absmaxwaveheight_calc = np.max(np.abs(current_wave))
            scenario_absmaxwaveheight = max(scenario_absmaxwaveheight_calc,
                                                      max(np.abs(scenario_min_height[scenario][gauge]),
                                                              np.abs(scenario_max_height[scenario][gauge])))
            arrtime_trigger = scenario_absmaxwaveheight * arrtime_percentage
            #arrtime_trigger = scenario_absmaxwaveheight_calc * arrtime_percentage
            arrtime_index = np.argmax(np.array(np.abs(current_wave)) >= arrtime_trigger)
            scenario_arrival_times[scenario, gauge] = scenario_time[arrtime_index]

    stop = time.time()    
    print(f"Calculating the arrival times took {stop - start} s.\n")

    # Cut off gauge times and data that lie outside the arrival time range 
    gauge_cut_times = []
    gauge_cut_data = []
    for gauge in range(ngauges):
        # gauge_tmp_times = time_list[gauge][time_list[gauge] >= arrival_times[gauge]]
        # gauge_tmp_data = data_list[gauge][time_list[gauge] >= arrival_times[gauge]]
        gauge_time_limit = arrival_times[gauge] + 3600.
        print(f"Gauge {gauge_name[gauge]}, arrival time {arrival_times[gauge]}, end time {gauge_time_limit}")  
        gauge_tmp_times = time_list[gauge][(time_list[gauge] >= arrival_times[gauge]) & (time_list[gauge] <= gauge_time_limit)]
        gauge_tmp_data = data_list[gauge][(time_list[gauge] >= arrival_times[gauge]) & (time_list[gauge] <= gauge_time_limit)]  
        gauge_cut_times.append(gauge_tmp_times)
        gauge_cut_data.append(gauge_tmp_data)

    # Get indices which compare the gauge time with the closest PTF scenario time
    gauge_indices = get_PTF_time_indices(ngauges, gauge_cut_times, scenario_time)

    # Cut off scenario data that lies outside the arrival time range 
    scenario_results_cut = []
    for scenario in range(N):
        for gauge in range(ngauges):
            scenario_results_tmp = scenario_results[scenario][scenario_time >= scenario_arrival_times[scenario, gauge], gauge]
            # gauge_time_limit = arrival_times[gauge] + 10000.  
            # scenario_results_tmp = scenario_results[scenario][(scenario_time >= scenario_arrival_times[scenario, gauge], gauge) &
            #                                                   (scenario_time <= gauge_time_limit, gauge)]
            scenario_results_cut.append(scenario_results_tmp)

    # Get maximum index for wave comparison (gauge and scenario data have window that needs to be matched for comparison)
    PTF_maxindex = np.zeros((N, ngauges), dtype=int)
    for gauge in range(ngauges):
        max_index_length = len(gauge_cut_data[gauge])
        gauge_index = gauge_indices[gauge]
        # determine index increment for gauge (is either 1 or 2; gauge_data always has increment 1, so the gauge index will be stored with an increment of 1)
        index_dt = gauge_index[1]-gauge_index[0]  
        len_gauge = len(gauge_index)    
        for scenario in range(N):
            # get length of cut scenario results and check if length has to be changed due to index increment
            len_scenario_old = len(scenario_results_cut[gauge + ngauges*scenario])
            new_scenario_index = np.arange(0, len_scenario_old, index_dt, dtype=int)
            len_scenario = len(new_scenario_index)
    
            # calculate highest possible index = lowest length of timeseries
            min_len = min(len_scenario, max_index_length, len_gauge)
            PTF_maxindex[scenario, gauge] = min_len


    # # Plotting arrival times (and gauge data)
    # for gauge in range(ngauges):
    #     plt.figure()
    #     plt.title(f"Gauge data and arrival time (blue line) for {gauge_name[gauge]}")
    #     plt.xlabel('Time after earthquake = model time [s]')
    #     plt.ylabel('Wave height [m]')
    #     # plt.show()
    #     for scenario in range(N):
    #         max_index = PTF_maxindex[scenario, gauge]
    #         arrtime_index = np.where(scenario_time == scenario_arrival_times[scenario, gauge])
    #         # plt.plot(scenario_time[arrtime_index: arrtime_index + max_index], scenario_results_cut[gauge][0:max_index])
    #         plt.plot(scenario_time, scenario_results[scenario][:,gauge], color = 'grey')
    #     plt.plot(gauge_cut_times[gauge], gauge_cut_data[gauge])
    #     plt.axvline(x = arrival_times[gauge], color = 'b')
    #     plt.xlim(gauge_cut_times[gauge][0] - 60. , gauge_cut_times[gauge][-1])

    #     plt.savefig(gauge_name[gauge] + '.png')
    
    return scenario_arrival_times, arrival_times, gauge_cut_data, scenario_results_cut, PTF_maxindex, min_waveheight, max_waveheight

def gauge_scaling_setup(ngauges,min_waveheight,max_waveheight,time_range,synthetic_gauge,statistical_misfit,plot_arrivaltimes):
    """
    Function for scaling setup
    """

    # First Gauge set up
    # Remove index 0 and 21 from misfit if syntethic data is used (because for both, the closest station is over 1° away). Index 3 and 22 are removed because the data is bad.
    if (synthetic_gauge):
        if (not statistical_misfit):
            min_waveheight[[0,3,4,21,22]] = 0.
            max_waveheight[[0,3,4,21,22]] = 0.

    # Set up GAUGES type 
    # dummy coordinates (because they are not needed for this type of measurement)
    coords_gauges = np.ones(ngauges)
    weigths_gauges = np.ones(ngauges) / ngauges

    range_gauges = [np.min(min_waveheight), np.max(max_waveheight)]
    GAUGES = TypeMeasurement(ngauges, coords_gauges, weigths_gauges, range_gauges, measuretypeweight = 0.15)
    scalmax= range_gauges[1]-range_gauges[0]
    GAUGES.scale_measured_data(scalmax)


    # Now Arrival times setup
    # Set up ARRTIME (= arrival times) type at each gauge (will use the same coordinates)
    # The arrival time will be based on the first occurrence of a certain percentage of the maximum wave height
    weights_arrtime = np.ones(ngauges) / ngauges
    range_arrtime = time_range

    ARRTIME = TypeMeasurement(ngauges, coords_gauges, weights_arrtime, range_arrtime, measuretypeweight = 0.85)
    scalmax = time_range[1]-time_range[0]
    ARRTIME.scale_measured_data(scalmax)

    print('INFO: scaling factors:')
    print('         Arrival time: ',ARRTIME.scaling)
    print('                Gauge: ',GAUGES.scaling)
    # Remove index 0 and 21 from misfit if syntethic data is used (because for both, the closest station is over 1° away). Index 3 and 22 are removed because the data is bad.

    if (synthetic_gauge):
        if (not statistical_misfit):
            weigths_gauges[[0,3,4,21,22]] = 0.
            GAUGES.renormalize_stationweights()

            weights_arrtime[[0,3,4,21,22]] = 0.
            ARRTIME.renormalize_stationweights()
            print('Removing bad gauge data.')

    # # Plotting arrival times (and gauge data)
    # if (plot_arrivaltimes):
    #     for gauge in range(ngauges):
    #         plt.figure()
    #         plt.plot(gauge_times[gauge], gauge_data[gauge])
    #         plt.axvline(x = arrival_times[gauge], color = 'b')
    #         plt.title(f"Gauge data and arrival time (blue line) for {gauge_list[gauge]}")
    #         plt.xlabel('Time after earthquake = model time [s]')
    #         plt.ylabel('Wave height [m]')
    #     plt.show()
   
    # --- return values
    return GAUGES, ARRTIME

#------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_gauge_dist(N,ngauges,PTF_maxindex,scenario_results_cut,gauge_cut_data,arrival_times,GAUGES):
    """
    Calculate distances for gauge data
    """

    print("Calculating gauge distances and norms.")
    start = time.time()

    # This loop structure is faster than calulating the scaled distances/norms in one big loop
    gauges_dist = []
    for gauge in range(ngauges):
        for scenario in range(N):
            max_index = PTF_maxindex[scenario, gauge]
            current_scenario_data = scenario_results_cut[gauge + ngauges*scenario][0:max_index]
            current_gauge_data = gauge_cut_data[gauge][0:max_index]
            # Each entry is a len(indices) array of distances; Results will be normalized
            gauges_dist.append( np.abs(current_gauge_data - current_scenario_data))
  
    gauges_norms = np.zeros((N, ngauges))
    idx = 0
    for gauge in range(ngauges):
        for scenario in range(N):
            current_dist_gauge = gauges_dist[idx]
            gauges_norms[scenario, gauge] = GAUGES.scaling_func(np.linalg.norm(current_dist_gauge,1))
            idx += 1
    
    stop = time.time()    
    print(f"Calculating gauge distances and norms took {stop - start} s.\n")
    
    # --- return values
    return gauges_norms

#------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_arrival_dist(N,ngauges,scenario_arrival_times,arrival_times,ARRTIME):
    """
    Calculate distances for arrival times
    """

    print("Calculating arrival time distances and norms.")
    start = time.time()

    arrtime_dist_scaled = np.zeros((N, ngauges))
    for scenario in range(N):
        for gauge in range(ngauges):
            arrtime_dist = np.abs(scenario_arrival_times[scenario, gauge] - arrival_times[gauge])
            arrtime_dist_scaled[scenario, gauge] = ARRTIME.scaling_func(arrtime_dist)
  
    stop = time.time()    
    print(f"Calculating arrival time distances and norms took {stop - start} s.\n")
    
    # --- return values
    return arrtime_dist_scaled

#------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_misfit(gauges_norms, arrtime_dist_scaled, GAUGES, ARRTIME):
    """
    Calculate misfit
    """

    print("Calculating misfit.")
    start = time.time()

    # Calculate misfit for each type (= sum of the norms and each stationweight)
    GAUGES_misfit = np.dot(gauges_norms, GAUGES.stationweights)

    ARRTIME_misfit = np.dot(arrtime_dist_scaled, ARRTIME.stationweights)

    # Calculate total misfit (sum of misfits for each type and the typeweights)
    MISFIT = 1.-(GAUGES_misfit * GAUGES.measuretypeweight + \
               ARRTIME_misfit * ARRTIME.measuretypeweight)

    stop = time.time()    
    print(f"Calculating the misfit took {stop - start} s.\n")

    return GAUGES_misfit, ARRTIME_misfit, MISFIT

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

def find_closest_pois(sealevel_dict, pois_d, logger):
    """
    Find closest POIs to each sea level stations
    and return a list of indexes 
    """
    logger.info("Finding closest POIs to each gauge")
    
    npois = len(pois_d['pois_coords'])
    gauge_pois = []
    selected_pois_lon = []
    selected_pois_lat = []
    for gauge in sealevel_dict.keys():
        lon_gauge = sealevel_dict[gauge]['coords'][1]
        lat_gauge = sealevel_dict[gauge]['coords'][0]
        dist=np.ones(npois)
        for i in range(npois):
            lon_poi = pois_d['pois_coords'][i,0]
            lat_poi = pois_d['pois_coords'][i,1]
            dist[i] = haversine(lon_gauge, lat_gauge, lon_poi, lat_poi)
        idx = np.argmin(dist)
        
        # logger.info("Gauge %s (%.2f,%.2f): POI %d (%.2f,%.2f)" % (gauge, lon_gauge, lat_gauge, pois_d['pois_index'][idx], pois_d['pois_coords'][idx,0], pois_d['pois_coords'][idx,1]))
        logger.info("Gauge %s (%.2f,%.2f): POI %d (%.2f,%.2f)" % (gauge, lon_gauge, lat_gauge, idx, pois_d['pois_coords'][idx,0], pois_d['pois_coords'][idx,1]))
        
        # gauge_pois.append(pois_d['pois_index'][idx])
        gauge_pois.append(idx)
        selected_pois_lon.append(pois_d['pois_coords'][idx,0])
        selected_pois_lat.append(pois_d['pois_coords'][idx,1])
    # plot_selected_pois(selected_pois_lon, selected_pois_lat, sealevel_dict)

    return gauge_pois

def update_probabilities(probs, misfit_values):
    """
    Update the probabilities based on the misfit values.
    The higher the misfit, the higher the probability.
    """

    # in the global version, only SBS => N_sbs=len(probs)=len(misfit_values) 
    updated_probs = probs * misfit_values
    # Normalize the updated probabilities
    updated_probs = updated_probs / np.sum(updated_probs)
    
    return updated_probs

#------------------------------------------------------------------------------------------------------------------------------------------------
# The main program
#------------------------------------------------------------------------------------------------------------------------------------------------
def main(**kwargs):
    """
    the main program
    """
    workdir = kwargs.get('workdir', None)
    gauge_data = kwargs.get('gauge_data_dict', None)
    sim_folder_BS = kwargs.get('sim_folder_BS', None)
    sim_folder_PS = kwargs.get('sim_folder_PS', None)
    sim_folder_SBS = kwargs.get('sim_folder_SBS', None)
    gauge_pois = kwargs.get('gauge_pois', None)
    synthetic_gauge = kwargs.get('synthetic_gauge', False) 
    synthetic_data_path = kwargs.get('synthetic_data_path', None)
    statistical_misfit = kwargs.get('statistical_misfit', False)
    arrtime_percentage = kwargs.get('arrival_time_percentage', 0.1)
    plot_arrivaltimes = kwargs.get('plot_arrivaltimes', False)

    if gauge_data is None:
        file_gauge_data = os.path.join(workdir, 'data/sealevel_data.npy')
        try:
            gauge_data = np.load(file_gauge_data, allow_pickle=True).item()
        except:
            raise Exception(f"Error reading file: {file_gauge_data}")
    if gauge_pois is None:
        file_gauge_pois = os.path.join(workdir, 'data/sealevel_gauge_pois.npy')
        try:
            gauge_pois = list(np.load(file_gauge_pois))
        except:
            raise Exception(f"Error reading file: {file_gauge_pois}")

    if (synthetic_gauge):
    # Synthetic gauge data TO BE FIXED
        ngauges, gauge_pois, time_list, data_list, maskedarray_indices = get_gauge_synthetic(synthetic_data_path, gauge_pois, statistical_misfit)
    else:
    # Real gauge data
        ngauges = len(gauge_data)
        npois_gauge = len(gauge_pois)
        if  npois_gauge != ngauges:
            raise ValueError('The same number of gauges and associated POIs have to be specified!')
        maskedarray_indices = []

    # read scenario data
    scenario_data_path = [sim_folder_BS, sim_folder_PS, sim_folder_SBS]
    scenario_results,\
        scenario_time,\
        scenario_min_height,\
        scenario_max_height,\
        N,\
        time_range = load_scenario_data(ngauges, scenario_data_path, gauge_pois, synthetic_gauge, maskedarray_indices)

    scenario_arrival_times,\
        arrival_times,\
        gauge_cut_data,\
        scenario_results_cut,\
        PTF_maxindex,\
        min_waveheight,\
        max_waveheight = \
        pick_arrival_times(ngauges,N,synthetic_gauge,\
        gauge_data,scenario_time,scenario_results,\
        time_range,arrtime_percentage,\
        scenario_min_height,scenario_max_height)

    # initialize gauge data
    GAUGES, ARRTIME = gauge_scaling_setup(ngauges,min_waveheight,max_waveheight,time_range,synthetic_gauge,statistical_misfit,plot_arrivaltimes)

    # compute normalized distance for gauge data
    gauges_norms = calculate_gauge_dist(N,ngauges,PTF_maxindex,scenario_results_cut,gauge_cut_data,arrival_times,GAUGES)

    # compute normalized arrival time distances
    arrtime_dist_scaled = calculate_arrival_dist(N,ngauges,scenario_arrival_times,arrival_times,ARRTIME)

    # compute the overall misfit
    GAUGES_misfit, ARRTIME_misfit, MISFIT = calculate_misfit(gauges_norms, arrtime_dist_scaled, GAUGES, ARRTIME)

    # Print misfit
    print("The misfits are:\n")
    print('Arrival time:\n', ARRTIME_misfit)
    print('Gauges:\n', GAUGES_misfit)
    print('Total:\n', MISFIT)
    np.savetxt(os.path.join(workdir, 'arrtime_misfit.txt'), ARRTIME_misfit)
    np.savetxt(os.path.join(workdir, 'gauge_misfit.txt'), GAUGES_misfit)
    np.savetxt(os.path.join(workdir, 'total_misfit.txt'), MISFIT)

    # # Plot misfit
    # fig = plt.figure()
    # fig.subplots_adjust(top=0.88)
    # titlestring = "all gauges"
  
    # fig.suptitle(f"Misfit for the Samos test case with {titlestring}" )
    # plt.subplot(2,2,1)
    # plt.plot(ARRTIME_misfit, '.')
    # plt.title('Arrival time misfit')
    # plt.xlabel('Scenario number')
    # plt.ylabel('Normalized arrival time misfit')

    # plt.subplot(2,2,2)
    # plt.plot(GAUGES_misfit, '.')
    # plt.title('Gauge misfit')
    # plt.xlabel('Scenario number')
    # plt.ylabel('Normalized gauge misfit')

    # plt.subplot(2,2,3)
    # plt.plot(MISFIT, '.')
    # plt.title('Total misfit')
    # plt.xlabel('Scenario number')
    # plt.ylabel('Normalized total misfit')
    # #fig.set_size_inches(18.5, 10.5)
    # plt.tight_layout()
    # plt.show()

def plot_selected_pois(pois_lon, pois_lat, sealevel_dict):
    '''
    CONTROLLO SU MAPPA DELLA SELEZIONE DEI POIS
    '''

    import cartopy
    import matplotlib
    import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')
    # import matplotlib.rcsetup as rcsetup
    # print(rcsetup.all_backends)

    proj = cartopy.crs.PlateCarree()
    #cmap = plt.cm.magma_r
    cmap = plt.cm.jet
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=cartopy.crs.Mercator())
    coastline = cartopy.feature.GSHHSFeature(scale='low', levels=[1])
    #coastline = cartopy.feature.GSHHSFeature(scale='high', levels=[1])
    ax.add_feature(coastline, edgecolor='#000000', facecolor='#cccccc', linewidth=1)
    ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
    ax.add_feature(cartopy.feature.STATES.with_scale('50m'))
    ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'))
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1,
                        color="#ffffff", alpha=0.5, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    ax.plot(pois_lon[:], pois_lat[:], markerfacecolor='#ff0000', marker="o", 
                linewidth=0, markeredgecolor="#000000",
                transform=proj, zorder=10)

    for gauge in sealevel_dict.keys():
        lon_gauge = sealevel_dict[gauge]['coords'][1]
        lat_gauge = sealevel_dict[gauge]['coords'][0]
        ax.plot(lon_gauge, lat_gauge, linewidth=0, marker='^', markersize=14, 
                markerfacecolor='yellow', markeredgecolor='#000000', 
                transform=proj)

    ax.set_xlabel(r'Longitude ($^\circ$)', fontsize=14)
    ax.set_ylabel(r'Latitude ($^\circ$)', fontsize=14)
    # plt.show() #not working
    plt.savefig('check_gauge.png', format='png', dpi=150, bbox_inches='tight')


if __name__=='__main__':
    main()
