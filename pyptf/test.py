import os
import sys
import json
from pyptf.pyptf import PyPTF
from pyptf.pyptf_exceptions import PyPTFException

# test 
 
pyPTF = PyPTF(cfg="./cfg/ptf.config",
              input_workflow="./cfg/workflow_input.json",
              hazard_mode="lognormal_pl",
              sigma=2,
              in_memory=False,
              type_df='polars',
              save=True,
              percentiles=[2, 16, 50, 84, 98])

print(pyPTF)
#sys.exit()

# event_file = '/nvme01/INPUT4PTF/earlyEst/2023_0206_turkey_stat.json'
event_file = '/nvme01/INPUT4PTF/earlyEst/2022_1109_pesaro_stat.json'

with open (event_file, 'r') as f:
    event_dict = eval(f.read())

try:
    result = pyPTF.run(event_dict)

    hc02 = result.get_hc('p02')
    hc16 = result.get_hc('p16')
    hc50 = result.get_hc('p50')
    hc84 = result.get_hc('p84')
    hc98 = result.get_hc('p98')
    hcmean = result.get_hc('mean')
 
    pois_alert_levels02 = result.get_pois_alert_levels('p02')
    pois_alert_levels16 = result.get_pois_alert_levels('p16')
    pois_alert_levels50 = result.get_pois_alert_levels('p50')
    pois_alert_level84 = result.get_pois_alert_levels('p84')
    pois_alert_levels98 = result.get_pois_alert_levels('p98')
    pois_alert_levelsmean = result.get_pois_alert_levels('mean')
    pois_alert_levelsmatrix = result.get_pois_alert_levels('matrix')

    print (pois_alert_levelsmean)
    #alert_levels = pyPTF.run(event_dict,run_sigma=1.5,run_hazard_mode="lognormal")
except PyPTFException as e:
    print(str(e))

print(pyPTF)

