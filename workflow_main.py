#!/usr/bin/env python

# =============== Probabilistic Tsunami Forecasting WORKFLOW =================#

# THIS WORKFLOW VERSION IS REFERRED TO etc etc
# FUTURE VERSIONS etc etc

# REQUIRED INPUTS #
# 1.
# 2.
# 3. Input file workflow_input.json

# USAGE #
# ...

# ============================================================================#

# Import system modules
import os
import sys
import configparser

sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), 'py'))
import run_steps
from ptf_parser import parse_ptf_stdin
from ptf_mix_utilities import create_workflow_dict, update_workflow_dict
from ptf_load_event import load_event_dict
import ptf_rabbit_consume as rabbit_consume

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

args = parse_ptf_stdin()
mode = args.mode                   # event/rabbit
event_format = args.event_format   # event parameter file format 

# reading configuration file
cfg_file = args.cfg
Config = configparser.RawConfigParser()
Config.read(cfg_file)

# creating workflow dictionary
workflow_dict = create_workflow_dict(args   = args,
                                     Config = Config)

workflow_dict = update_workflow_dict(args          = args,
                                     workflow_dict = workflow_dict)

# 
print('============================')
print('========== EVENT ===========')
print('============================')

if (mode == 'event'):
    #  
    if not os.path.exists(workflow_dict['event_file']):
        sys.exit('Event file ' + workflow_dict['event_file'] + ' not found. Exit')

    event_dict = load_event_dict(cfg_file      = cfg_file,
                                 args          = args,
                                 workflow_dict = workflow_dict)

    run_steps.main(cfg_file      = cfg_file,
                   args          = args,
                   workflow_dict = workflow_dict,
                   event_dict    = event_dict)
    

elif (mode == 'rabbit'):
    rabbit_consume.main(cfg_file      = cfg_file,
                        args          = args,
                        workflow_dict = workflow_dict)   

else:
    sys.exit('Accepted values for the argument mode are event [default] or rabbit.')
