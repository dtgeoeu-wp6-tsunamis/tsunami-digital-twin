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
import ast

from pyptf.pyptf import PyPTF
from pyptf.pyptf_exceptions import PyPTFException

from pyptf.ptf_parser import parse_ptf_stdin
import pyptf.ptf_rabbit_consume as rabbit_consume


# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

args = parse_ptf_stdin()
mode = args.mode                   # event/rabbit
event_format = args.event_format   # event parameter file format 

# instance of PyPTF class 
pyPTF = PyPTF(cfg=args.cfg,
              input_workflow=args.input_workflow, 
              hazard_mode=args.hazard_mode,
              ptf_version=args.ptf_version,
              user_gridname=args.user_gridname,
              mag_sigma_val=args.mag_sigma_val,
              sigma=args.sigma,
              logn_sigma=args.logn_sigma,
              in_memory=ast.literal_eval(args.in_memory),
              type_df=args.type_df,
              percentiles=args.percentiles)
print(pyPTF)

if (mode == 'event'):
    try:
        ptf_results = pyPTF.run_from_file(args.event_file)
    except PyPTFException as e:
        print(str(e))

elif (mode == 'rabbit'):
    try:
        ptf_results = rabbit_consume.main(pyPTF = pyPTF)
    except PyPTFException as e:
        print(str(e))

else:
    sys.exit('Accepted values for the argument mode are event [default] or rabbit.')
