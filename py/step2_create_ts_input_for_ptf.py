#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import xarray as xr


def parse_line():
    """
    """
    Description = "Nc - THySEA file handler"
    examples = "Example:\n" + sys.argv[0] + " --ts_path path_to_scenario_folders --depth_file depth.dat"
  
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=Description, epilog=examples)
  
    parser.add_argument("--depth_file", default = None,
                                      help = "Input POIs depth file. Default: None")
    parser.add_argument("--out_path", default = "",
                                help = "Output folder for the .nc file. Default: working folder")
    parser.add_argument("--ptf_measure", default = "ts_p2t_gl",    # possible values: ts_max,ts_min,ts_max_gl,ts_max_off,ts_min_off,ts_max_off_gl,ts_p2t,ts_p2t_gl 
                                help = "Intensity measure type used for ptf. Default: peak to trough + green law (ts_p2t_gl)")
    parser.add_argument("--failed_BS", default = None,
                                      help = "Filename where failed BS simulations are listed. Default: None")
    parser.add_argument("--failed_PS", default = None,
                                      help = "Filename where failed PS simulations are listed. Default: None")
    parser.add_argument("--outfile_BS", default = None,
                                      help = "Output filename for BS. Default: None")
    parser.add_argument("--outfile_PS", default = None,
                                      help = "Output filename for PS. Default: None")
    parser.add_argument("--file_simdir_BS", default = "",
                                help = "Input BS folder containing nc files. Default: None")
    parser.add_argument("--file_simdir_PS", default = "",
                                help = "Input PS folder containing nc files. Default: None")
    args = parser.parse_args()
  
    return args


def read_depth(depthfile):
    """
    """

    if(os.path.isfile(depthfile) == False):
        sys.exit("File {0} not Found".format(depthfile))

    # poi_dep = []
    # with open(depthfile) as f:
    #     for line in f:
    #         tmp = float(line.split()[0])
    #         poi_dep.append(tmp)
    #         
    # return np.array(poi_dep)

    poi_dep = np.load(depthfile)

    return(poi_dep)

def save_ts_out_ptf_tot(scenarios, failed, outname, pois, outdir):
    """
    """

    n_scenarios = len(scenarios)
    n_pois = len(pois)

    ts_measure = np.zeros((n_scenarios, n_pois))

    errfile = os.path.join(outdir, failed)
    ferr = open(errfile,'w')

    for isc, scenario in enumerate(scenarios):

        # tsfile = os.path.join(path, scenario, "out_ts_ptf.nc")
        tsfile = os.path.join(scenario, "out_ts_ptf.nc")
        if (os.path.isfile(tsfile) == False):
            ferr.write(tsfile + ' not found' + '\n')
            ts_measure[isc,:] = -9999
        else:
            nc = xr.open_dataset(tsfile)
            #print(ts_max[isc,:].shape, nc["ts_max"].values.shape)
            ts_measure[isc,:] = nc[ptf_measure].values

    pois = range(n_pois)
    scenarios = range(n_scenarios)
    outfile = os.path.join(outdir, outname)

    ds = xr.Dataset(
            data_vars={ptf_measure: (["scenarios", "pois"], ts_measure)},
            coords={"pois": pois, "scenarios": scenarios},
            attrs={"description": outfile})


    encode = {"zlib": True, "complevel": 9, "dtype": "float32", 
              "_FillValue": False}
    encoding = {var: encode for var in ds.data_vars}
    ds.to_netcdf(outfile, format="NETCDF4", encoding=encoding)

    ferr.close()

######################################################################

args = parse_line()
n_arguments = sys.argv[1:]

if not n_arguments:
    sys.exit("Use -h or --help option for Help")

if(args.file_simdir_BS is not None):
    bs_path = args.file_simdir_BS
else:
    sys.exit("BS file path is missing")
if(args.file_simdir_PS is not None):
    ps_path = args.file_simdir_PS
else:
    sys.exit("PS file path is missing")

if(args.depth_file is not None):
    poi_depth = read_depth(args.depth_file)
else:
    sys.exit("depth file path is missing")

if(args.ptf_measure is not None):
    ptf_measure = args.ptf_measure

if(args.out_path is not None):
    outdir = args.out_path

if(args.failed_BS is not None):
    failed_bs  = args.failed_BS
if(args.failed_PS is not None):
    failed_ps  = args.failed_PS

if(args.outfile_BS is not None):
    outfile_bs  = args.outfile_BS
if(args.outfile_PS is not None):
    outfile_ps  = args.outfile_PS

# if os.path.isdir(bs_path):
#    scenarios_bs = [d for d in sorted(os.listdir(bs_path)) if "BS" in d]
#    save_ts_out_ptf_tot(bs_path, scenarios_bs, failed_bs, outfile_bs, poi_depth, outdir)
if os.path.exists(bs_path):
    with open(bs_path) as f:
        scenarios_bs = [line.rstrip() for line in f]
    save_ts_out_ptf_tot(scenarios_bs, failed_bs, outfile_bs, poi_depth, outdir)

# if os.path.isdir(ps_path):
#     scenarios_ps = [d for d in sorted(os.listdir(ps_path)) if "PS" in d]
#     save_ts_out_ptf_tot(ps_path, scenarios_ps, failed_ps, outfile_ps, poi_depth, outdir)
if os.path.exists(ps_path):
    with open(ps_path) as f:
        scenarios_ps = [line.rstrip() for line in f]
    save_ts_out_ptf_tot(scenarios_ps, failed_ps, outfile_ps, poi_depth, outdir)


