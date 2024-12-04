#!/usr/bin/env python

# Import system modules
import os
import sys
import shutil
import time
import math
import subprocess as sp
from pyutil import filereplace

from files_utils import force_symlink
from ptf_mix_utilities import get_info_from_scenario_list
# from step2_create_ts_input_for_ptf_compss import step2_postproc
import t_hysea as hysea
#import step2_remote_connection as remote

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_barrier
    from pycompss.api.api import compss_wait_on
    from pycompss.api.api import compss_wait_on_directory
    from pycompss.api.parameter import *
    from pycompss.api.container import container
    from pycompss.api.constraint import constraint
    from pycompss.api.binary import binary
    from pycompss.api.mpi import mpi
    from pycompss.api.prolog import prolog
except ImportError:
    from dummy.task import task
    from dummy.api import compss_wait_on
    from dummy.api import compss_barrier
#    from dummy.api import container
#    from dummy.api import constraint
#    from dummy.api import binary
#    from dummy.api import mpi

IMAGENAME = os.environ["IMAGENAME"]
NGPUS_PER_NODE = os.environ["COMPUTING_UNITS"]

# """
@constraint(processors=[{'processorType':'CPU', 'computingUnits':'1'},{'processorType':'GPU', 'computingUnits':"${COMPUTING_UNITS}"}])
@container(engine="SINGULARITY", image=IMAGENAME, options='--nv --no-home')
@prolog(binary="chmod", args="744 {{chunkdir}}/step2_run_inside_container.sh")
@binary(binary=f"./step2_run_inside_container.sh",working_dir="{{chunkdir}}") #(binary='/Step2_BS/run-inside-container.sh')
@task(chunkdir=DIRECTORY_INOUT, datadir=DIRECTORY_IN)
def runHySEA(chunkdir, datadir):
    pass
# """

def create_setup(**kwargs):
    wd = kwargs.get('workflow_dict', None)

    workdir = wd['workdir']
    inpdir = wd['inpdir']
    sim_domain = wd['sim_domain']
    domain = wd['domain']
    eventID = wd['uniqueID']
    hpc_cluster = wd['hpc_cluster']
    UCmode = wd['UCmode']
    pydir = os.path.join(wd['wf_path'], 'py')

    if sim_domain == 'regional':
        poifile = os.path.join(workdir, wd['pois'])
        bathyfile = os.path.join(inpdir, domain, wd['regional_bathy_file'])
        depthfile = os.path.join(inpdir, domain, wd['regional_pois_depth'])
    else:  # local (TODO: add automatic procedure for cutting the grid and selecting the POIs)
        poifile = os.path.join(inpdir, domain, 'POIs_local_domain_' + eventID + '.npy')
        bathyfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_' + eventID + '.grd')
        depthfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_' + eventID + '_POIs_depth.npy')
        # !!only fos SAMOS!!
        poifile = os.path.join(inpdir, domain, 'POIs_local_domain_2020_1030_samos.npy')
        bathyfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_2020_1030_samos.grd')
        depthfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_2020_1030_samos_POIs_depth.npy')

    hysea.save_pois_for_hysea(poifile = poifile,
                              workdir = workdir,
                              outfile = wd['step2_ts_file'])
    
    force_symlink(bathyfile, os.path.join(workdir, wd['bathy_filename']))
    force_symlink(depthfile, os.path.join(workdir, wd['depth_filename']))

    print('--------------')
    print('Preparing submission scripts for ' + hpc_cluster)

    cp = shutil.copy(wd['BS_parfile_tmp'], workdir)
    cp = shutil.copy(wd['PS_parfile_tmp'], workdir)
    cp = shutil.copy(wd['sim_postproc'], workdir)
    cp = shutil.copy(wd['final_postproc'], workdir)

    runtmp = os.path.join(workdir, wd['run_sim_filename'])
    postproc = os.path.join(workdir, wd['run_post_filename'])
    cp = shutil.copy(wd['run_sim_tmp'], runtmp) 
    cp = shutil.copy(wd['run_post_tmp'], postproc) 

    if hpc_cluster == 'leonardo':  # leonardo @CINECA#

        ngpus = 4  # 4 GPUs/node on leonardo @CINECA

        if UCmode:   # !to check
            cp = shutil.copy('sh/Step2_launch_simulARRAY_leonardo.sh', os.path.join(workdir, 'Step2_launch_simul.sh'))
            cp = shutil.copy('sh/Step2_runARRAY_tmp@leonardo.sh', runtmp) 
        filereplace(runtmp, 'leonardoACC', wd['account'])
        filereplace(runtmp, 'leonardoPART', wd['partition'])
        filereplace(runtmp, 'leonardoQOS', wd['quality'])
        filereplace(postproc, 'leonardoACC', wd['account'])
        filereplace(postproc, 'leonardoPART', wd['partition'])
        filereplace(postproc, 'leonardoQOS', wd['quality'])

    elif hpc_cluster == 'mercalli':  #Mercalli @INGV#

        ngpus = 8  # 8 GPUs/node on mercalli @INGV

    elif hpc_cluster != 'leonardo' and hpc_cluster != 'mercalli':

        print('HPC cluster for simulations not properly defined')
        sys.exit('Nothing to do....Exiting')

    filereplace(runtmp, 'LOADENV', wd['envfile'])
    filereplace(runtmp, 'PYDIR', pydir)
    # filereplace(runtmp, 'WDIR', workdir)
    filereplace(postproc, 'LOADENV', wd['envfile'])

    return  ngpus

def presimulations_setup(**kwargs):
    """
    Set up folders and files before running simulations.
    This setup is specific for running T-HySEA simulations
    in a singularity container using compss.
    """
    wd = kwargs.get('wd', None)
    nsce = kwargs.get('nsce', None)
    njobs = kwargs.get('njobs', None)
    ngpus = kwargs.get('ngpus', None)
    seistype = kwargs.get('seistype', None)

    workdir = wd['workdir']
    inpdir = wd['inpdir']
    simdir = wd['step2_dir_sim_' + seistype]
    nd = wd['n_digits']
    BS_parfile_tmp = wd['BS_parfile_tmp']
    PS_parfile_tmp = wd['PS_parfile_tmp']
    scenlist = wd['step1_list_' + seistype]
    sim_domain = wd['sim_domain']
    domain = wd['domain']
    eventID = wd['uniqueID']
    propagation = wd['propagation']
    PS_inicond = wd['ps_inicond_med']

    if not os.path.exists(simdir):
        os.mkdir(simdir)

    datadir_sim = os.path.join(workdir,'datadir_sim')  # TODO: put datadir_sim folder name in config
    if not os.path.exists(datadir_sim):
        os.mkdir(datadir_sim)

    if sim_domain == 'regional':
        poifile = os.path.join(workdir, wd['pois'])
        bathyfile = os.path.join(inpdir, domain, wd['regional_bathy_file'])
        depthfile = os.path.join(inpdir, domain, wd['regional_pois_depth'])
    else:  # local (TODO: add automatic procedure for cutting the grid and selecting the POIs)
        poifile = os.path.join(inpdir, domain, 'POIs_local_domain_' + eventID + '.npy')
        bathyfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_' + eventID + '.grd')
        depthfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_' + eventID + '_POIs_depth.npy')
        # !!only fos SAMOS!!
        poifile = os.path.join(inpdir, domain, 'POIs_local_domain_2020_1030_samos.npy')
        bathyfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_2020_1030_samos.grd')
        depthfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_2020_1030_samos_POIs_depth.npy')
    
    # Copy POI, bathy, and depth file to data directory that will be mounted in the container
    hysea.save_pois_for_hysea(poifile = poifile,
                              workdir = datadir_sim,
                              outfile = wd['step2_ts_file'])
    bathyname = os.path.join(datadir_sim,'step2_bathygrid.grd')
    cp = shutil.copy(bathyfile, bathyname)
    depthname = os.path.join(workdir,'step2_pois_depth.npy')
    cp = shutil.copy(depthfile, depthname)

    # Copy template parfile for BS and PS simulations to workdir
    cp = shutil.copy(BS_parfile_tmp, workdir)
    cp = shutil.copy(PS_parfile_tmp, workdir)
   
    run_file_tmp = "/leonardo/home/userexternal/mvolpe00/tsunami-digital-twin/sh/step2_run_inside_container.sh" #TODO: put in config
    cp = shutil.copy(run_file_tmp, workdir)
    runtmp = os.path.join(workdir, 'step2_run_inside_container.sh')

    simtime = str(float(propagation)*3600+2)

    if seistype == 'BS':
        with open(scenlist, 'r') as f:
            lines = f.readlines()
            scenum = [x.split( )[0] for x in lines]
            eqlon = [x.split( )[3] for x in lines]
            eqlat = [x.split( )[4] for x in lines]
            eqtop = [x.split( )[5] for x in lines]
            eqstk = [x.split( )[6] for x in lines] 
            eqdip = [x.split( )[7] for x in lines]
            eqrak = [x.split( )[8] for x in lines]
            eqlen = [x.split( )[9] for x in lines]
            eqarea = [x.split( )[10] for x in lines]
            eqslip = [x.split( )[11] for x in lines]

    elif seistype == 'PS':
        with open(scenlist, 'r') as f:
            lines = f.readlines()
            scenum = [x.split( )[0] for x in lines]
            inicond = [x.split()[2] for x in lines]

    for n in range(2): #range(njobs):
        chunkdir = os.path.join(simdir, 'chunk' + str(n+1))
        if not os.path.exists(chunkdir):
            os.mkdir(chunkdir)

        kk = n + 1
        s1 = (n * ngpus) + 1
        s2 = kk * ngpus
        if kk == njobs:
            s2 = nsce
        nn = (s2 - s1) + 1

        cp = shutil.copy(runtmp, chunkdir)
        runfile = os.path.join(chunkdir, 'step2_run_inside_container.sh')
        filereplace(runfile,'NUMPROC', str(nn))

        fname = os.path.join(chunkdir,'problems.txt')
        f = open(fname,'w')

        for i in range(s1-1,s2):
            scen = seistype + '_Scenario' + str(scenum[i]).zfill(nd) 
            scendir = os.path.join(chunkdir, scen)
            parfile_text = os.path.join(scen, 'parfile.txt')
            if not os.path.exists(scendir):
                os.mkdir(scendir)
            f.writelines(parfile_text)
            f.writelines('\n')

            if seistype == 'BS':
                tmpfile = os.path.join(workdir, os.path.basename(BS_parfile_tmp).strip())
                parfile = os.path.join(scendir, 'parfile.txt')
                cp = shutil.copy(tmpfile, parfile)

                eqwid = str(float(eqarea[i])/float(eqlen[i]))
                if float(eqtop[i]) == 1:
                   eqtop[i] = '0'    #correction for okada
                eqdep = str(float(eqtop[i])+float(eqwid)/2.*math.sin(math.radians(float(eqdip[i]))))

                filereplace(parfile,'eqlon', eqlon[i])
                filereplace(parfile,'eqlat', eqlat[i])
                filereplace(parfile,'eqdep', eqdep)
                filereplace(parfile,'eqlen', eqlen[i])
                filereplace(parfile,'eqwid', eqwid)
                filereplace(parfile,'eqstk', eqstk[i])
                filereplace(parfile,'eqdip', eqdip[i])
                filereplace(parfile,'eqrak', eqrak[i])
                filereplace(parfile,'eqslip', eqslip[i])
               
            elif seistype == 'PS':
                tmpfile = os.path.join(workdir, os.path.basename(PS_parfile_tmp).strip())
                parfile = os.path.join(scendir, 'parfile.txt')
                cp = shutil.copy(tmpfile, parfile)

                inifile = os.path.join(workdir, PS_inicond.strip(), inicond[i])
                with open(inifile,'r') as fini:
                    trilist = fini.readlines()
                    numtri = len(trilist)
                trilist = [ '0 ' + item for item in trilist]

                tmpfile = open(parfile, 'r').readlines()
                outfile = open(parfile, 'w')
                for line in tmpfile:
                    outfile.write(line)
                    if 'ntri' in line:
                        for item in trilist:
                            outfile.write(item)
                outfile.close()
                filereplace(parfile,'ntri', str(numtri))

            #filereplace(parfile,'step2_bathygrid.grd', '../step2_bathygrid.grd')
            #filereplace(parfile,'step2_ts.dat', '../step2_ts.dat')
            filereplace(parfile,'simfolder/idscen', scen)
            filereplace(parfile,'simtime', simtime) 
         
        f.close()


def main(**kwargs):
    # Command args
    wd = kwargs.get('workflow_dict', None)
    inpfileBS = kwargs.get('file_simBS', None)
    inpfilePS = kwargs.get('file_simPS', None)

    wf_path = wd['wf_path']
    workdir = wd['workdir']
    inpdir = wd['inpdir']
    domain = wd['domain']
    hpc_cluster = wd['hpc_cluster']
    local_host = wd['local_host']
    UCmode = wd['UCmode']
    propagation = wd['propagation']
    sim_domain = wd['sim_domain']
    ptf_measure_type = wd['ptf_measure_type']
    
    # inpfileBS = os.path.join(workdir, wd['step1_list_BS'])
    # inpfilePS = os.path.join(workdir, wd['step1_list_PS'])

    print('Creating simulation setup')
    # ngpus = create_setup(workflow_dict = wd)
    ngpus = int(NGPUS_PER_NODE) #4 

    if hpc_cluster == local_host:

        print('...going to run T-HySea')
        print('--------------')
        os.chdir(workdir)
        
        if os.path.getsize(inpfileBS) > 0:   #BS
            nsce_bs, first_line_bs = get_info_from_scenario_list(filename = inpfileBS,
                                                                 seistype = 'BS')

            njobs = int(nsce_bs / ngpus) + (nsce_bs % ngpus > 0)
            print('Number of jobs = ' + str(njobs))

            presimulations_setup(wd       = wd,
                                 nsce     = nsce_bs,
                                 njobs    = njobs,
                                 ngpus    = ngpus,
                                 seistype = 'BS')
            
            simdir = os.path.join(workdir,wd['step2_dir_sim_BS'])
            datadir = os.path.join(workdir,'datadir_sim')
            # loop over number of chunks
            for i in range(2): #range(njobs)
                chunkdir = os.path.join(simdir, 'chunk' + str(i+1))
                runHySEA(chunkdir, datadir)
                compss_wait_on_directory(chunkdir)
                # TODO: postprocessing of each simualtion output
                # do_chunk_postproc(chunkdir)    # chunkdir is DIRECTORY_INOUT or only IN if I save all outside chunkdirs as original PTF
            

        if os.path.getsize(inpfilePS) > 0:   #PS
            nsce_ps, first_line_ps = get_info_from_scenario_list(filename = inpfilePS,
                                                                 seistype = 'PS')
            if domain == 'med-tsumaps':
                inifolder = os.path.join(inpdir, domain, wd['ps_inicond_med'])
            force_symlink(inifolder, os.path.join(workdir, wd['ps_inicond_med']))

            njobs = int(nsce_ps / ngpus) + (nsce_ps % ngpus > 0)
            print('Number of jobs = ' + str(njobs))

            presimulations_setup(wd       = wd,
                                 nsce     = nsce_ps,
                                 njobs    = njobs,
                                 ngpus    = ngpus,
                                 seistype = 'PS')

            simdir = os.path.join(workdir,wd['step2_dir_sim_PS'])
            datadir = os.path.join(workdir,'datadir_sim')
            # loop over number of chunks
            for i in range(2): #range(njobs)
                chunkdir = os.path.join(simdir, 'chunk' + str(i+1))
                runHySEA(chunkdir, datadir)
                compss_wait_on_directory(chunkdir)
                # TODO: postprocessing of each simualtion output                                                                                      # do_chunk_postproc(chunkdir)    # chunkdir is DIRECTORY_INOUT or only IN if I save all outside chunkdirs as original PTF
        
        # compss_barrier()   # need to wait for all simulations (BS and PS) and first postproc stage to be done before doing postprocsseing that puts everything together
        # do_all_postproc    



#        step2_postproc(ptf_measure = ptf_measure_type,
#                       outdir      = workdir, 
#                       outfile_bs  = wd['step2_hmax_sim_BS'],
#                       outfile_ps  = wd['step2_hmax_sim_PS'],
#                       bs_path     = inpfileBS,
#                       ps_path     = inpfilePS,
#                       poi_depth   = "step2_pois_depth.npy")

# ###########################################################
if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
