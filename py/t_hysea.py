import os
import sys
import math
import shutil
import numpy as np
import subprocess as sp
from pyutil import filereplace


def execute_lb_on_localhost(**kwargs):
    wd = kwargs.get('workflow_dict', None)
    seistype = kwargs.get('seistype', None)
    line = kwargs.get('line', None)

    workdir = wd['workdir']
    propagation = wd['propagation']
    ndig = wd['n_digits']

    create_input(line = line,
                 seistype = seistype,
                 simdir_BS =  wd['step2_dir_sim_BS'],
                 simdir_PS =  wd['step2_dir_sim_PS'],
                 BS_parfile_tmp = wd['BS_parfile_tmp'],
                 PS_parfile_tmp = wd['PS_parfile_tmp'],
                 PS_inicond = wd['ps_inicond_med'],
                 wdir = workdir,
                 outdir = workdir,
                 ndig = ndig,
                 prop = propagation)

    if wd['hpc_cluster'] == 'mercalli':
        sp.run('module load gcc/10.3.0 openmpi/4.1.1/gcc-10.3.0 cuda/11.4 hdf5/1.12.1/gcc-10.3.0 netcdf/4.7.4/gcc-10.3.0 hysea/3.9.0; get_load_balancing parfile.txt 1 1', shell=True)
    elif wd['hpc_cluster'] == 'leonardo':
        sp.run('/leonardo_work/DTGEO_T1_2/T-HySEA-leonardo/T-HySEA_3.9.0_MC/bin_lb/get_load_balancing parfile.txt 1 1', shell=True)
    os.remove('parfile.txt')


def save_pois_for_hysea(**kwargs):
    """
    write the coordinates of the pois to file for T-HYSEA
    """
    poifile = kwargs.get('poifile', None)
    workdir = kwargs.get('workdir', None)
    outfile = kwargs.get('outfile', None)

    pois = np.load(poifile, allow_pickle=True).item()['pois_coords']
    npois = pois.shape[0]
    poifile_ts = os.path.join(workdir, outfile)
    np.savetxt(poifile_ts,pois,fmt='%f',header=str(npois),comments='')


def create_input(**kwargs):
    line = kwargs.get('line', None)
    seistype = kwargs.get('seistype', None)
    simdir_BS = kwargs.get('simdir_BS', None)
    simdir_PS = kwargs.get('simdir_PS', None)
    BS_parfile_tmp = kwargs.get('BS_parfile_tmp', None)
    PS_parfile_tmp = kwargs.get('PS_parfile_tmp', None)
    PS_inicond = kwargs.get('PS_inicond', None)
    workdir = kwargs.get('wdir', None)
    outdir = kwargs.get('outdir', None)
    nd = kwargs.get('ndig', None)
    prophours = kwargs.get('prop', None)

    seistype = seistype.strip()
    workdir = workdir.strip()
    outdir = outdir.strip()

    simtime = str(float(prophours)*3600+2)   #8h=28800sec

    scenum = line.split()[0]

    if seistype == 'BS':
        tmpfile = os.path.join(workdir, os.path.basename(BS_parfile_tmp).strip()) #'step2_parfile_tmp.txt')
        parfile = os.path.join(outdir, 'parfile.txt')
        cp = shutil.copy(tmpfile, parfile)

        eqlon = line.split( )[3]
        eqlat = line.split( )[4]
        eqtop = line.split( )[5]
        eqstk = line.split( )[6]
        eqdip = line.split( )[7]
        eqrak = line.split( )[8]
        eqlen = line.split( )[9]
        eqarea = line.split( )[10]
        eqslip = line.split( )[11]

        idscen = 'BS_Scenario' + scenum.zfill(nd)
        simfolder = simdir_BS
        eqwid = str(float(eqarea)/float(eqlen))
        if float(eqtop) == 1:
            eqtop = '0'    #correction for okada
        eqdep = str(float(eqtop)+float(eqwid)/2.*math.sin(math.radians(float(eqdip))))

        filereplace(parfile,'eqlon', eqlon)
        filereplace(parfile,'eqlat', eqlat)
        filereplace(parfile,'eqdep', eqdep)
        filereplace(parfile,'eqlen', eqlen)
        filereplace(parfile,'eqwid', eqwid)
        filereplace(parfile,'eqstk', eqstk)
        filereplace(parfile,'eqdip', eqdip)
        filereplace(parfile,'eqrak', eqrak)
        filereplace(parfile,'eqslip', eqslip)

    elif seistype == 'PS':
        tmpfile = os.path.join(workdir, os.path.basename(PS_parfile_tmp).strip()) #'step2_parfile_TRI_tmp.txt')
        parfile = os.path.join(outdir,'parfile.txt')
        cp = shutil.copy(tmpfile, parfile)

        inicond = line.split()[2]

        idscen = 'PS_Scenario' + scenum.zfill(nd)
        simfolder = simdir_PS
        inifile = os.path.join(workdir, PS_inicond.strip(), inicond)
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

    filereplace(parfile,'simfolder', simfolder)
    filereplace(parfile,'idscen', idscen)
    filereplace(parfile,'simtime', simtime)

def submit_jobs(**kwargs):
    seistype = kwargs.get('seistype', None)
    scenarios_file = kwargs.get('scenarios_file', None)
    wd = kwargs.get('workflow_dict', None)
    local_host = kwargs.get('local_host', None)
    hpc_cluster = kwargs.get('hpc_cluster', None)
    nsce = kwargs.get('nscenarios', None)
    njobs = kwargs.get('njobs', None)
    nep = kwargs.get('ngpus', None)   #Number of events to be simulated in joint packet

    seistype = seistype.strip()

    if hpc_cluster == local_host:
        workdir = wd['workdir'].strip()
        sim_folder = wd['step2_dir_sim_' + seistype]
        log_folder = wd['step2_dir_log_' + seistype]
        parfile_tmp = wd[seistype + '_parfile_tmp']
        run_filename = wd['run_sim_filename']
        ps_inicond = wd['ps_inicond_med']
        # hpc_cluster = wd['hpc_cluster']
        nd = wd['n_digits']
        prophours = wd['propagation']
    else:   #when passed by remote_connection all arguments are strings
        remote_workpath = wd.split('remote_workpath:')[1].split(',')[0].strip()
        eventID = wd.split('eventID:')[1].split(',')[0].strip()
        sim_folder = wd.split('step2_dir_sim_' + seistype + ':')[1].split(',')[0].strip()
        log_folder = wd.split('step2_dir_log_' + seistype + ':' )[1].split(',')[0].strip()
        parfile_tmp = wd.split(seistype + '_parfile_tmp:')[1].split(',')[0].strip()
        run_filename = wd.split('run_sim_filename:')[1].split(',')[0].strip()
        ps_inicond = wd.split('ps_inicond_med:')[1].split(',')[0].strip()
        nd = wd.split('n_digits:')[1].split(',')[0].strip()
        prophours = wd.split('propagation:')[1].split(',')[0].strip()
        workdir = os.path.join(remote_workpath, eventID)
        scenarios_file = os.path.join(workdir,os.path.basename(scenarios_file))
        nsce = int(nsce)
        njobs = int(njobs)
        nep = int(nep)

    if not os.path.exists(sim_folder):
        os.mkdir(sim_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

#submit jobs 
    #for j in range(njobs):
    for j in range(2):     ###for testing###
        kk = j + 1
        print('chunk ' + str(kk) + '/' + str(njobs))

        s1 = (j * nep) + 1
        s2 = kk * nep
        if kk == njobs:
            s2 = nsce
        nn = (s2 - s1) + 1

        tmpfile = os.path.join(workdir, run_filename) #os.path.join(workdir,'step2_run_tmp.sh')
        runfile = os.path.join(workdir,'run' + seistype + str(kk) + '.sh')
        cp = shutil.copy(tmpfile, runfile)
        filereplace(runfile,'XX', seistype)
        filereplace(runfile,'ZZZ', str(kk))
        filereplace(runfile,'SCENARIOS_FILE', scenarios_file)
        filereplace(runfile,'LOG_DIR', log_folder)
        filereplace(runfile,'SIM_DIR', sim_folder)
        filereplace(runfile, 'WDIR', workdir)
        filereplace(runfile,'PARFILETMP', parfile_tmp)
        filereplace(runfile,'INITCONDPS', ps_inicond)
        filereplace(runfile,'NUMPROC', str(nn))
        filereplace(runfile,'START', str(s1))
        filereplace(runfile,'END', str(s2))
        filereplace(runfile,'NDIG', str(nd))
        filereplace(runfile,'HOURS', str(prophours))

        if hpc_cluster == 'mercalli':
            sp.run('qsub -W block=true ' + runfile + '&',shell=True)
        elif hpc_cluster == 'leonardo':
            sp.run('sbatch -W ' + runfile + '&',shell=True)

    print(seistype + ' simulations submitted')

   #  cmd = './Step2_launch_simul.sh ' + seistype + ' ' + scenarios_file + ' ' + sim_folder + ' ' + str(nsce) + ' ' + str(njobs) + ' ' + hpc_cluster + ' ' + str(nep) + ' ' + workdir + ' ' + str(nd) + ' ' + str(prophours) + ' ' + log_folder 
   #  sp.run(cmd,shell=True)
