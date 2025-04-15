import os
import sys
import errno
import math
import shutil
import numpy as np
import subprocess as sp
from pyutil import filereplace


def force_symlink(file1,file2):
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)


def get_info_from_scenario_list(**kwargs):

    filename = kwargs.get('filename', None)
    seistype = kwargs.get('seistype', None)
    logger = kwargs.get('logger', None)

    with open(filename, 'r') as f:
        first_line = f.readline()
        if first_line:
            nsce = len(f.readlines()) + 1
            logger.info('Number of {} scenario = {}'.format(seistype, str(nsce)))
        else:
            raise Exception

    return nsce, first_line


def create_setup(**kwargs):
    wd = kwargs.get('workflow_dict', None)
    seismicity_types = kwargs.get('seis_types', None)
    inpfileBS = kwargs.get('inpfileBS', None)
    inpfilePS = kwargs.get('inpfilePS', None)
    inpfileSBS = kwargs.get('inpfileSBS', None)
    pois_d = kwargs.get('pois_d', None)
    logger = kwargs.get('logger', None)

    workdir = wd['workdir']
    inpdir = wd['inpdir']
    sim_domain = wd['sim_domain']
    domain = wd['domain']
    eventID = wd['uniqueID']
    hpc_cluster = wd['hpc_cluster']
    UCmode = wd['UCmode']
    # pydir = os.path.join(wd['wf_path'], 'pyptf')

    if sim_domain == 'regional': # TODO automatic procedure for creating grid for the global version 
        # poifile = os.path.join(workdir, wd['pois'])
        pois = pois_d['pois_coords']
        if wd['ptf_version'] == 'neam':
            bathyfile = os.path.join(inpdir, domain, wd['regional_bathy_file'])
            force_symlink(bathyfile, os.path.join(workdir, wd['bathy_filename']))
            # force_symlink(depthfile, os.path.join(workdir, wd['depth_filename']))

        # print(type(pois_d['pois_depth']), np.mean(pois_d['pois_depth']), len(pois_d['pois_depth']))
        np.save(os.path.join(workdir, wd['depth_filename']), pois_d['pois_depth'])
        #depthfile = os.path.join(inpdir, domain, wd['regional_pois_depth'])
    else:  # local (TODO: add automatic procedure for cutting the grid and selecting the POIs)
        # poifile = os.path.join(inpdir, domain, 'POIs_local_domain_' + eventID + '.npy') TODO
        bathyfile = os.path.join(workdir, 'local_domain_' + eventID + '.grd')
        # depthfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_' + eventID + '_POIs_depth.npy')
        # # !!only fos SAMOS!!
        # # poifile = os.path.join(inpdir, domain, 'POIs_local_domain_2020_1030_samos.npy') TODO
        # bathyfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_2020_1030_samos.grd')
        # depthfile = os.path.join(inpdir, domain, 'bathy_grids/local_domain_2020_1030_samos_POIs_depth.npy')

    save_pois_for_hysea(pois    = pois, #poifile = poifile,
                        workdir = workdir,
                        outfile = wd['step2_ts_file'])

    input_files = {'BS': inpfileBS, 'PS': inpfilePS, 'SBS': inpfileSBS}
    num_scenarios = dict()
    first_lines = dict()
    for seistype in seismicity_types:
        try:
            num_scenarios[seistype], first_lines[seistype] = get_info_from_scenario_list(filename = input_files[seistype],
                                                                                         seistype = seistype,
                                                                                         logger   = logger)
            cp = shutil.copy(wd[seistype + '_parfile_tmp'], workdir)
            if seistype == 'PS':
                if domain == 'med-tsumaps':
                    inifolder = os.path.join(inpdir, domain, wd['ps_inicond_med'])
                    force_symlink(inifolder, os.path.join(workdir, wd['ps_inicond_med']))
        except:
            logger.info(f'No {seistype} scenarios')
            num_scenarios[seistype] = 0
            first_lines[seistype] = 'None'
        
    logger.info('--------------')
    logger.info('Preparing submission scripts for ' + hpc_cluster)

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

        logger.info('HPC cluster for simulations not properly defined')
        sys.exit('Nothing to do....Exiting')

    filereplace(runtmp, 'LOADENV', wd['envfile'])
    filereplace(runtmp, 'WFDIR', wd['wf_path'])
    # filereplace(runtmp, 'PYDIR', pydir)
    # filereplace(runtmp, 'WDIR', workdir)
    filereplace(postproc, 'LOADENV', wd['envfile'])

    return  num_scenarios, first_lines, ngpus

def execute_lb_on_localhost(**kwargs):
    wd = kwargs.get('workflow_dict', None)
    seistype = kwargs.get('seistype', None)
    line = kwargs.get('line', None)

    workdir = wd['workdir']
    propagation = wd['propagation']
    ndig = wd['n_digits']

    create_input(line = line,
                 seistype = seistype,
                 simdir =  wd['step2_dir_sim_' + seistype],
                 parfile_tmp = wd[seistype + '_parfile_tmp'],
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
    # poifile = kwargs.get('poifile', None)
    pois = kwargs.get('pois', None)
    workdir = kwargs.get('workdir', None)
    outfile = kwargs.get('outfile', None)

    # pois = np.load(poifile, allow_pickle=True).item()['pois_coords']
    npois = pois.shape[0]
    poifile_ts = os.path.join(workdir, outfile)
    np.savetxt(poifile_ts,pois,fmt='%f',header=str(npois),comments='')


def create_input(**kwargs):
    line = kwargs.get('line', None)
    seistype = kwargs.get('seistype', None)
    simdir = kwargs.get('simdir', None)
    parfile_tmp = kwargs.get('parfile_tmp', None)
    PS_inicond = kwargs.get('PS_inicond', None)
    workdir = kwargs.get('wdir', None)
    outdir = kwargs.get('outdir', None)
    nd = kwargs.get('ndig', None)
    prophours = kwargs.get('prop', None)

    seistype = seistype.strip()
    workdir = workdir.strip()
    outdir = outdir.strip()

    simtime = str(float(prophours)*3600+2)   #8h=28800sec

    scenum = line.split()[0].split('.')[0]

    tmpfile = os.path.join(workdir, os.path.basename(parfile_tmp).strip()) #'step2_parfile_tmp.txt')
    parfile = os.path.join(outdir, 'parfile.txt')
    cp = shutil.copy(tmpfile, parfile)

    if seistype == 'BS' or seistype == 'SBS':

        eqlon = line.split( )[3]
        eqlat = line.split( )[4]
        eqtop = line.split( )[5]
        eqstk = line.split( )[6]
        eqdip = line.split( )[7]
        eqrak = line.split( )[8]
        eqlen = line.split( )[9]
        eqarea = line.split( )[10]
        eqslip = line.split( )[11]

        idscen = seistype + '_Scenario' + scenum.zfill(nd)
        eqwid = str(float(eqarea)/float(eqlen))
        if float(eqtop) == 1:
            eqtop = '0'    #correction for okada
        eqdep = str(float(eqtop)+float(eqwid)/2.*math.sin(math.radians(float(eqdip))))

        filereplace(parfile,'XX', seistype)
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

        inicond = line.split()[2]

        idscen = seistype + '_Scenario' + scenum.zfill(nd)
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

    filereplace(parfile,'simfolder', simdir)
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
        nd = wd['n_digits']
        prophours = wd['propagation']
    else:   #when passed by remote_connection all arguments are strings
        wd = wd.split('{')[1].split('}')[0]
        workdir = wd.split('remote_path:')[1].split(',')[0].strip()
        sim_folder = wd.split('step2_dir_sim_' + seistype + ':')[1].split(',')[0].strip()
        log_folder = wd.split('step2_dir_log_' + seistype + ':' )[1].split(',')[0].strip()
        parfile_tmp = wd.split(seistype + '_parfile_tmp:')[1].split(',')[0].strip()
        run_filename = wd.split('run_sim_filename:')[1].split(',')[0].strip()
        ps_inicond = wd.split('ps_inicond_med:')[1].split(',')[0].strip()
        nd = wd.split('n_digits:')[1].split(',')[0].strip()
        prophours = wd.split('propagation:')[1].split(',')[0].strip()
        scenarios_file = os.path.join(workdir, scenarios_file)
        nsce = int(nsce)
        njobs = int(njobs)
        nep = int(nep)

    if not os.path.exists(sim_folder):
        os.mkdir(sim_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    #submit jobs 
    #for j in range(njobs):
    for j in range(1):     ###for testing###
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
