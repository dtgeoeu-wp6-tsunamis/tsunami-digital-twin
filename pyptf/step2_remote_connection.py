import os
import sys
import fabric
import numpy as np
import h5py


def connect_to_remote_cluster(**kwargs):
    wd = kwargs.get('workflow_dict', None)
    input_files = kwargs.get('input_files', None)
    nscen = kwargs.get('nscen', None)
    first_lines = kwargs.get('first_lines', None)
    seismicity_types = kwargs.get('seismicity_types', None)
    ngpus = kwargs.get('ngpus', None)
    ptf_measure_type = kwargs.get('ptf_measure_type', None)

    workdir = wd['workdir']
    domain = wd['domain']
    if domain == 'med-tsumaps':
        inifolder = wd['ps_inicond_med']

    hpc_cluster = wd['hpc_cluster']
    username = wd['remote_user']
    remote_path = os.path.join(wd['remote_workpath'], wd['eventID'] + f"__s{wd['sigma']}" + f"__{wd['hazard_mode']}")
    remote_inpdir = wd['remote_inpdir']
    remote_pyptf = os.path.join(remote_path, 'pyptf')

    if hpc_cluster == 'leonardo':  # leonardo @CINECA#
        print('You are connecting to Leonardo @CINECA')
        print('Please be sure the temporary ssh certificate has been dowloaded')
        print('via smallstep client, otherwise the connection will fail')
        print('(The certificate has a 12-hours validity)')
        c_host = username + '@login.leonardo.cineca.it'
    elif hpc_cluster == 'mercalli':  #Mercalli @INGV#
        print('You are connecting to Mercalli @INGV')
        print('Please be sure you are on the INGV network, otherwise the connection will fail')
        c_host = username + '@mercalli.int.ingv.it'
    elif hpc_cluster != 'leonardo' and hpc_cluster != 'mercalli':
        print('Remote HPC cluster for simulations not properly defined')
        sys.exit('Nothing to do....Exiting')
   
    with fabric.Connection(host=c_host) as c:
        print('Creating remote folder...')
        c.run('mkdir -p ' + remote_path)
        c.run('mkdir -p ' + remote_pyptf)
        print('Transferring files...')
        c.put(os.path.join(wd['wf_path'], 'pyptf', 'tsunami_simulations.py'), remote_pyptf)
        c.put(os.path.join(wd['wf_path'], 'pyptf', '__init__.py'), remote_pyptf)
        c.put(os.path.join(workdir, 'step2_run_tmp.sh'), remote_path)
        c.put(os.path.join(workdir, 'step2_final_postproc.sh'), remote_path)
        c.put(os.path.join(workdir, 'step2_extract_ts.py'), remote_path)
        c.put(os.path.join(workdir, 'step2_create_ts_input_for_ptf.py'), remote_path)
        c.put(os.path.join(workdir, 'step2_bathygrid.grd'), remote_path)
        c.put(os.path.join(workdir, 'step2_pois_depth.npy'), remote_path)
        c.put(os.path.join(workdir, 'step2_ts.dat'), remote_path)
        c.put(os.path.join(workdir, 'step2_parfile_tmp.txt'), remote_path)
        c.put(os.path.join(workdir, 'step2_parfile_TRI_tmp.txt'), remote_path)

        print('...going to run T-HySea')
        print('The remote working folder is ' + remote_path)
        print('The remote input folder is  ' + remote_inpdir)
        print('The environment will be activated through the file ' + wd['envfile'])
        print('--------------')

        for seistype in seismicity_types:

            if nscen[seistype] != 0:
        #if os.path.getsize(inpfileBS) > 0:
                c.put(input_files[seistype], remote_path)
                c.put(os.path.join(workdir, wd['step2_list_simdir_' + seistype]), remote_path)

                if seistype == 'PS' and domain == 'med-tsumaps':
                    c.run('ln -sf ' + os.path.join(remote_inpdir, domain, inifolder + ' ' + remote_path))
                
                remote_execution(c           = c,
                                 wd          = wd,
                                 seistype    = seistype,
                                 inpfile     = os.path.basename(input_files[seistype]),
                                 inifolder   = inifolder,
                                 line        = first_lines[seistype],
                                 nscen       = nscen[seistype],
                                 remote_path = remote_path,
                                 ngpus       = ngpus)

        c.run('rm -f ' + remote_path + '/run*sh', shell=True)
        # POSTPROC: Once ALL the simulations have been completed
        print('Postprocessing simulations...')
        if hpc_cluster == 'mercalli':
            cmd = 'qsub -W block=true -v ptf_measure_type=' + ptf_measure_type + \
                  ',log_failed_BS=' + wd['step2_log_failed_BS'] + ',log_failed_PS=' + wd['step2_log_failed_PS'] + ',log_failed_SBS=' + wd['step2_log_failed_SBS'] + \
                  ',outfile_BS=' + wd['step2_hmax_sim_BS'] + ',outfile_PS=' + wd['step2_hmax_sim_PS'] + ',outfile_SBS=' + wd['step2_hmax_sim_SBS'] + \
                  ',file_simdir_BS=' + wd['step2_list_simdir_BS'] + ',file_simdir_PS=' + wd['step2_list_simdir_PS'] + ',file_simdir_SBS=' + wd['step2_list_simdir_SBS'] + \
                  ' ./step2_final_postproc.sh &'
        elif hpc_cluster == 'leonardo':
            cmd = 'sbatch -W ./step2_final_postproc.sh ' + ptf_measure_type + ' ' + \
                  wd['step2_log_failed_BS'] + ' ' + wd['step2_log_failed_PS'] + ' ' + wd['step2_log_failed_SBS'] + ' ' + \
                  wd['step2_hmax_sim_BS'] + ' ' +  wd['step2_hmax_sim_PS'] + ' ' + wd['step2_hmax_sim_SBS'] + ' ' + \
                  wd['step2_list_simdir_BS'] + ' ' + wd['step2_list_simdir_PS'] + ' ' + wd['step2_list_simdir_SBS'] + '&'

        c.run('cd ' + remote_path + ';' + cmd)

        print('Copying back the final output from the simulations...')
        for seistype in seismicity_types:
            if nscen[seistype] != 0:
                c.get(os.path.join(remote_path, wd['step2_hmax_sim_' + seistype]), os.path.join(workdir, wd['step2_hmax_sim_' + seistype]))
                c.get(os.path.join(remote_path, wd['step2_log_failed_' + seistype]), os.path.join(workdir, wd['step2_log_failed_' + seistype]))

 
def remote_execution(**kwargs):
    c = kwargs.get('c', None)
    wd = kwargs.get('wd', None)
    seistype = kwargs.get('seistype', None)
    inpfile = kwargs.get('inpfile', None)
    inifolder = kwargs.get('inifolder', None)
    line        = kwargs.get('line', None)
    nscen       = kwargs.get('nscen', None)
    remote_path = kwargs.get('remote_path', None)
    ngpus = kwargs.get('ngpus', None)

    prop = wd['propagation']
    nd = wd['n_digits']
    parfile_tmp = wd[seistype + '_parfile_tmp']
    local_host = wd['local_host']
    hpc_cluster = wd['hpc_cluster']

    wd_new = dict()
    wd_new['propagation'] = prop
    wd_new['n_digits'] = nd
    wd_new['ps_inicond_med'] = wd['ps_inicond_med']
    wd_new['run_sim_filename'] = wd['run_sim_filename']
    wd_new[seistype + '_parfile_tmp'] = wd[seistype + '_parfile_tmp']
    wd_new['step2_dir_sim_' + seistype] = wd['step2_dir_sim_' + seistype]
    wd_new['step2_dir_log_' + seistype] = wd['step2_dir_log_' + seistype]
    wd_new['remote_path'] = remote_path

    sim_folder = wd['step2_dir_sim_' + seistype]
    log_folder = wd['step2_dir_log_' + seistype]
    c.run('mkdir -p ' + os.path.join(remote_path, sim_folder))
    c.run('mkdir -p ' + os.path.join(remote_path, log_folder))

    # load balancing
    pycmd = ''' source ''' + wd['envfile'] + '''; python -c "import os,sys; import pyptf; from pyptf import tsunami_simulations as tsu_sim; pline = ' ''' + line.strip() + ''' ';\
            tsu_sim.create_input(line = pline, seistype = ' ''' + seistype + ''' ', simdir = ' ''' + sim_folder + ''' ', parfile_tmp = ' ''' + parfile_tmp + \
            ''' ', PS_inicond = ' ''' + inifolder + ''' ', wdir = ' ''' + remote_path + ''' ', outdir = ' ''' + remote_path + ''' ', ndig = ''' + str(nd) + ''', prop = ''' + str(prop) + ''')" '''

    if hpc_cluster == 'mercalli':
        c.run('cd ' + remote_path + ';' + pycmd + '; module load gcc/10.3.0 openmpi/4.1.1/gcc-10.3.0 cuda/11.4 hdf5/1.12.1/gcc-10.3.0 netcdf/4.7.4/gcc-10.3.0 hysea/3.9.0; get_load_balancing parfile.txt 1 1; rm parfile.txt')
    elif hpc_cluster == 'leonardo':
        c.run('cd ' + remote_path + ';' + pycmd + '; /leonardo_work/DTGEO_T1_2/T-HySEA-leonardo/T-HySEA_3.9.0_MC/bin_lb/get_load_balancing parfile.txt 1 1; rm parfile.txt')

    njobs = int(nscen / ngpus) + (nscen % ngpus > 0)
    print('Number of jobs = ' + str(njobs))

    # submit jobs
    pycmd = f'''python -c 'import os,sys; import pyptf; from pyptf import tsunami_simulations as tsu_sim; tsu_sim.submit_jobs(seistype=\"{seistype}\",scenarios_file=\"{inpfile}\", workflow_dict=\"{wd_new}\",local_host=\"{local_host}\",hpc_cluster=\"{hpc_cluster}\",nscenarios=\"{nscen}\",njobs= \"{njobs}\",ngpus=\"{ngpus}\")' '''
    #print(pycmd)
    c.run('cd ' + remote_path + '; source ' + wd['envfile'] + '; ' + pycmd)

