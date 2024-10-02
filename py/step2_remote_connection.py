import os
import sys
import fabric


def connect_to_remote_cluster(**kwargs):
    wd = kwargs.get('workflow_dict', None)
    inpfileBS = kwargs.get('inpfileBS', None)
    inpfilePS = kwargs.get('inpfilePS', None)
    nsce_bs = kwargs.get('nsce_bs', None)
    nsce_ps = kwargs.get('nsce_ps', None)
    first_line_bs = kwargs.get('first_line_bs', None)
    first_line_ps = kwargs.get('first_line_ps', None)
    ngpus = kwargs.get('ngpus', None)
    ptf_measure_type = kwargs.get('ptf_measure_type', None)

    workdir = wd['workdir']
    domain = wd['domain']
    if domain == 'med-tsumaps':
        inifolder = wd['ps_inicond_med']

    hpc_cluster = wd['hpc_cluster']
    username = wd['remote_user']
    remote_path = os.path.join(wd['remote_workpath'], wd['eventID'])
    remote_inpdir = wd['remote_inpdir']

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
        print('Transferring files...')
        c.put(os.path.join(wd['wf_path'], 'py', 't_hysea.py'), remote_path)
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
        if os.path.getsize(inpfileBS) > 0:
            c.put(inpfileBS, remote_path)
            c.put(os.path.join(workdir, wd['step2_list_simdir_BS']), remote_path)

            remote_execution(c           = c,
                             wd          = wd,
                             seistype    = 'BS',
                             inpfile     = inpfileBS,
                             inifolder   = inifolder,
                             line        = first_line_bs,
                             nscen       = nsce_bs,
                             remote_path = remote_path,
                             ngpus       = ngpus)

        if os.path.getsize(inpfilePS) > 0:
            c.put(inpfilePS, remote_path)
            c.put(os.path.join(workdir, wd['step2_list_simdir_PS']), remote_path)
            if domain == 'med-tsumaps':
                c.run('ln -sf ' + os.path.join(remote_inpdir, domain, inifolder + ' ' + remote_path))

            remote_execution(c           = c,
                             wd          = wd,
                             seistype    = 'PS',
                             inpfile     = inpfilePS,
                             inifolder   = inifolder,
                             line        = first_line_ps,
                             nscen       = nsce_ps,
                             remote_path = remote_path,
                             ngpus       = ngpus)

        c.run('rm -f ' + remote_path + '/run*sh', shell=True)
        # POSTPROC: Once ALL the simulations have been completed
        print('Postprocessing simulations...')
        if hpc_cluster == 'mercalli':
            cmd = 'qsub -W block=true -v ptf_measure_type=' + ptf_measure_type + ',log_failed_BS=' + wd['step2_log_failed_BS'] + ',log_failed_PS=' + wd['step2_log_failed_PS'] + \
                  ',outfile_BS=' + wd['step2_hmax_sim_BS'] + ',outfile_PS=' + wd['step2_hmax_sim_PS'] + \
                  ',file_simdir_BS=' + wd['step2_list_simdir_BS'] + ',file_simdir_PS=' + wd['step2_list_simdir_PS'] + ' ./step2_final_postproc.sh &'
        elif hpc_cluster == 'leonardo':
            cmd = 'sbatch -W ./step2_final_postproc.sh ' + ptf_measure_type + ' ' + wd['step2_log_failed_BS'] + ' ' + wd['step2_log_failed_PS'] + ' ' + \
                  wd['step2_hmax_sim_BS'] + ' ' +  wd['step2_hmax_sim_PS'] + ' ' + wd['step2_list_simdir_BS'] + ' ' + wd['step2_list_simdir_PS'] + '&'
        c.run('cd ' + remote_path + ';' + cmd)

        print('Copying back the final output from the simulations...')
        if os.path.getsize(inpfileBS) > 0:
            c.get(os.path.join(remote_path, wd['step2_hmax_sim_BS']), os.path.join(workdir, wd['step2_hmax_sim_BS']))
            c.get(os.path.join(remote_path, wd['step2_log_failed_BS']), os.path.join(workdir, wd['step2_log_failed_BS']))
        if os.path.getsize(inpfilePS) > 0:
            c.get(os.path.join(remote_path, wd['step2_hmax_sim_PS']), os.path.join(workdir, wd['step2_hmax_sim_PS']))
            c.get(os.path.join(remote_path, wd['step2_log_failed_PS']), os.path.join(workdir, wd['step2_log_failed_PS']))

 
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
    
    sim_folder = wd['step2_dir_sim_' + seistype]
    log_folder = wd['step2_dir_log_' + seistype]
    c.run('mkdir -p ' + os.path.join(remote_path, sim_folder))
    c.run('mkdir -p ' + os.path.join(remote_path, log_folder))

    pycmd = ''' source ''' + wd['envfile'] + '''; python -c "import os,sys; import t_hysea as hysea; pline = ' ''' + line.strip() + ''' ';\
            hysea.create_input(line = pline, seistype = ' ''' + seistype + ''' ', simdir_''' + seistype + ''' = ' ''' + sim_folder + ''' ', ''' + \
            seistype + '''_parfile_tmp = ' ''' + parfile_tmp + ''' ', PS_inicond = ' ''' + inifolder + \
            ''' ', wdir = ' ''' + remote_path + ''' ', outdir = ' ''' + remote_path + ''' ', ndig = ''' + str(nd) + ''', prop = ''' + str(prop) + ''')" '''

    if hpc_cluster == 'mercalli':
        c.run('cd ' + remote_path + ';' + pycmd + '; module load gcc/10.3.0 openmpi/4.1.1/gcc-10.3.0 cuda/11.4 hdf5/1.12.1/gcc-10.3.0 netcdf/4.7.4/gcc-10.3.0 hysea/3.9.0; get_load_balancing parfile.txt 1 1; rm parfile.txt')
    elif hpc_cluster == 'leonardo':
        c.run('cd ' + remote_path + ';' + pycmd + '; /leonardo_work/DTGEO_T1_2/T-HySEA-leonardo/T-HySEA_3.9.0_MC/bin_lb/get_load_balancing parfile.txt 1 1; rm parfile.txt')

    njobs = int(nscen / ngpus) + (nscen % ngpus > 0)
    print('Number of jobs = ' + str(njobs))

    pycmd = f'''python -c 'import os,sys; import t_hysea as hysea; hysea.submit_jobs(seistype=\"{seistype}\",scenarios_file=\"{inpfile}\", workflow_dict=\"{wd}\",local_host=\"{local_host}\",hpc_cluster=\"{hpc_cluster}\",nscenarios=\"{nscen}\",njobs= \"{njobs}\",ngpus=\"{ngpus}\")' '''
    #print(pycmd)
    c.run('cd ' + remote_path + '; source ' + wd['envfile'] + '; ' + pycmd)

