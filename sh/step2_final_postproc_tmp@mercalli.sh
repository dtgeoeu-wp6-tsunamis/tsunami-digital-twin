#PBS -N postprocessing
#PBS -o log_postproc.out
#PBS -e SIM.err
#PBS -j oe
##PBS -q mercalli_gpu
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=300G

source LOADENV

cd $PBS_O_WORKDIR

python step2_create_ts_input_for_ptf.py --ptf_measure $ptf_measure_type --failed_BS $log_failed_BS --failed_PS $log_failed_PS --failed_SBS $log_failed_SBS --outfile_BS $outfile_BS --outfile_PS $outfile_PS --outfile_SBS $outfile_SBS --file_simdir_BS $file_simdir_BS --file_simdir_PS $file_simdir_PS --file_simdir_SBS $file_simdir_SBS --depth_file step2_pois_depth.npy