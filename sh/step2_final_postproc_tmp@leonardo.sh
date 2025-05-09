#!/bin/bash -l

#SBATCH --job-name=postprocessing
#SBATCH --output=log_%j.out
#SBATCH --account=leonardoACC
#SBATCH --partition=leonardoPART
#SBATCH --qos=leonardoQOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=12:00:00

source LOADENV
cd $SLURM_SUBMIT_DIR

ptf_measure_type=$1
log_failed_BS=$2
log_failed_PS=$3
log_failed_SBS=$4
outfile_BS=$5
outfile_PS=$6
outfile_SBS=$7
file_simdir_BS=$8
file_simdir_PS=$9
file_simdir_SBS=${10}
python step2_create_ts_input_for_ptf.py --ptf_measure $ptf_measure_type --failed_BS $log_failed_BS --failed_PS $log_failed_PS --failed_SBS $log_failed_SBS --outfile_BS $outfile_BS --outfile_PS $outfile_PS --outfile_SBS $outfile_SBS --file_simdir_BS $file_simdir_BS --file_simdir_PS $file_simdir_PS --file_simdir_SBS $file_simdir_SBS --depth_file step2_pois_depth.npy
