#!/bin/bash -l

#SBATCH --job-name=misfit_evaluator
#SBATCH --output=log_%j.out
#SBATCH --account=leonardoACC
#SBATCH --partition=leonardoPART
#SBATCH --qos=leonardoQOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=2:00:00

source LOADENV
cd $SLURM_SUBMIT_DIR

workdir=$1
sim_folder_BS=$2
sim_folder_PS=$3
sim_folder_SBS=$4

export workdir
export sim_folder_BS
export sim_folder_PS
export sim_folder_SBS
python3 -c "import os,sys; sys.path.append('WFDIR'); import pyptf; from pyptf import misfit_evaluator as misfit; wdir=os.environ['workdir']; fBS = os.environ['sim_folder_BS']; fPS = os.environ['sim_folder_PS']; fSBS = os.environ['sim_folder_SBS']; misfit.main(workdir = wdir, sim_folder_BS = fBS, sim_folder_PS = fPS, sim_folder_SBS = fSBS)"