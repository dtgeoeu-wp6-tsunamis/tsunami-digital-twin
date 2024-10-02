#!/bin/bash -l
 
#SBATCH --job-name=TSU_SIM_XXZZZ
#SBATCH --output=LOG_DIR/log_XX_ZZZ_%j.out
#SBATCH --account=leonardoACC
#SBATCH --partition=leonardoPART
#SBATCH --qos=leonardoQOS
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
 
source LOADENV
cd $SLURM_SUBMIT_DIR 

EVENTSFILE=SCENARIOS_FILE

inpfile=simulationsXXZZZ.txt
touch $inpfile

for i in $(seq START END); do
    # numsce=$(printf %0NDIG\i $i)
    numsce_tmp=`sed -n ${i}p $EVENTSFILE | awk '{print $1}'`
    numsce=$(printf %0NDIG\i $numsce_tmp)
    EVENTNAME=XX_Scenario$numsce
    SIMDIR=SIM_DIR/$EVENTNAME
    export SIMDIR
    if [ ! -d $SIMDIR ]; then
        mkdir $SIMDIR
    fi
    params=`sed -n ${i}p $EVENTSFILE`
    export params
    #python3 -c "import os,sys; sys.path.append('PYDIR'); import t_hysea as hysea; pline=os.environ['params']; scedir=os.environ['SIMDIR']; hysea.create_input(line = pline, seistype = 'XX', simdir_XX = 'SIM_DIR', wdir = 'WDIR', outdir = scedir, ndig = NDIG, prop = HOURS)"
    python3 -c "import os,sys; sys.path.append('PYDIR'); import t_hysea as hysea; pline=os.environ['params']; scedir=os.environ['SIMDIR']; hysea.create_input(line = pline, seistype = 'XX', simdir_XX = 'SIM_DIR', XX_parfile_tmp = 'PARFILETMP', PS_inicond = 'INITCONDPS', wdir = 'WDIR', outdir = scedir, ndig = NDIG, prop = HOURS)"
    parfile=$SIMDIR/parfile.txt
    echo $parfile >> $inpfile
done

mpirun -np NUMPROC /leonardo_work/DTGEO_T1_2/T-HySEA-leonardo/T-HySEA_3.9.0_MC/bin/TsunamiHySEA $inpfile

while read line; do
    simdir=$(dirname $line)
    echo 'Postprocessing output for ' $simdir
    python step2_extract_ts.py --nc $simdir/out_ts.nc --depth_file step2_pois_depth.npy  --ptf_file $simdir/out_ts_ptf.nc
done < $inpfile
rm -f $inpfile
