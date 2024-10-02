#!/bin/bash -l
 
#SBATCH --job-name=TSU_SIM_XX
#SBATCH --output=logfiles_XX/log_XX_%a/log_XX_%A_%a.out
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
 
source Step2_requirements_leonardo.source 

cd $SLURM_SUBMIT_DIR 

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

EVENTSFILE=Step1_scenario_list_XX.txt
nsce=$(cat $EVENTSFILE | wc | awk '{print $1}')

inpfile=simulationsXX$SLURM_ARRAY_TASK_ID.txt
touch $inpfile

START=$(( ($SLURM_ARRAY_TASK_ID - 1 ) * 4 + 1 ))
END=$(( $SLURM_ARRAY_TASK_ID  * 4  ))
NUMPROC=4
if [ $END -gt $nsce ]; then
    END=$nsce
    NUMPROC=$(( ($END - $START ) + 1 ))
fi

for i in $(seq $START $END); do
    numsce=$(printf %0NDIG\i $i)
    EVENTNAME=XX_Scenario$numsce
    SIMDIR=Step2_XX/$EVENTNAME
    export SIMDIR
    if [ ! -d $SIMDIR ]; then
        mkdir $SIMDIR
    fi
    params=`sed -n ${i}p $EVENTSFILE`
    export params
    python -c "import os,sys; import t_hysea as hysea; pline=os.environ['params']; scedir=os.environ['SIMDIR']; hysea.create_input(line = pline, seistype = 'XX', wdir = 'WDIR', outdir = scedir, ndig = NDIG, grid = 'GRID', prop = HOURS)"

    parfile=$SIMDIR/parfile.txt
    echo $parfile >> $inpfile
done

mpirun -np $NUMPROC /leonardo_work/DTGEO_T1_2/T-HySEA-leonardo/T-HySEA_3.9.0_MC/bin/TsunamiHySEA $inpfile

while read line; do
    simdir=$(dirname $line)
    cp Step2_extract_ts.py $simdir
    cp Step2_pois_depth.npy $simdir
    echo 'Postprocessing output for ' $simdir
    cd $simdir
    srun -n1 --gres=gpu:1 --exclusive python Step2_extract_ts.py --nc out_ts.nc --depth_file Step2_pois_depth.npy --ptf_file out_ts_ptf.nc &
    cd ../..
    
done < $inpfile
wait
rm -f $inpfile
