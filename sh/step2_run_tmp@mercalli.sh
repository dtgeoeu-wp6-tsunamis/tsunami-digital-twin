#PBS -N TSU_SIM_XXZZZ
#PBS -o LOG_DIR/SIMZZZ.log
#PBS -e SIM.err
#PBS -j oe
#PBS -q mercalli_gpu
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mpiprocs=8:ngpus=8:mem=300G
 
module purge
#module load gcc/8.1.0 openmpi/4.1.1/gcc-8.1.0 cuda/10.2 hdf5/1.8.18/gcc-8.1.0 netcdf/4.6.1/gcc-8.1.0
#module load  hysea/3.8.4
module load gcc/10.3.0 openmpi/4.1.1/gcc-10.3.0 cuda/11.4 hdf5/1.12.1/gcc-10.3.0 netcdf/4.7.4/gcc-10.3.0
module load  hysea/3.9.0

source LOADENV
cd $PBS_O_WORKDIR

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
    #python3 -c "import os,sys; sys.path.append('PYDIR'); import tsunami_simulations as tsu_sim; pline=os.environ['params']; scedir=os.environ['SIMDIR']; tsu_sim.create_input(line = pline, seistype = 'XX', simdir_XX = 'SIM_DIR', wdir = 'WDIR', outdir = scedir, ndig = NDIG, prop = HOURS)"
    python3 -c "import os,sys; sys.path.append('WFDIR'); import pyptf; from pyptf import tsunami_simulations as tsu_sim; pline=os.environ['params']; scedir=os.environ['SIMDIR']; tsu_sim.create_input(line = pline, seistype = 'XX', simdir = 'SIM_DIR', parfile_tmp = 'PARFILETMP', PS_inicond = 'INITCONDPS', wdir = 'WDIR', outdir = scedir, ndig = NDIG, prop = HOURS)"
    parfile=$SIMDIR/parfile.txt
    echo $parfile >> $inpfile
done

mpirun -np NUMPROC TsunamiHySEA $inpfile

while read line; do
    simdir=$(dirname $line)
    echo 'Postprocessing output for ' $simdir
    python3 step2_extract_ts.py --nc $simdir/out_ts.nc --depth_file step2_pois_depth.npy  --ptf_file $simdir/out_ts_ptf.nc 
done < $inpfile
rm -f $inpfile
