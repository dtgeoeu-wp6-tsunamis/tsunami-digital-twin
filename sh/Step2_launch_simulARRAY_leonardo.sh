#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    echo "Usage: Step2_launch_simul.sh BS (or PS) nsce njobs hpc_cluster wdir pydir grid ndig prophours"
    exit 1
fi

seistype=$1
nscenarios=$2
narraytasks=$3
hpc_cluster=$4
nep=$5
workdir=$6
grid=$7
nd=$8
prophours=$9

#for j in $(seq 1 $narraytasks); do
for j in $(seq 1 2); do    #######################TEST

    if [ ! -d logfiles_$seistype/log_${seistype}_$i ]; then
        mkdir logfiles_$seistype/log_${seistype}_$i
    fi
done

#submitting jobs
sed s/XX/$seistype/g < Step2_run_tmp.sh > run$seistype\.sh
sed -i s+WDIR+$workdir+ run$seistype\.sh
sed -i s/GRID/$grid/ run$seistype\.sh
sed -i s/NDIG/$nd/ run$seistype\.sh
sed -i s/HOURS/$prophours/ run$seistype\.sh
    
#sbatch -W --array=1-$narraytasks run$seistype\.sh   #|to check
echo 'sbatch -W --array=1-$narraytasks run$seistype\.sh   #|to check'

wait

echo $seistype ' simulations completed'
rm run$seistype\.sh 
