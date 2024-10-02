#!/bin/bash

cd REMOTEPATH

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    echo "Usage: Step2_leonardo_exe.sh BS(or PS) host  grdfile hours"
    exit 1
fi

seistype=$1
hpc_cluster=$2
grid=$3
prophours=$4

./Step2_launch_simul.sh $seistype $hpc_cluster $grid $prophours > log_BSsimul.out 2>&1 &
