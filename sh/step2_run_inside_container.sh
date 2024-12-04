#!/bin/bash
echo "PARAMS: $1 $2"
source /etc/profile.d/z10_spack_environment.sh
cd $1
n=$(ls -d *Scenario* | head -1)
cp $2/* .
cp "${n}/parfile.txt" .
/opt/view/bin/get_load_balancing parfile.txt 1 1
mpirun -n NUMPROC /opt/view/bin/TsunamiHySEA problems.txt
rm *.bin *.grd *.dat 
