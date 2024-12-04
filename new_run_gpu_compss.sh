#/bin/bash

# --worker_in_master_cpus=16 \
export COMPUTING_UNITS=4
export IMAGENAME="/leonardo_work/DTGEO_T1_2/images/dt-geo_t-hysea_x86_64_openmpi_4.1.4_cuda_11.8_v_dt-geo.sif"

enqueue_compss \
        --num_nodes=2 \
        --gpus_per_node=4 \
        --worker_working_dir=/leonardo/home/userexternal/mvolpe00/dtgeo/mini-ST610106 \
        --exec_time=30 \
        --lang=python \
	--python_interpreter=/leonardo/pub/userexternal/mvolpe00/spack-0.19.1-03/install/linux-rhel8-icelake/gcc-12.2.0/python-3.9.15-yb3f3aple76ttf5xu3c44kx2mtzixrvg/bin/python \
        --pythonpath=$PYTHONPATH:/leonardo/home/userexternal/mvolpe00/dtgeo/mini-ST610106:/leonardo/home/userexternal/mvolpe00/dtgeo/mini-ST610106/py \
        --debug \
	--keep_workingdir \
        --queue=boost_usr_prod \
	--qos=boost_qos_dbg \
        /leonardo/home/userexternal/mvolpe00/dtgeo/mini-ST610106/workflow_main.py --cfg /leonardo/home/userexternal/mvolpe00/dtgeo/mini-ST610106/cfg/ptf_main.config --input_workflow /leonardo/home/userexternal/mvolpe00/dtgeo/mini-ST610106/workflow_input.json

	#--qos=boost_qos_dbg \
