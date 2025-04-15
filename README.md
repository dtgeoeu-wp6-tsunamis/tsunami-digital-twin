## TSUNAMI DIGITAL TWIN


## Description
Tsunami Digital Twin (TDT) is the tsunami component of the prototype Digital Twin developed in the frame of DT-GEO project.
The component is based on the Python implementation of the Probabilistic Tsunami Forecasting (PTF) tool [1].

The workflow starts by using the first available information of an earthquake event, stored in a .json file, and performs the following steps:  
Step 1: Compute the ensemble of scenarios compatible with the occurring event, and associated probabilities  
Step 2: a) Retrieve tsunami simulations' results from precomputed database or b) Run simulations for each scenario (using the GPU-based code T-HySEA [2])  
Step 3: Compute hazard curves for each point of interest  
Step 4: Compute alert levels  
Step 5: Visualise the results (e.g., hazard and alert maps)  

#### References
[1] J. Selva, S. Lorito, M. Volpe, F. Romano, R. Tonini, P. Perfetti, F. Bernardi, M. Taroni, A. Scala, A. Babeyko, F. Løvholt, S. J. Gibbons, J. Macías, M. J. Castro, J. M. González-Vida, C. Sánchez-Linares, H. B. Bayraktar, R. Basili, F. E. Maesano, M. M. Tiberti, F. Mele, A. Piatanesi & Amato, A. (2021). Probabilistic tsunami forecasting for early warning. Nature communications, 12(1), 5677.

[2] https://edanya.uma.es/hysea/

## Requirements
TDT requires Python 3.x and the Python modules listed in the requirements.txt file.

Optional Step 2b requires the installation of T-HySEA (MC version).


## Installation
Download the whole package or clone the repository doing:
```
git clone git@gitlab.rm.ingv.it:dt-geo/tsunami-digital-twin.git
```
We suggest to use the package manager Spack to setup a working environment with all the needed packages to run this workflow ([see instructions](https://dtgeoeu-wp6-tsunamis.github.io/dt-geo-wp6-docs/spack-for-ptf/local) - Note that these instructions include the installation of Spack, which might not be needed if you are running the workflow on an HPC cluster. In that case, check if Spack is already available as a module.). 

Once created, the Spack environment can be activated with the command:

```
spack env activate -p <SPACK-ENV-NAME>
```

Note that any other package manager can be used as well (e.g., _Conda_, _pyenv_, _pip_, _poetry_).

## Preparation
Some operations are required before running the TDT.  
First, choose a local working directory to store personal settings and results:
```
mkdir path_to_your_working_dir/
``` 
Then copy the template of the main input file:  
```
cd tsunami-digital-twin
cp templates/workflow_input.json.template path_to_your_working_dir/workflow_input.json
```
In this .json file you can find the main setting to run the tool (see below).

If you are running Step 2b you need also to copy this template file:
```
cp templates/load_env.source.template path_to_your_working_dir/load_env.source
```
In this .source file you have to modify the command line according to the environment method and name.

### Use of remote connection
If you are planning to execute Step 2b on a remote HPC cluster (available clusters are presently mercalli @INGV and leonardo @CINECA), some operations must be repeated on the remote cluster, i.e. i) setting up the Pyhton environment, ii) creating a working folder and, iii) adding the file `load_env.source` containing the proper command to activate the environment. Specific command line arguments will be needed (see section Usage).

## Inputs

### Input file workflow_input.json
JSON (JavaScript Object Notation) is an open standard file format that uses human-readable text. The data are written in key/value pairs, separated by a colon(:). Different key/value pairs are separated by a comma(,).

A template of the `workflow_input.json` is provided, where all the required keys are already defined and grouped as objects, i.e. a collection of key/value pairs surrounded by a curly brace. Comments are disseminated all along the file for the sake of clarity, and featured by the key "_comment".

Before running the workflow, the JSON input file should be appropriately filled, although some sections may be neglected, depending on which workflow steps will be executed. In other words, not all the sections in the JSON file need to be filled from the beginning if the corresponding step is not expected to be executed, as each step can be turned on/off by setting True/False within the input file. On the other hand, each step needs the output of the previous steps.

Three main sections are present:

- Section STEPS: this is to define whether you want to run or not a step by setting the parameter to True or False.
- Section SETTINGS: some key parameters such as, name of the event, value of sigma, run simulations or use precomputed database, ... (see comments in workflow_input.json to know more about each specific parameter) 
- Section HPC CLUSTER: name of the cluster and login credentials. You do not need to fill this in if you run the workflow in "precomputed" mode (i.e., without running the simulations)

### Input data folder
TDT needs the access to a data directory `INPUT4PTF` that contains all the files necessary to run the workflow (e.g., precomputed scenarios, bathymetry grids, points of interests, ...).

On the HPC clusters, the paths to reach the folder are: 
- On Mercalli at INGV: `/scratch/projects/cat/SPTHA/INPUT4PTF`  
- On Galileo100: `/g100_work/DTGEO_T1/INPUT4PTF`   
- On Leonardo: `/leonardo_work/DTGEO_T1_2/INPUT4PTF`  

Alternatively, you can downloaded it from here: https://www.dropbox.com/scl/fo/v7i4j2i791zak5x2c5x70/h?dl=0&rlkey=adtifpsu5sb8zyzgzncamqtez. Be aware that it is about 150 GB.    

This data folder also includes a subfolder `earlyEst` where some JSON files containing the parameters of (some) past/synthetic events are stored, to use the TDT workflow in "event" mode by reading a file (see the option `--mode` and the description of the usage with the rabbit listener in the subsequent sections).

## Usage
1. Create the configutation file (just once). This is to configure some paths, file names, and it includes parameters that do not need to be changed each time you run the workflow. 
```
./cfg/do_main_ptf_config_file.py --cfg ./cfg/ptf_main.config --data_path path_to_input4ptf --work_path path_to_your_working_dir [--wf_path path_to_workflow] [--work_path_remote work_path_remote_remote] [--data_path_remote path_to_input4ptf_remote]
```
Mandatory arguments:
- `--cfg`: name of configuration file (file name w/ path)
- `--data_path`: path of the input data folder 
- `--work_path`: path of the working directory 

Optional arguments:
- `--wf_path`: path of the TDT workflow [default: ./]

Mandatory arguments for remote connection only:
- `--work_path_remote`: path of the working directory on the remote HPC cluster - only for remote connection [default: None]
- `--data_path_remote`: path of the input dta folder on the remote HPC cluster - only for remote connection [default: None]

2. Edit file `workflow_input.json` and modify it according to your settings/needs.

3. Run TDT by the command:
```
python workflow_main.py --cfg ./cfg/ptf_main.config --input_workflow path_to_your_working_dir/workflow_input.json --event_file path_to_event_json_files/eventname.json
```
Mandatory arguments:
- `--cfg`: name of configuration file (file name w/ path)
- `--input_workflow`:  name of the input json file (file name w/ path)
- `--event_file`: name of the event json file (file name w/ path)

Useful optional parameters can also be set from command line to overwrite the values defined in the `workflow_input.json` or in the configuration file:

- `--mode`: modality of receiving the event information [allowed values: event, rabbit; default: event]
- `--sigma`: scenario search radius around the epicenter [overwrite the .json input file; example: 2.0]
- `--percentiles`: list of selected percentiles calculated by the TDT from 1 to 99 [overwrite the .json input file; example: 5 50 95]
- `--hazard_mode`: mode to define the uncertainty of the hazard curve [overwrite the cfg file; allowed values: no_uncertainty, lognormal, lognormal_v1; default: lognormal_v1]

### Usage with Rabbit listener
There is an option to trigger the run of all the steps only once an event file is published in a message queue instead of reading from a static .json file. In this case, the flag `--mode rabbit` can be used when executing the workflow.   
This will start the workflow, but no steps will run until an event is published in the queue the rabbit is listening to. To publish an event in the queue, see README file in the `utils` directory. Presently, a "testing queue" is in use, only accessible from the INGV network.

[commented section]: #
[## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.]: #
[## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.]: #
[## License
For open source projects, say how it is licensed.]: #
[## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.]: #
