#!/bin/sh
# ---------------------------------------------------------------------
# LSF Directives - Settings for the scheduler
# ---------------------------------------------------------------------

### -- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J thesisExp1

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --
#BSUB -W 4:00

### -- request RAM in GB. A100 nodes have lots of RAM, so we can ask for a good amount. --
#BSUB -R "rusage[mem=16GB]"

### -- specify your email for notifications --
##BSUB -u s205421@dtu.dk

### -- send notification at start & completion --
#BSUB -B
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
#BSUB -o experiment/hpc_output/gpu_%J.out
#BSUB -e experiment/hpc_err/gpu_%J.err
