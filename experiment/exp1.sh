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
###    This will place the files in the folders you requested.
#BSUB -o experiment/hpc_output/gpu_%J.out
#BSUB -e experiment/hpc_err/gpu_%J.err

# ---------------------------------------------------------------------
# Environment Setup and Execution
# ---------------------------------------------------------------------
module load cuda/12.2
module load python3/3.11.9

source /zhome/25/e/155273/masters/hpc_venv/bin/activate

# --- Sanity checks for easy debugging ---
# These commands will be logged to your .out file
echo "--- JOB DIAGNOSTICS ---"
echo "Job running on host: $HOSTNAME"
echo "GPU status:"
nvidia-smi
echo "-----------------------"
echo "Python version: $(python --version)"
echo "Python executable: $(which python)"
echo "PYTHONPATH is set to: $PYTHONPATH"
echo "-----------------------"

# --- Main command ---
# Run your experiment using the python -m flag for correct module resolution
echo "Starting Python script..."
python -m experiment.experiment1_training
echo "Script execution finished."