#!/bin/sh
# ---------------------------------------------------------------------
# LSF Directives for a Single Bayesian Teacher Test Run
# This version saves all outputs to a unique, descriptive directory.
# ---------------------------------------------------------------------
#BSUB -q gpuv100
#BSUB -J TauTest
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00 # NB! I gotta remember to adjust for iterations
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"

# --- Set the specific hyperparameters for this test run ---
ITERATIONS=200000
TAU_PARAM=250.0
POLY_A_PARAM=2.23e-5
POLY_B_PARAM=1150.0

# --- THE KEY CHANGE: Define a unique directory for this run ---
# 1. Create a descriptive directory name from the parameters
RUN_NAME="T${ITERATIONS}_tau${TAU_PARAM}_a${POLY_A_PARAM}_b${POLY_B_PARAM}"

# 2. Define the full path for the output directory
OUTPUT_DIR="experiment/single_runs/$RUN_NAME"

### -- Specify the output and error file TO GO INSIDE the unique directory --
#BSUB -o $OUTPUT_DIR/run_output.out
#BSUB -e $OUTPUT_DIR/run_error.err

# ---------------------------------------------------------------------
# Environment Setup and Execution
# ---------------------------------------------------------------------
echo "--- Setting up environment ---"

# 3. Create the unique directory before the job starts
mkdir -p $OUTPUT_DIR

# Load modules
module load cuda/12.2
module load python3/3.11.9

# Activate virtual environment
source /zhome/25/e/155273/masters/hpc_venv/bin/activate

# --- Run the Python script with the selected parameters ---
echo "--- Starting Single Test Run ---"
echo "All outputs will be saved to: $OUTPUT_DIR"
echo "Parameters: T=$ITERATIONS, tau=$TAU_PARAM, poly_a=$POLY_A_PARAM, poly_b=$POLY_B_PARAM"
echo "Running on GPU:"
nvidia-smi

python /zhome/25/e/155273/masters/experiment/experiment1_sensitivity_study.py \
    --iterations $ITERATIONS \
    --tau $TAU_PARAM \
    --tr_poly_a $POLY_A_PARAM \
    --tr_poly_b $POLY_B_PARAM \
    --output_dir $OUTPUT_DIR # 4. Pass the directory path to Python

echo "--- Single Test Run Finished ---"