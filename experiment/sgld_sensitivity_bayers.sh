#!/bin/sh
# ---------------------------------------------------------------------
# LSF Directives for Bayesian Teacher Sensitivity Study on MNIST
# ---------------------------------------------------------------------
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00  # Walltime per trial (Increased for 1M iterations)
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"

# --- Job Array ---
# This submits all 16 trials for our study.
#BSUB -J "BayesAblation[1-16]"

# --- Output Files ---
# Use %J for the main job ID and %I for the array index for unique logs.
#BSUB -o experiment/sensitivity/logs/abl_%J_%I.out
#BSUB -e experiment/sensitivity/logs/abl_%J_%I.err

# ---------------------------------------------------------------------
# Environment Setup and Execution
# ---------------------------------------------------------------------
echo "--- Setting up environment for Job Array Index: $LSB_JOBINDEX ---"

# Create log directories if they don't exist
mkdir -p experiment/sensitivity/logs

# Load modules
module load cuda/12.2
module load python3/3.11.9

# Activate virtual environment
source /zhome/25/e/155273/masters/hpc_venv/bin/activate

# --- Hyperparameter Selection ---
# Set baseline parameters first (our best guess so far)
TAU_PARAM=15.0
POLY_A_PARAM=2.23e-5
POLY_B_PARAM=1150.0

# Use a 'case' statement to override one parameter based on the job index
case $LSB_JOBINDEX in
    1)
        # Trial 1: BASELINE RUN - uses the default parameters set above
        echo "Running BASELINE trial"
        ;;
    # --- GROUP 1: Varying Prior Precision (tau) ---
    2)  TAU_PARAM=5.0 ;;
    3)  TAU_PARAM=10.0 ;;
    4)  TAU_PARAM=20.0 ;;
    5)  TAU_PARAM=30.0 ;;
    6)  TAU_PARAM=50.0 ;;
    
    # --- GROUP 2: Varying LR Initial Value (via poly_a) ---
    7)  POLY_A_PARAM=1.00e-5 ;; # Lower initial LR
    8)  POLY_A_PARAM=5.00e-5 ;; # Higher initial LR
    9)  POLY_A_PARAM=1.00e-4 ;; # Even higher initial LR
    
    # --- GROUP 3: Varying LR Decay Shape (via poly_b) ---
    10) POLY_B_PARAM=500.0 ;;  # Faster initial decay
    11) POLY_B_PARAM=2000.0 ;; # Slower initial decay
    12) POLY_B_PARAM=5000.0 ;; # Very slow initial decay

    # --- GROUP 4: Extreme values / Sanity checks ---
    13) TAU_PARAM=100.0 ;;     # Very strong prior
    14) POLY_A_PARAM=1.00e-6 ;; # Very low initial LR
    15) POLY_B_PARAM=100.0 ;;   # Very fast decay
    16) TAU_PARAM=1.0 ;;        # Very weak prior

esac

# --- Run the Python script with the selected parameters ---
echo "--- Starting Trial $LSB_JOBINDEX ---"
echo "Parameters: T=1e6, tau=$TAU_PARAM, poly_a=$POLY_A_PARAM, poly_b=$POLY_B_PARAM"

python /zhome/25/e/155273/masters/experiment/experiment1_sensitivity_study.py \
    --iterations 1000000 \
    --tau $TAU_PARAM \
    --tr_poly_a $POLY_A_PARAM \
    --tr_poly_b $POLY_B_PARAM

echo "--- Trial $LSB_JOBINDEX Finished ---"
