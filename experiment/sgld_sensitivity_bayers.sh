#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -J "BayesAblation[1-16]"
#BSUB -o experiment/tmp/abl_out_%J_%I.out
#BSUB -e experiment/tmp/abl_err_%J_%I.err

echo "--- Setting up environment for Job Array Index: $LSB_JOBINDEX ---"
module load cuda/12.2
module load python3/3.11.9
source /zhome/25/e/155273/masters/hpc_venv/bin/activate

# --- Hyperparameter Selection ---
TAU_PARAM=15.0
POLY_A_PARAM=2.23e-5
POLY_B_PARAM=1150.0
POLY_GAMMA_PARAM=0.55

# Use a 'case' statement to override one parameter
case $LSB_JOBINDEX in
    1) echo "Running BASELINE trial" ;;
    # Varying Prior Precision (tau)
    2)  TAU_PARAM=5.0 ;;
    3)  TAU_PARAM=10.0 ;;
    4)  TAU_PARAM=15.0 ;;
    5)  TAU_PARAM=20.0 ;;
    6)  TAU_PARAM=30.0 ;;
    # Varying LR Initial Value (poly_a)
    7)  POLY_A_PARAM=1.00e-5 ;;
    8)  POLY_A_PARAM=5.00e-5 ;;
    9)  POLY_A_PARAM=1.00e-4 ;;
    # Varying LR Decay Shape (poly_b)
    10) POLY_B_PARAM=500.0 ;;
    11) POLY_B_PARAM=2000.0 ;;
    12) POLY_B_PARAM=5000.0 ;;
    # Extreme values
    13) TAU_PARAM=100.0 ;;
    14) POLY_A_PARAM=1.00e-6 ;;
    15) POLY_B_PARAM=100.0 ;;
    16) TAU_PARAM=1.0 ;;
esac

RUN_NAME="T1M_tau${TAU_PARAM}_a${POLY_A_PARAM}_b${POLY_B_PARAM}_g${POLY_GAMMA_PARAM}"
OUTPUT_DIR="experiment/sensitivity/$RUN_NAME"
mkdir -p $OUTPUT_DIR

echo "--- Starting Trial $LSB_JOBINDEX ---"
echo "Saving all artifacts to: $OUTPUT_DIR"
echo "Parameters: T=1e6, tau=$TAU_PARAM, poly_a=$POLY_A_PARAM, poly_b=$POLY_B_PARAM, poly_g=$POLY_GAMMA_PARAM"

python /zhome/25/e/155273/masters/experiment/experiment1_sensitivity_study.py \
    --iterations 1000000 \
    --tau $TAU_PARAM \
    --tr_poly_a $POLY_A_PARAM \
    --tr_poly_b $POLY_B_PARAM \
    --output_dir $OUTPUT_DIR # Pass the unique directory to Python

echo "--- Python Script Finished ---"

echo "Moving log files to final destination..."
mv experiment/tmp/single_run_$LSB_JOBID.out $OUTPUT_DIR/run_output.out
mv experiment/tmp/single_run_$LSB_JOBID.err $OUTPUT_DIR/run_error.err
echo "--- Trial $LSB_JOBINDEX Fully Completed ---"