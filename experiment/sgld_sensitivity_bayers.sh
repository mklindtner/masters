#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -J "BayesAblation[1-10]"
#BSUB -o experiment/tmp/abl_out_%J_%I.out
#BSUB -e experiment/tmp/abl_err_%J_%I.err

echo "--- Setting up environment for Job Array Index: $LSB_JOBINDEX ---"
module load cuda/12.2
module load python3/3.11.9
source /zhome/25/e/155273/masters/hpc_venv/bin/activate

# --- Hyperparameter Selection ---
TAU_PARAM=520.0
POLY_A_PARAM=2.23e-5
POLY_B_PARAM=1150.0
POLY_GAMMA_PARAM=0.55

# Use a 'case' statement to override one parameter
case $LSB_JOBINDEX in
    1) echo "Running BASELINE trial" ;;
    # Varying Prior Precision (tau)
    2)  TAU_PARAM=540.0 ;;
    3)  TAU_PARAM=560.0 ;;
    4)  TAU_PARAM=580.0 ;;
    5)  TAU_PARAM=600.0 ;;
    6)  TAU_PARAM=620.0 ;;
    7)  TAU_PARAM=640.0 ;;
    8)  TAU_PARAM=660.0 ;;
    9)  TAU_PARAM=680.0 ;;
    10) TAU_PARAM=700.0 ;;   
esac

RUN_NAME="T1M_tau${TAU_PARAM}_a${POLY_A_PARAM}_b${POLY_B_PARAM}_g${POLY_GAMMA_PARAM}"
OUTPUT_DIR="experiment/sensitivity/$RUN_NAME"
mkdir -p $OUTPUT_DIR

echo "--- Starting Trial $LSB_JOBINDEX ---"
echo "Saving all artifacts to: $OUTPUT_DIR"
echo "Parameters: T=1e6, tau=$TAU_PARAM, poly_a=$POLY_A_PARAM, poly_b=$POLY_B_PARAM, poly_g=$POLY_GAMMA_PARAM"

python /zhome/25/e/155273/masters/experiment/experiment1_sensitivity_study.py \
    --iterations 200000 \
    --tau $TAU_PARAM \
    --tr_poly_a $POLY_A_PARAM \
    --tr_poly_b $POLY_B_PARAM \
    --output_dir $OUTPUT_DIR # Pass the unique directory to Python

echo "--- Python Script Finished ---"

echo "Moving log files to final destination..."
mv experiment/tmp/single_run_$LSB_JOBID.out $OUTPUT_DIR/run_output.out
mv experiment/tmp/single_run_$LSB_JOBID.err $OUTPUT_DIR/run_error.err
echo "--- Trial $LSB_JOBINDEX Fully Completed ---"