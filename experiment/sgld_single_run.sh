#!/bin/sh
#BSUB -q gpua40
#BSUB -J sgldPolyRun
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
# --------------------------------------------------------------------- 
# NB! Adjust for wallclock time when adjusting iterations!
# --------------------------------------------------------------------- 
#BSUB -W 7:20 
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o experiment/tmp/single_runs/single_run_%J.out
#BSUB -e experiment/tmp/single_runs/single_run_%J.err


# --- Set the specific hyperparameters for this test run ---
ITERATIONS=1000000
TAU_PARAM=1
POLY_A_PARAM=4.00e-6
POLY_B_PARAM=0
POLY_GAMMA_PARAM=0
BATCH_SIZE=100

PARAM_NAME="T${ITERATIONS}_tau${TAU_PARAM}_a${POLY_A_PARAM}_b${POLY_B_PARAM}_g${POLY_GAMMA_PARAM}"
RUN_NAME="${PARAM_NAME}_J${LSB_JOBID}"
OUTPUT_DIR="experiment/single_runs/$RUN_NAME"

# --- Setup Environment --- 
mkdir -p $OUTPUT_DIR

module load cuda/12.2
module load python3/3.11.9
source /zhome/25/e/155273/masters/hpc_venv/bin/activate

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
    --tr_poly_gamma $POLY_GAMMA_PARAM \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR


echo "Moving log files to final destination..."
mv experiment/tmp/single_runs/single_run_$LSB_JOBID.out $OUTPUT_DIR/run_output.out
mv experiment/tmp/single_runs/single_run_$LSB_JOBID.err $OUTPUT_DIR/run_error.err
echo "--- Single Test Run Finished ---"